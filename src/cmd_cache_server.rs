use ahash::AHashMap as HashMap;
use core::time::Duration;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::cache::{CacheRequest, CacheResponse};
use crate::utils::{LinesWithRange, UniqueHash, hash_value, mmap_read};

fn load_server_cache(blob: &[u8]) -> Result<HashMap<UniqueHash, Range<usize>>, String> {
    let mut cache = HashMap::new();
    for (line, range) in LinesWithRange::new(blob) {
        if line.is_empty() {
            continue;
        }

        let cache_entry: serde_json::Value = match serde_json::from_slice(line) {
            Ok(value) => value,
            Err(error) => return Err(format!("failed to parse cache entry: {}", error)),
        };

        let serde_json::Value::Object(obj) = cache_entry else {
            return Err("cache entry is not an object".to_string());
        };

        let Some(raw_request) = obj.get("request") else {
            return Err("cache entry missing raw_request".to_string());
        };

        if !obj.contains_key("raw_response") {
            return Err("cache entry missing raw_response".to_string());
        };

        let request_hash = hash_value(&raw_request);
        cache.insert(request_hash, range);
    }

    Ok(cache)
}

fn get_from_storage(key_hash: UniqueHash, blob: &[u8], map: &RwLock<HashMap<UniqueHash, Range<usize>>>) -> Option<serde_json::Value> {
    let range = {
        let map = map.read().unwrap();
        map.get(&key_hash)?.clone()
    };

    let line = blob.get(range.clone())?;
    let cache_entry: serde_json::Value = serde_json::from_slice(line).ok()?;
    let serde_json::Value::Object(mut obj) = cache_entry else {
        return None;
    };
    obj.remove("raw_response")
}

async fn main_cache_server_impl(
    address: &str,
    cache_path: Option<PathBuf>,
) -> Result<Pin<Box<dyn futures::Future<Output = ()> + Send + 'static>>, String> {
    struct Cache {
        blob: Option<memmap2::Mmap>,
        map_cold: RwLock<HashMap<UniqueHash, Range<usize>>>,
        map_hot: RwLock<HashMap<UniqueHash, serde_json::Value>>,
        fp: Option<Mutex<std::io::BufWriter<std::fs::File>>>,
    }

    let mut cache = Cache {
        blob: None,
        map_cold: RwLock::new(HashMap::new()),
        map_hot: RwLock::new(HashMap::new()),
        fp: None,
    };

    let verbose = false;
    if let Some(ref cache_path) = cache_path {
        let mut output_needs_newline = false;

        if cache_path.exists() {
            eprintln!("INFO: Loading cache: {}", cache_path.display());
            let blob = mmap_read(&cache_path)?;
            if !blob.is_empty() && blob.last().copied() != Some(b'\n') {
                output_needs_newline = true;
            }

            let map_cold = load_server_cache(&blob)?;
            eprintln!("INFO: Loaded cache: {} entries", map_cold.len());

            cache.map_cold = RwLock::new(map_cold);
            cache.blob = Some(blob);
        } else {
            eprintln!(
                "INFO: Cache file {} does not exist; starting with empty cache",
                cache_path.display()
            );
        }

        let fp = std::fs::OpenOptions::new()
            .read(false)
            .write(true)
            .append(true)
            .truncate(false)
            .create(true)
            .open(&cache_path)
            .map_err(|error| format!("failed to open {} for writing: {error}", cache_path.display()))?;

        let mut fp = std::io::BufWriter::new(fp);
        if output_needs_newline {
            fp.write_all(b"\n")
                .map_err(|error| format!("failed to write a newline to {}: {error}", cache_path.display()))?;

            fp.flush()
                .map_err(|error| format!("failed to write a newline to {}: {error}", cache_path.display()))?;
        }

        cache.fp = Some(Mutex::new(fp));
    } else {
        eprintln!("INFO: Running in memory-only mode");
    }

    let cache = Arc::new(cache);
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| format!("failed to bind to address: {}", error))?;

    eprintln!("INFO: Cache server listening on {address}");
    let task = async move {
        let mut flush_interval = tokio::time::interval(Duration::from_secs(3));
        loop {
            let accept_result = tokio::select! {
                client = listener.accept() => {
                    client
                }
                _ = flush_interval.tick() => {
                    if let Some(ref fp) = cache.fp {
                        let _ = fp.lock().unwrap().flush();
                    }

                    continue;
                }
            };

            let (mut socket, addr) = match accept_result {
                Ok(result) => result,
                Err(error) => {
                    eprintln!("ERROR: Failed to accept connection: {}", error);
                    continue;
                }
            };

            let cache = cache.clone();

            tokio::spawn(async move {
                if verbose {
                    eprintln!("INFO: Connection from {}", addr);
                }

                let mut buffer = vec![0u8; 1024];
                let mut request_data = Vec::new();

                loop {
                    match socket.read(&mut buffer).await {
                        Ok(0) => {
                            if verbose {
                                eprintln!("INFO: Connection from {} closed", addr);
                            }

                            return;
                        }
                        Ok(n) => {
                            request_data.extend_from_slice(&buffer[..n]);

                            // Check if we have a complete request (ends with newline)
                            if request_data.contains(&b'\n') {
                                break;
                            }
                        }
                        Err(error) => {
                            if verbose {
                                eprintln!("ERROR: Failed to read from {}: {}", addr, error);
                            }
                            return;
                        }
                    }
                }

                let request: CacheRequest = match serde_json::from_slice(&request_data) {
                    Ok(request) => request,
                    Err(error) => {
                        if verbose {
                            eprintln!("ERROR: Failed to parse request: {error}");
                        }

                        return;
                    }
                };

                if verbose {
                    eprintln!("INFO: Received request from {}: {:?}", addr, request);
                }

                let response = match request {
                    CacheRequest::Get { key } => {
                        let key_hash = hash_value(&key);

                        let mut value = {
                            let map = cache.map_hot.read().unwrap();
                            map.get(&key_hash).cloned()
                        };

                        if value.is_none() {
                            if let Some(ref blob) = cache.blob {
                                value = get_from_storage(key_hash, &blob, &cache.map_cold);
                            }
                        }

                        match value {
                            Some(value) => {
                                let response = CacheResponse::Found(value.clone());
                                match serde_json::to_string(&response) {
                                    Ok(mut json_str) => {
                                        json_str.push('\n');
                                        Ok(json_str)
                                    }
                                    Err(error) => Err(format!("failed to serialize response: {}", error)),
                                }
                            }
                            None => {
                                let response = CacheResponse::NotFound;
                                match serde_json::to_string(&response) {
                                    Ok(json_str) => Ok(format!("{}\n", json_str)),
                                    Err(error) => Err(format!("failed to serialize response: {}", error)),
                                }
                            }
                        }
                    }
                    CacheRequest::Put { key, value } => {
                        let key_hash = hash_value(&key);

                        let value_clone = value.clone();
                        let mut map_hot = cache.map_hot.write().unwrap();
                        let write_to_file = match map_hot.entry(key_hash) {
                            std::collections::hash_map::Entry::Occupied(_) => false,
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                entry.insert(value_clone);
                                true
                            }
                        };
                        core::mem::drop(map_hot);

                        if write_to_file {
                            if let Some(ref fp) = cache.fp {
                                let entry = serde_json::json!({
                                    "request": key,
                                    "raw_response": value,
                                });
                                if let Ok(mut entry) = serde_json::to_string(&entry) {
                                    entry.push('\n');

                                    let mut fp = fp.lock().unwrap();
                                    let result = fp.write_all(entry.as_bytes());
                                    core::mem::drop(fp);

                                    if let Err(error) = result {
                                        eprintln!("ERROR: Failed to write to cache: {error}");
                                    }
                                }
                            }
                        }

                        Ok("".into())
                    }
                };

                let response = match response {
                    Ok(response) => {
                        if verbose {
                            eprintln!("INFO: Sending response: {}", response);
                        }

                        response
                    }
                    Err(error) => {
                        if verbose {
                            eprintln!("ERROR: Failed to process request: {}", error);
                        }

                        format!("ERROR: {}\n", error)
                    }
                };

                if let Err(error) = socket.write_all(response.as_bytes()).await {
                    if verbose {
                        eprintln!("ERROR: Failed to write to {}: {}", addr, error);
                    }
                }
            });
        }
    };

    Ok(Box::pin(task))
}

pub async fn main_cache_server(address: &str, cache_path: Option<PathBuf>) -> Result<(), String> {
    Ok(main_cache_server_impl(address, cache_path).await?.await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_server() {
        crate::small_runtime().block_on(test_cache_server_impl()).unwrap();
    }

    async fn test_cache_server_impl() -> Result<(), String> {
        let address = "127.0.0.1:9998";
        let task = main_cache_server_impl(address, None).await?;
        tokio::spawn(task);

        let test_key = serde_json::json!("test_key");
        let test_value = serde_json::json!("test_value");

        crate::cache::cache_put(&address, &test_key, &test_value).await?;

        let retrieved_value = crate::cache::cache_get(&address, &test_key).await?;
        assert_eq!(retrieved_value, Some(test_value));

        let non_existent_key = serde_json::json!("non_existent_key");
        let retrieved_value = crate::cache::cache_get(&address, &non_existent_key).await?;
        assert_eq!(retrieved_value, None);

        Ok(())
    }
}
