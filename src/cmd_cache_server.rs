use core::time::Duration;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::cache::Cache;
use crate::cache_client::{CacheRequest, CacheResponse};

async fn main_cache_server_impl(
    address: &str,
    cache_path: Option<PathBuf>,
) -> Result<Pin<Box<dyn futures::Future<Output = ()> + Send + 'static>>, String> {
    let mut cache = Cache::new("request".into());
    let verbose = false;
    if let Some(ref cache_path) = cache_path {
        cache.acquire(cache_path)?;
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
                    cache.flush();
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
                        let value = cache.get(&key);
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
                        cache.put(key, value);
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

        crate::cache_client::cache_put(&address, &test_key, &test_value).await?;

        let retrieved_value = crate::cache_client::cache_get(&address, &test_key).await?;
        assert_eq!(retrieved_value, Some(test_value));

        let non_existent_key = serde_json::json!("non_existent_key");
        let retrieved_value = crate::cache_client::cache_get(&address, &non_existent_key).await?;
        assert_eq!(retrieved_value, None);

        Ok(())
    }
}
