use crate::utils::{LinesWithRange, UniqueHash, hash_value, mmap_read};
use ahash::AHashMap as HashMap;
use core::ops::Range;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, RwLock};

fn load_server_cache(request_key: &str, blob: &[u8]) -> Result<HashMap<UniqueHash, Range<usize>>, String> {
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

        let Some(raw_request) = obj.get(request_key) else {
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

pub struct Cache {
    blob: Option<memmap2::Mmap>,
    map_cold: RwLock<HashMap<UniqueHash, Range<usize>>>,
    map_hot: RwLock<HashMap<UniqueHash, serde_json::Value>>,
    fp: Option<Mutex<std::io::BufWriter<std::fs::File>>>,
    request_key: String,
}

impl Cache {
    pub fn new(request_key: String) -> Self {
        Cache {
            blob: None,
            map_cold: RwLock::new(HashMap::new()),
            map_hot: RwLock::new(HashMap::new()),
            fp: None,
            request_key,
        }
    }

    pub fn acquire(&mut self, cache_path: &Path) -> Result<(), String> {
        let mut output_needs_newline = false;

        if cache_path.exists() {
            eprintln!("INFO: Loading cache: {}", cache_path.display());
            let blob = mmap_read(&cache_path)?;
            if !blob.is_empty() && blob.last().copied() != Some(b'\n') {
                output_needs_newline = true;
            }

            let map_cold = load_server_cache(&self.request_key, &blob)?;
            eprintln!("INFO: Loaded cache: {} entries", map_cold.len());

            self.map_cold = RwLock::new(map_cold);
            self.blob = Some(blob);
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

        self.fp = Some(Mutex::new(fp));
        Ok(())
    }

    pub fn flush(&self) {
        if let Some(ref fp) = self.fp {
            let _ = fp.lock().unwrap().flush();
        }
    }

    pub fn get(&self, key: &serde_json::Value) -> Option<serde_json::Value> {
        let key_hash = hash_value(&key);

        let mut value = {
            let map = self.map_hot.read().unwrap();
            map.get(&key_hash).cloned()
        };

        if value.is_none() {
            if let Some(ref blob) = self.blob {
                value = get_from_storage(key_hash, &blob, &self.map_cold);
            }
        }

        value
    }

    pub fn put(&self, key: serde_json::Value, value: serde_json::Value) {
        let key_hash = hash_value(&key);

        let value_clone = value.clone();
        let mut map_hot = self.map_hot.write().unwrap();
        let write_to_file = match map_hot.entry(key_hash) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(value_clone);
                true
            }
        };
        core::mem::drop(map_hot);

        if write_to_file {
            if let Some(ref fp) = self.fp {
                let mut entry = serde_json::Map::new();
                entry.insert(self.request_key.clone(), key);
                entry.insert("raw_response".into(), value);

                let entry = serde_json::Value::Object(entry);
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
    }
}
