use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CacheRequest {
    Get { key: serde_json::Value },
    Put { key: serde_json::Value, value: serde_json::Value },
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum CacheResponse {
    Found(serde_json::Value),
    NotFound,
}

pub async fn cache_get(address: &str, key: &serde_json::Value) -> Result<Option<serde_json::Value>, String> {
    let mut stream = tokio::net::TcpStream::connect(address)
        .await
        .map_err(|error| format!("failed to connect to cache server at '{address}': {}", error))?;

    let cache_request = CacheRequest::Get { key: key.clone() };
    let mut request = serde_json::to_string(&cache_request).map_err(|error| format!("failed to serialize request object: {}", error))?;

    request.push('\n');
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|error| format!("failed to send request: {}", error))?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|error| format!("failed to read response: {}", error))?;

    let response_json: CacheResponse = serde_json::from_slice(&response).map_err(|error| format!("invalid JSON response: {}", error))?;

    match response_json {
        CacheResponse::Found(value) => Ok(Some(value)),
        CacheResponse::NotFound => Ok(None),
    }
}

pub async fn cache_put(addr: &str, key: &serde_json::Value, value: &serde_json::Value) -> Result<(), String> {
    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .map_err(|error| format!("failed to connect to cache server: {}", error))?;

    let cache_request = CacheRequest::Put {
        key: key.clone(),
        value: value.clone(),
    };
    let mut request = serde_json::to_string(&cache_request).map_err(|error| format!("failed to serialize CacheRequest: {}", error))?;
    request.push('\n');

    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|error| format!("failed to send request: {}", error))?;
    core::mem::drop(request);

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|error| format!("failed to read response: {}", error))?;

    if response.is_empty() {
        Ok(())
    } else {
        Err("unexpected response from the cache server PUT handler".into())
    }
}
