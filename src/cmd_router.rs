use bytes::Bytes;
use futures::StreamExt;
use futures::TryStreamExt;
use http::{Request, Response, StatusCode, header};
use http_body::Body as HttpBody;
use http_body::Frame;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::OnOff;
use crate::cache::Cache;
use crate::openai_client::{self, FinishReason, ResponseError};

fn add_response_to_cache(
    cache: &Cache,
    (cache_key, cache_key_with_token_limit): (serde_json::Value, Option<serde_json::Value>),
    response: openai_client::Response,
) {
    match response.obj {
        Ok(Ok(ref response_ok)) => {
            let cache_key = match response_ok.finish_reason {
                Some(FinishReason::Stop) => Some(cache_key),
                Some(FinishReason::Length) => cache_key_with_token_limit,
                None => None,
            };

            if let Some(cache_key) = cache_key {
                if let Some(raw_response) = response.raw_json() {
                    cache.put(cache_key, raw_response);
                }
            }
        }
        Ok(Err(_)) => {}
        Err(error) => {
            eprintln!("WARN: failed to cache response: {error}");
        }
    }
}

struct ProxyState {
    backend_url: String,
    cache: Arc<Cache>,
    force_non_streaming: bool,
    should_cache_streaming: bool,
    should_cache_non_streaming: bool,
}

type ResultResponse = Result<Response<Pin<Box<dyn HttpBody<Data = Bytes, Error = std::convert::Infallible> + Send>>>, String>;

fn response_error(status: StatusCode, response: ResponseError) -> ResultResponse {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Box::pin(Full::from(serde_json::to_string(&response).unwrap())) as _)
        .map_err(|e| e.to_string())
}

fn response_not_found() -> ResultResponse {
    response_error(StatusCode::NOT_FOUND, ResponseError::not_found())
}

fn response_bad_request(message: String) -> ResultResponse {
    response_error(StatusCode::BAD_REQUEST, ResponseError::bad_request(message))
}

fn response_backend_error(message: String) -> ResultResponse {
    return response_error(StatusCode::INTERNAL_SERVER_ERROR, ResponseError::internal_server_error(message));
}

fn response_ok_str_json(value: String) -> ResultResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Box::pin(Full::from(value)) as _)
        .map_err(|e| e.to_string())
}

fn response_ok(value: serde_json::Value) -> ResultResponse {
    response_ok_str_json(serde_json::to_string(&value).unwrap())
}

async fn proxy_request(state: &Arc<ProxyState>, path: &str, raw_request: Vec<u8>) -> ResultResponse {
    let raw_request = match String::from_utf8(raw_request) {
        Ok(raw_request) => raw_request,
        Err(_) => return response_bad_request("invalid UTF-8".into()),
    };

    if path == "/v1/models" {
        let response = proxy_models_request(state).await?;
        let status = response.status();
        let body = response.into_body();
        let response = Response::builder()
            .status(status)
            .body(Box::pin(body) as _)
            .map_err(|error| error.to_string())?;

        return Ok(response);
    }

    if path != "/v1/chat/completions" && path != "/v1/completions" {
        return response_not_found();
    }

    let path = path.strip_prefix("/v1").unwrap();

    let mut request: serde_json::Value = match serde_json::from_str(&raw_request) {
        Ok(request) => request,
        Err(_) => return response_bad_request("invalid JSON".into()),
    };

    let serde_json::Value::Object(ref mut body_obj) = request else {
        return response_bad_request("expected an object".into());
    };

    let is_streaming = body_obj.get("stream").and_then(|value| value.as_bool()).unwrap_or(false);
    let is_streaming = if state.force_non_streaming && is_streaming {
        body_obj.remove("stream");
        false
    } else {
        is_streaming
    };

    let should_cache = if is_streaming {
        state.should_cache_streaming
    } else {
        state.should_cache_non_streaming
    };

    let mut cache_key = {
        let mut cache_key = body_obj.clone();

        cache_key.retain(|_, value| !value.is_null());
        cache_key.remove("stream");
        cache_key.remove("cache_prompt");
        cache_key.remove("priority");

        let cache_key_with_token_limit = if cache_key.contains_key("max_tokens") {
            let cache_key_with_token_limit = cache_key.clone();
            cache_key.remove("max_tokens");
            Some(serde_json::Value::Object(cache_key_with_token_limit))
        } else {
            None
        };

        let cache_key = serde_json::Value::Object(cache_key);
        if let Some(response) = state.cache.get(&cache_key) {
            return response_ok(response);
        }

        if let Some(ref cache_key_with_token_limit) = cache_key_with_token_limit {
            if let Some(response) = state.cache.get(cache_key_with_token_limit) {
                return response_ok(response);
            }
        }

        if should_cache {
            Some((cache_key, cache_key_with_token_limit))
        } else {
            None
        }
    };

    let client = reqwest::Client::new();
    let request_url = format!("{}{}", state.backend_url, path);
    let backend_response = match client
        .post(&request_url)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            eprintln!("ERROR: Failed to send a request to the backend: {}", error);
            return response_backend_error(format!("HTTP error: {error}"));
        }
    };

    let status = backend_response.status();
    let status_u16 = status.as_u16();
    let http_status = StatusCode::from_u16(status_u16).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = backend_response.headers().get(reqwest::header::CONTENT_TYPE).cloned();

    if !is_streaming || http_status != StatusCode::OK {
        let raw_response = match backend_response.bytes().await {
            Ok(raw_response) => raw_response,
            Err(error) => {
                eprintln!("ERROR: Failed to read backend response: {}", error);
                return response_backend_error(format!("failed to fetch response from the backend: {error}"));
            }
        };

        if http_status == StatusCode::OK {
            if let Some(cache_key) = cache_key {
                let raw_response = match std::str::from_utf8(&raw_response) {
                    Ok(raw_response) => raw_response,
                    Err(_) => {
                        return response_backend_error(format!("response from the backend is not valid UTF-8"));
                    }
                };

                let response = openai_client::Response::from_raw(&raw_response, None);
                add_response_to_cache(&state.cache, cache_key, response);
            }
        }

        let mut response = Response::builder().status(http_status);
        if let Some(content_type) = content_type {
            response = response.header(reqwest::header::CONTENT_TYPE, content_type);
        }

        let response = response
            .body(Pin::new(Box::new(Full::from(raw_response))) as _)
            .map_err(|error| error.to_string())?;

        return Ok(response);
    }

    let mut stream = crate::openai_client::handle_streaming(backend_response)?;

    use openai_client::StreamingChunk;
    let chunk = match stream.next().await {
        Some(StreamingChunk::Payload(payload)) => {
            return response_ok_str_json(payload);
        }
        None | Some(StreamingChunk::Finish) => {
            return response_backend_error(format!("unexpected end of stream"));
        }
        Some(StreamingChunk::Chunk(chunk)) => StreamingChunk::Chunk(chunk),
        Some(StreamingChunk::Error(error)) => {
            return response_backend_error(format!("backend error: {error}"));
        }
    };

    let (tx, rx) = mpsc::channel::<Result<Bytes, Box<dyn std::error::Error + Send + Sync>>>(128);
    let boxed_body: Pin<Box<dyn HttpBody<Data = Bytes, Error = std::convert::Infallible> + Send>> = Box::pin(StreamBody::new(
        ReceiverStream::new(rx).map_ok(Frame::data).map_err(|_| unreachable!()),
    ));

    let response = Response::builder()
        .status(http_status)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(boxed_body)
        .map_err(|e| e.to_string())?;

    let mut delta_state = openai_client::DeltaState::default();
    let cache = state.cache.clone();

    tokio::spawn(async move {
        let mut chunk_opt: Option<StreamingChunk> = Some(chunk);
        while let Some(chunk) = chunk_opt.take() {
            match chunk {
                StreamingChunk::Payload(chunk) | StreamingChunk::Chunk(chunk) => {
                    if cache_key.is_some() {
                        match serde_json::from_str(&chunk) {
                            Ok(chunk) => {
                                if let Err(error) = delta_state.apply(&chunk) {
                                    eprintln!("WARN: Failed to apply delta state for caching: {error}");
                                    cache_key = None;
                                }
                            }
                            Err(error) => {
                                eprintln!("WARN: Failed to deserialize chunk for caching: {error}");
                                cache_key = None;
                            }
                        }
                    }

                    let chunk = format!("data: {chunk}\n\n");
                    if tx.send(Ok(chunk.into())).await.is_err() {
                        break;
                    }
                }
                StreamingChunk::Finish => {
                    if let Some(cache_key) = cache_key {
                        match delta_state.finalize() {
                            Ok(response) => {
                                let response = openai_client::Response::from_raw(&serde_json::to_string(&response).unwrap(), None);
                                add_response_to_cache(&cache, cache_key, response);
                            }
                            Err(error) => {
                                eprintln!("WARN: Failed to finalize delta state for caching:: {error}");
                            }
                        }
                    }

                    if tx.send(Ok("data: [DONE]\n\n".into())).await.is_err() {
                        break;
                    }

                    break;
                }
                StreamingChunk::Error(error) => {
                    eprintln!("WARN: Encountered an error while streaming: {error}");
                    return;
                }
            }

            chunk_opt = stream.next().await;
        }
    });

    Ok(response)
}

async fn proxy_models_request(state: &Arc<ProxyState>) -> Result<Response<Full<Bytes>>, String> {
    let client = reqwest::Client::new();
    let backend_base = if state.backend_url.ends_with("/v1") {
        state.backend_url.trim_end_matches("/v1").to_string()
    } else {
        state.backend_url.trim_end_matches("/").to_string()
    };

    let backend_response = match client.get(&format!("{}/v1/models", backend_base)).send().await {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("ERROR: Failed to fetch models from backend: {}", e);
            let response = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::from(
                    serde_json::json!({"error": format!("Backend error: {}", e)}).to_string(),
                ))
                .map_err(|e| e.to_string())?;
            return Ok(response);
        }
    };

    let status = backend_response.status();
    let status_u16 = status.as_u16();
    let http_status = StatusCode::from_u16(status_u16).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let response_body = match backend_response.text().await {
        Ok(text) => text,
        Err(e) => {
            eprintln!("ERROR: Failed to read backend response: {}", e);
            let response = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::from(
                    serde_json::json!({"error": format!("Backend error: {}", e)}).to_string(),
                ))
                .map_err(|e| e.to_string())?;
            return Ok(response);
        }
    };

    let response = Response::builder()
        .status(http_status)
        .body(Full::from(response_body))
        .map_err(|e| e.to_string())?;

    Ok(response)
}

async fn handle_connection(stream: tokio::net::TcpStream, state: Arc<ProxyState>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let io = TokioIo::new(stream);

    let svc = hyper::service::service_fn(move |req: Request<hyper::body::Incoming>| {
        let state = state.clone();
        async move {
            let path = req.uri().path().to_string();
            let method = req.method().clone();
            let body = match req.into_body().collect().await {
                Ok(collected) => collected.to_bytes().to_vec(),
                Err(e) => {
                    let response = Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Pin::new(Box::new(Full::from(format!("Failed to collect body: {}", e)))) as _)
                        .map_err(|e| e.to_string())?;
                    return Ok(response);
                }
            };

            if method != http::Method::POST {
                let response = Response::builder()
                    .status(StatusCode::METHOD_NOT_ALLOWED)
                    .body(Pin::new(Box::new(Full::from("Method not allowed"))) as _)
                    .map_err(|e| e.to_string())?;
                return Ok(response);
            }

            proxy_request(&state, &path, body).await
        }
    });

    let h1_conn = hyper::server::conn::http1::Builder::new().serve_connection(io, svc);

    if let Err(e) = h1_conn.await {
        if !e.is_closed() && !e.is_canceled() && !e.is_incomplete_message() {
            eprintln!("ERROR: Connection error: {}", e);
        }
    }

    Ok(())
}

#[derive(clap::Args)]
pub struct RouterArgs {
    #[clap(long, default_value = "127.0.0.1")]
    host: String,

    #[clap(long, default_value_t = 9002)]
    port: u32,

    #[clap(long)]
    cache_path: Option<PathBuf>,

    #[clap(long)]
    force_non_streaming: bool,

    #[clap(long, default_value = "on")]
    cache_streaming: OnOff,

    #[clap(long, default_value = "on")]
    cache_non_streaming: OnOff,

    #[clap(long, default_value = "http://127.0.0.1:9001/v1")]
    backend: String,
}

pub async fn main_proxy_server(
    RouterArgs {
        host,
        port,
        cache_path,
        force_non_streaming,
        cache_streaming,
        cache_non_streaming,
        mut backend,
    }: RouterArgs,
) -> Result<(), String> {
    let listen_address = format!("{host}:{port}");
    let mut cache = Cache::new("raw_request".into());
    if let Some(ref cache_path) = cache_path {
        cache.acquire(cache_path)?;
    } else {
        eprintln!("INFO: Running in memory-only mode");
    }

    let addr: SocketAddr = listen_address
        .parse()
        .map_err(|e| format!("failed to parse listen address: {}", e))?;

    while backend.ends_with('/') {
        backend.pop();
    }

    eprintln!("INFO: Proxy server listening on {}", addr);
    eprintln!("INFO: Proxying to {}", backend);

    let cache = Arc::new(cache);
    let state = Arc::new(ProxyState {
        backend_url: backend,
        cache,
        force_non_streaming,
        should_cache_streaming: cache_streaming.into(),
        should_cache_non_streaming: cache_non_streaming.into(),
    });

    let listener = TcpListener::bind(addr).await.map_err(|e| e.to_string())?;
    loop {
        let mut flush_interval = tokio::time::interval(core::time::Duration::from_secs(3));
        let accept_result = tokio::select! {
            client = listener.accept() => {
                client
            }
            _ = flush_interval.tick() => {
                state.cache.flush();
                continue;
            }
        };

        let (stream, addr) = accept_result.map_err(|e| e.to_string())?;
        let state = state.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state).await {
                eprintln!("ERROR: Failed to handle connection from {}: {}", addr, e);
            }
        });
    }
}
