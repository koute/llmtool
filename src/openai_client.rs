use futures::prelude::*;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
pub struct F32(f32);

impl Eq for F32 {}
impl Ord for F32 {
    fn cmp(&self, rhs: &F32) -> core::cmp::Ordering {
        self.0.partial_cmp(&rhs.0).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub url: String,
    pub api_key: String,
    pub providers: Vec<String>,
    pub allow_fallbacks: bool,
}

impl Endpoint {
    pub fn new(url: String) -> Self {
        Self {
            url,
            api_key: String::new(),
            providers: Vec::new(),
            allow_fallbacks: true,
        }
    }

    pub fn local(port: u32) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"))
    }

    pub fn openrouter(api_key: String) -> Self {
        Endpoint {
            url: "https://openrouter.ai/api".into(),
            api_key,
            providers: Vec::new(),
            allow_fallbacks: true,
        }
    }

    pub fn is_local(&self) -> bool {
        self.url.contains("/127.0.0.1:") || self.url.contains("/localhost:")
    }

    fn completion_url(&self) -> String {
        format!("{}/v1/completions", self.url)
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.url)
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.url)
    }
}

fn is_false(value: &bool) -> bool {
    !value
}

fn is_true(value: &bool) -> bool {
    *value
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize)]
pub struct RawProvider {
    pub order: Vec<String>,
    #[serde(skip_serializing_if = "is_true")]
    pub allow_fallbacks: bool,
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawJsonSchema {
    pub name: String,
    pub schema: Value,
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    pub json_schema: Option<RawJsonSchema>,
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawStructuredOutputs {
    pub choice: Option<Vec<String>>,
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawGenerationArgs {
    pub model: String,

    // General.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "is_false")]
    pub echo: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub cache_prompt: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<RawProvider>,

    #[serde(skip_serializing_if = "is_false")]
    pub ban_eos_token: bool,

    // Sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<F32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<F32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<F32>,

    // Penalties.
    #[serde(skip_serializing_if = "is_false")]
    pub penalize_nl: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<F32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<F32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<F32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty_range: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<RawResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<RawStructuredOutputs>,

    #[serde(skip_serializing_if = "is_false")]
    pub logprobs: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

const TIMEOUT: core::time::Duration = core::time::Duration::from_secs(60 * 60);

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct ResponseError {
    pub code: u32,
    pub message: String,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub param: Option<String>,
}

#[derive(Copy, Clone, PartialEq, Eq, serde::Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Length,
    Stop,
}

#[derive(Clone, serde::Deserialize, Debug)]
pub struct Usage {
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    // pub total_tokens: u64,
}

#[derive(serde::Deserialize, Debug)]
struct RawMessage {
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct RawDelta {
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct RawChoiceError {
    code: u32,
    message: String,
    // metadata: Option<serde_json::Value>,
}

#[derive(serde::Deserialize, Debug)]
struct RawChoice {
    // index: u64,
    finish_reason: Option<FinishReason>,

    text: Option<String>,
    message: Option<RawMessage>,
    delta: Option<RawDelta>,

    error: Option<RawChoiceError>,
}

#[derive(serde::Deserialize, Debug)]
struct RawResponseOk {
    choices: Vec<RawChoice>,
    usage: Option<Usage>,
    // id: String,
    // object: String,
    model: String,
    // system_fingerprint: Option<String>,
    // created: u64,
}

#[derive(Clone, Debug)]
pub struct ResponseOk {
    pub finish_reason: Option<FinishReason>,
    pub text: String,
    pub reasoning_content: Option<String>,
    pub usage: Option<Usage>,
    pub model: String,
}

#[derive(Debug)]
pub struct Response {
    pub obj: Result<Result<ResponseOk, ResponseError>, String>,
    pub raw: Option<String>,
    pub original_request: Option<Arc<String>>,
}

impl Response {
    pub fn raw_json(&self) -> Option<serde_json::Value> {
        let raw = self.raw.as_ref()?;
        serde_json::from_str(&raw).ok()
    }

    pub fn raw_request_json(&self) -> Option<serde_json::Value> {
        let req = self.original_request.as_ref()?;
        serde_json::from_str(&req).ok()
    }

    pub fn from_raw(raw_string: &str, raw_request: Arc<String>) -> Self {
        parse_response(raw_string, raw_request)
    }
}

fn parse_response(raw_string: &str, raw_request: Arc<String>) -> Response {
    let raw_value: Result<Value, _> = serde_json::from_str(raw_string);
    let raw = Some(raw_string.to_owned());
    let raw_value = match raw_value {
        Ok(raw_value) => raw_value,
        Err(error) => {
            return Response {
                obj: Err(format!("response is not valid JSON: {error}")),
                raw,
                original_request: Some(raw_request),
            };
        }
    };

    let Some(value) = raw_value.as_object() else {
        return Response {
            obj: Err(format!("response is not an object")),
            raw,
            original_request: Some(raw_request),
        };
    };

    if let Some(error) = value.get("error") {
        let error: Result<ResponseError, _> = serde_json::from_value(error.clone());
        return Response {
            obj: match error {
                Ok(error) => Ok(Err(error)),
                Err(error) => Err(format!("failed to parse 'error': {error}")),
            },
            raw,
            original_request: Some(raw_request),
        };
    }

    let response: Result<RawResponseOk, _> = serde_json::from_value(raw_value.clone());
    Response {
        obj: match response {
            Ok(response) => {
                let Some(choice) = response.choices.into_iter().next() else {
                    return Response {
                        obj: Err(format!("response is missing choices")),
                        raw,
                        original_request: Some(raw_request),
                    };
                };

                if let Some(error) = choice.error {
                    return Response {
                        obj: Err(format!("response returned an error: code {}: {}", error.code, error.message)),
                        raw,
                        original_request: Some(raw_request),
                    };
                }

                let (text, reasoning_content) = if let Some(message) = choice.message {
                    (
                        message.content.unwrap_or(String::new()),
                        message.reasoning_content.or(message.reasoning),
                    )
                } else if let Some(text) = choice.text {
                    (text, None)
                } else if let Some(delta) = choice.delta {
                    (delta.content.unwrap_or(String::new()), delta.reasoning_content.or(delta.reasoning))
                } else {
                    return Response {
                        obj: Err(format!("response is missing 'text' and 'message'")),
                        raw,
                        original_request: Some(raw_request),
                    };
                };

                let response = ResponseOk {
                    finish_reason: choice.finish_reason,
                    text,
                    reasoning_content: reasoning_content,
                    model: response.model,
                    usage: response.usage,
                };

                Ok(Ok(response))
            }
            Err(error) => Err(format!("failed to parse response: {error}")),
        },
        raw,
        original_request: Some(raw_request),
    }
}

#[derive(Clone, serde::Serialize)]
pub struct GenerationArgs {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty_range: Option<u32>,
    #[serde(skip_serializing_if = "is_false")]
    pub request_prompt_caching: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i64>,
    #[serde(skip_serializing_if = "is_false")]
    pub logprobs: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

impl RawGenerationArgs {
    fn new(endpoint: &Endpoint, args: &GenerationArgs) -> Self {
        let mut raw = RawGenerationArgs::default();
        raw.model = args.model.clone();
        raw.seed = args.seed;
        raw.max_tokens = args.max_tokens;
        raw.temperature = args.temperature.map(F32);
        raw.frequency_penalty = args.frequency_penalty.map(F32);
        raw.presence_penalty = args.presence_penalty.map(F32);
        raw.repetition_penalty = args.repetition_penalty.map(F32);
        raw.repetition_penalty_range = args.repetition_penalty_range;
        raw.cache_prompt = args.request_prompt_caching;
        raw.priority = args.priority;
        raw.logprobs = args.logprobs;
        raw.top_logprobs = args.top_logprobs;
        if !endpoint.providers.is_empty() {
            raw.provider = Some(RawProvider {
                order: endpoint.providers.clone(),
                allow_fallbacks: endpoint.allow_fallbacks,
            })
        }
        raw
    }
}

#[derive(Clone, serde::Serialize)]
pub struct CompletionRequest {
    pub prompt: String,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Schema {
    JsonObject,
    JsonArrayOfStrings,
    JsonArrayOfObjects,
    JsonArrayOfArrays,
    JsonSchema(String),
    Choice(Vec<String>),
}

#[derive(Clone, serde::Serialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "is_false")]
    pub disable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Schema>,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    Completion(CompletionRequest),
    Chat(ChatRequest),
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    #[allow(dead_code)]
    pub max_sequence_length: u32,
    pub raw_info: serde_json::Value,
}

#[derive(serde::Deserialize, Debug)]
struct RawModelInfo {
    id: String,
    max_model_len: Option<u32>,
    context_length: Option<u32>,
}

#[derive(serde::Deserialize, Debug)]
struct RawModelsResponse {
    data: Vec<serde_json::Value>,
}

pub async fn fetch_models(endpoint: &Endpoint) -> Result<Vec<ModelInfo>, String> {
    let response = reqwest::Client::new()
        .get(&endpoint.models_url())
        .timeout(TIMEOUT)
        .send()
        .await
        .map_err(|error| format!("failed to fetch models: HTTP request failed: {error}"))?;

    let response = response
        .json::<RawModelsResponse>()
        .await
        .map_err(|error| format!("failed to fetch models: failed to parse reply as JSON: {error}"))?;
    let mut output = Vec::with_capacity(response.data.len());
    for raw_info in response.data {
        let raw_parsed_info = serde_json::from_value::<RawModelInfo>(raw_info.clone())
            .map_err(|error| format!("failed to fetch models: failed to parse reply: {error}"))?;

        output.push(ModelInfo {
            name: raw_parsed_info.id,
            max_sequence_length: match raw_parsed_info.max_model_len.or(raw_parsed_info.context_length) {
                Some(value) => value,
                None => {
                    return Err("failed to fetch models: reply is missing the context length field".into());
                }
            },
            raw_info,
        });
    }

    Ok(output)
}

#[derive(Clone, serde::Serialize)]
pub struct Request {
    pub args: GenerationArgs,
    pub kind: RequestKind,
}

impl Request {
    async fn send_impl(
        &self,
        endpoint: &Endpoint,
        stream: bool,
    ) -> Result<(Result<reqwest::Response, reqwest::Error>, Arc<String>), String> {
        let mut raw_request = RawGenerationArgs::new(endpoint, &self.args);
        raw_request.stream = stream;

        let url = match self.kind {
            RequestKind::Completion(CompletionRequest { ref prompt }) => {
                raw_request.prompt = Some(prompt.clone());
                endpoint.completion_url()
            }
            RequestKind::Chat(ChatRequest {
                ref messages,
                disable_thinking,
                ref reasoning_effort,
                ref schema,
            }) => {
                raw_request.messages = messages.clone();
                if disable_thinking {
                    let chat_template_kwargs = raw_request.chat_template_kwargs.get_or_insert(Value::Object(Default::default()));
                    let Value::Object(kwargs) = chat_template_kwargs else {
                        unreachable!()
                    };
                    kwargs.insert("enable_thinking".into(), false.into());
                    kwargs.insert(
                        "thinking".into(),
                        Value::Object({
                            let mut map = serde_json::Map::new();
                            map.insert("type".into(), "disabled".into());
                            map
                        }),
                    );
                }
                if let Some(reasoning_effort) = reasoning_effort {
                    let chat_template_kwargs = raw_request.chat_template_kwargs.get_or_insert(Value::Object(Default::default()));
                    let Value::Object(kwargs) = chat_template_kwargs else {
                        unreachable!()
                    };
                    kwargs.insert("reasoning_effort".into(), reasoning_effort.clone().into());
                }

                fn schema_preset(schema: &str) -> Option<RawResponseFormat> {
                    let schema: Value = serde_json::from_str(schema).unwrap();
                    Some(RawResponseFormat {
                        kind: "json_schema".into(),
                        json_schema: Some(RawJsonSchema { name: "".into(), schema }),
                    })
                }

                match schema {
                    None => {}
                    Some(Schema::JsonObject) => {
                        raw_request.response_format = Some(RawResponseFormat {
                            kind: "json_object".into(),
                            json_schema: None,
                        });
                    }
                    Some(Schema::JsonArrayOfStrings) => {
                        raw_request.response_format = schema_preset(include_str!("schema/json-array-of-strings.json"));
                    }
                    Some(Schema::JsonArrayOfObjects) => {
                        raw_request.response_format = schema_preset(include_str!("schema/json-array-of-objects.json"));
                    }
                    Some(Schema::JsonArrayOfArrays) => {
                        raw_request.response_format = schema_preset(include_str!("schema/json-array-of-arrays.json"));
                    }
                    Some(Schema::JsonSchema(schema)) => {
                        let Ok(Value::Object(ref mut map)) = serde_json::from_str(&schema) else {
                            return Err("failed to parse given JSON schema".into());
                        };

                        map.remove("$schema");
                        raw_request.response_format = Some(RawResponseFormat {
                            kind: "json_schema".into(),
                            json_schema: Some(RawJsonSchema {
                                name: "".into(),
                                schema: Value::Object(map.clone()),
                            }),
                        });
                    }
                    Some(Schema::Choice(choices)) => {
                        raw_request.structured_outputs = Some(RawStructuredOutputs {
                            choice: Some(choices.clone()),
                        })
                    }
                }
                endpoint.chat_url()
            }
        };

        let raw_request_s = serde_json::to_string(&raw_request).unwrap();
        let client = reqwest::Client::new();
        let mut client = client
            .post(&url)
            .timeout(TIMEOUT)
            .header("Content-Type", "application/json")
            .body(raw_request_s.clone());

        if !endpoint.api_key.is_empty() {
            client = client.header("Authorization", format!("Bearer {}", endpoint.api_key));
        }

        Ok((client.send().await, Arc::new(raw_request_s)))
    }

    pub async fn send(&self, endpoint: &Endpoint) -> Response {
        let (response, raw_request) = match self.send_impl(endpoint, false).await {
            Ok((Ok(response), raw_request)) => (response, raw_request),
            Ok((Err(error), raw_request)) => {
                return Response {
                    obj: Err(format!("HTTP error: {error}")),
                    raw: None,
                    original_request: Some(raw_request),
                };
            }
            Err(error) => {
                return Response {
                    obj: Err(error),
                    raw: None,
                    original_request: None,
                };
            }
        };

        let response = match response.bytes().await {
            Ok(response) => response,
            Err(error) => {
                return Response {
                    obj: Err(format!("failed to fetch response: {error}")),
                    raw: None,
                    original_request: Some(raw_request),
                };
            }
        };

        let response = match std::str::from_utf8(&response) {
            Ok(response) => response,
            Err(_) => {
                return Response {
                    obj: Err("response is not valid UTF-8".into()),
                    raw: None,
                    original_request: Some(raw_request),
                };
            }
        };

        parse_response(&response, raw_request)
    }

    pub async fn send_streaming(&self, endpoint: &Endpoint) -> Result<Pin<Box<dyn futures::Stream<Item = Response>>>, String> {
        let (client, raw_request) = match self.send_impl(endpoint, true).await {
            Ok((Ok(response), raw_request)) => (response, raw_request),
            Ok((Err(error), _)) => {
                return Err(format!("HTTP error: {error}"));
            }
            Err(error) => return Err(error),
        };

        struct State {
            buffer: Vec<u8>,
            client: reqwest::Response,
        }

        let raw_request_copy = raw_request.clone();
        let stream = futures::stream::unfold(
            State {
                client,
                buffer: Vec::new(),
            },
            move |mut state: State| {
                // https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream

                let raw_request = raw_request.clone();
                async move {
                    let mut is_finished = false;
                    loop {
                        let mut chunk_length = state.buffer.iter().position(|&ch| ch == b'\n' || ch == b'\r');
                        if chunk_length.is_none() && is_finished && !state.buffer.is_empty() {
                            chunk_length = Some(state.buffer.len());
                        }

                        if let Some(mut chunk_length) = chunk_length {
                            let chunk = &state.buffer[..chunk_length];
                            let Ok(mut chunk) = std::str::from_utf8(&chunk) else {
                                return Some((
                                    Ok(Response {
                                        obj: Err("response is not valid UTF-8".into()),
                                        raw: None,
                                        original_request: Some(raw_request.clone()),
                                    }),
                                    state,
                                ));
                            };

                            chunk_length += 1;
                            if state.buffer.get(chunk_length).copied() == Some(b'\n') {
                                chunk_length += 1;
                            }

                            chunk_length = chunk_length.min(state.buffer.len());

                            if chunk.starts_with("{") {
                                let chunk = parse_response(&chunk, raw_request);
                                state.buffer.drain(..chunk_length);
                                return Some((Ok(chunk), state));
                            }

                            let Some(index) = chunk.find(":") else {
                                state.buffer.drain(..chunk_length);
                                continue;
                            };

                            chunk = &chunk[index + 1..];
                            if chunk.starts_with(" ") {
                                chunk = &chunk[1..];
                            }

                            if chunk == "[DONE]" {
                                state.buffer.drain(..chunk_length);
                                return None;
                            }

                            let chunk = parse_response(&chunk, raw_request);
                            state.buffer.drain(..chunk_length);
                            return Some((Ok(chunk), state));
                        };

                        break match state.client.chunk().await {
                            Ok(Some(new_chunk)) => {
                                state.buffer.extend_from_slice(&new_chunk);
                                continue;
                            }
                            Ok(None) => {
                                is_finished = true;
                                if !state.buffer.is_empty() {
                                    continue;
                                }

                                None
                            }
                            Err(error) => Some((Err(error), state)),
                        };
                    }
                }
            },
        )
        .map(move |item| match item {
            Ok(response) => response,
            Err(error) => Response {
                obj: Err(format!("HTTP error: {error}")),
                raw: None,
                original_request: Some(raw_request_copy.clone()),
            },
        });

        Ok(Box::pin(stream))
    }
}

#[test]
fn test_parse_response_error_01() {
    let raw_response = include_str!("test-data/test-reply-01.json");
    let response = parse_response(&raw_response, Default::default());
    assert_eq!(
        response.obj.unwrap_err(),
        "response returned an error: code 502: Network connection lost."
    );
}
