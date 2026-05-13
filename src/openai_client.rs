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

    fn props_url(&self) -> String {
        format!("{}/props", self.url)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl ResponseError {
    pub fn new(code: u32, kind: Option<String>, message: String) -> Self {
        Self {
            code,
            message,
            kind,
            param: None,
        }
    }

    pub fn not_found() -> Self {
        Self::new(404, Some("not_found_error".into()), "not found".into())
    }

    pub fn bad_request(message: String) -> Self {
        Self::new(400, Some("bad_request".into()), message)
    }

    pub fn internal_server_error(message: String) -> Self {
        Self::new(500, Some("internal_server_error".into()), message)
    }
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
    pub kind: ResponseKind,
}

impl ResponseOk {
    pub fn is_reconstructed(&self) -> bool {
        matches!(self.kind, ResponseKind::Reconstructed)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ResponseKind {
    Normal,
    Streaming,
    Reconstructed,
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

    pub fn from_raw(raw_string: &str, raw_request: Option<Arc<String>>) -> Self {
        parse_response(raw_string, raw_request, None)
    }
}

fn parse_response(raw_string: &str, original_request: Option<Arc<String>>, delta_state: Option<&mut DeltaState>) -> Response {
    let raw_value: Result<Value, _> = serde_json::from_str(raw_string);
    let raw = Some(raw_string.to_owned());
    let create_response = |obj| Response {
        obj,
        raw,
        original_request,
    };

    let raw_value = match raw_value {
        Ok(raw_value) => raw_value,
        Err(error) => {
            return create_response(Err(format!("response is not valid JSON: {error}")));
        }
    };

    let Some(value) = raw_value.as_object() else {
        return create_response(Err(format!("response is not an object")));
    };

    if let Some(error) = value.get("error") {
        let error: Result<ResponseError, _> = serde_json::from_value(error.clone());
        return create_response(match error {
            Ok(error) => Ok(Err(error)),
            Err(error) => Err(format!("failed to parse 'error': {error}")),
        });
    }

    let response: Result<RawResponseOk, _> = serde_json::from_value(raw_value.clone());
    let obj = match response {
        Ok(response) => {
            let Some(choice) = response.choices.into_iter().next() else {
                return create_response(Err(format!("response is missing choices")));
            };

            if let Some(error) = choice.error {
                return create_response(Err(format!("response returned an error: code {}: {}", error.code, error.message)));
            }

            let (is_streaming, text, reasoning_content) = if let Some(message) = choice.message {
                (
                    false,
                    message.content.unwrap_or(String::new()),
                    message.reasoning_content.or(message.reasoning),
                )
            } else if let Some(text) = choice.text {
                (false, text, None)
            } else if let Some(delta) = choice.delta {
                (
                    true,
                    delta.content.unwrap_or(String::new()),
                    delta.reasoning_content.or(delta.reasoning),
                )
            } else {
                return create_response(Err(format!("response is missing 'text' and 'message'")));
            };

            if is_streaming {
                if let Some(delta_state) = delta_state {
                    if let Err(error) = delta_state.apply(&raw_value) {
                        return create_response(Err(format!("failed to apply delta state: {error}")));
                    }
                }
            }

            let response = ResponseOk {
                finish_reason: choice.finish_reason,
                text,
                reasoning_content: reasoning_content,
                model: response.model,
                usage: response.usage,
                kind: if is_streaming {
                    ResponseKind::Streaming
                } else {
                    ResponseKind::Normal
                },
            };

            Ok(Ok(response))
        }
        Err(error) => Err(format!("failed to parse response: {error}")),
    };

    create_response(obj)
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
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
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
        raw.top_p = args.top_p.map(F32);
        raw.min_p = args.min_p.map(F32);
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
    owned_by: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct RawModelsResponse {
    data: Vec<serde_json::Value>,
}

#[derive(serde::Deserialize, Debug)]
struct RawDefaultGenerationSettings {
    n_ctx: u32,
}

#[derive(serde::Deserialize, Debug)]
struct RawModelProps {
    default_generation_settings: RawDefaultGenerationSettings,
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
                    if raw_parsed_info
                        .owned_by
                        .as_ref()
                        .map(|owned_by| owned_by == "llamacpp")
                        .unwrap_or(false)
                    {
                        let response = reqwest::Client::new()
                            .get(&endpoint.props_url())
                            .timeout(TIMEOUT)
                            .send()
                            .await
                            .map_err(|error| format!("failed to fetch props: HTTP request failed: {error}"))?;

                        let response = response
                            .json::<RawModelProps>()
                            .await
                            .map_err(|error| format!("failed to fetch props: failed to parse reply as JSON: {error}"))?;

                        response.default_generation_settings.n_ctx
                    } else {
                        return Err("failed to fetch models: reply is missing the context length field".into());
                    }
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

pub enum StreamingChunk {
    Payload(String),
    Chunk(String),
    Finish,
    Error(String),
}

pub fn handle_streaming(client: reqwest::Response) -> Result<Pin<Box<dyn futures::Stream<Item = StreamingChunk> + Send>>, String> {
    struct State {
        buffer: Vec<u8>,
        client: reqwest::Response,
        is_finished: bool,
        sent_finish: bool,
    }

    let stream = futures::stream::unfold(
        State {
            client,
            buffer: Vec::new(),
            is_finished: false,
            sent_finish: false,
        },
        move |mut state: State| {
            // https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
            async move {
                loop {
                    let mut chunk_length = state.buffer.iter().position(|&ch| ch == b'\n' || ch == b'\r');
                    if chunk_length.is_none() && state.is_finished && !state.buffer.is_empty() {
                        chunk_length = Some(state.buffer.len());
                    }

                    if let Some(mut chunk_length) = chunk_length {
                        let chunk = &state.buffer[..chunk_length];
                        let Ok(mut chunk) = std::str::from_utf8(&chunk) else {
                            return Some((Ok(StreamingChunk::Error("response is not valid UTF-8".into())), state));
                        };

                        chunk_length += 1;
                        if state.buffer.get(chunk_length).copied() == Some(b'\n') {
                            chunk_length += 1;
                        }

                        chunk_length = chunk_length.min(state.buffer.len());

                        if chunk.starts_with("{") {
                            let chunk = StreamingChunk::Payload(chunk.to_owned());
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
                            state.buffer.clear();
                            state.is_finished = true;
                            continue;
                        }

                        if chunk == "OPENROUTER PROCESSING" {
                            state.buffer.drain(..chunk_length);
                            continue;
                        }

                        let chunk = StreamingChunk::Chunk(chunk.to_owned());
                        state.buffer.drain(..chunk_length);
                        return Some((Ok(chunk), state));
                    };

                    if state.is_finished {
                        if !state.sent_finish {
                            state.sent_finish = true;
                            return Some((Ok(StreamingChunk::Finish), state));
                        } else {
                            return None;
                        }
                    }

                    break match state.client.chunk().await {
                        Ok(Some(new_chunk)) => {
                            state.buffer.extend_from_slice(&new_chunk);
                            continue;
                        }
                        Ok(None) => {
                            state.is_finished = true;
                            continue;
                        }
                        Err(error) => Some((Err(error), state)),
                    };
                }
            }
        },
    )
    .map(move |item| match item {
        Ok(response) => response,
        Err(error) => StreamingChunk::Error(format!("HTTP error: {error}")),
    });

    Ok(Box::pin(stream))
}

impl Request {
    pub fn serialize_request(&self, endpoint: &Endpoint, stream: bool) -> Result<String, String> {
        let mut raw_request = RawGenerationArgs::new(endpoint, &self.args);
        raw_request.stream = stream;

        match self.kind {
            RequestKind::Completion(CompletionRequest { ref prompt }) => {
                raw_request.prompt = Some(prompt.clone());
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
            }
        };

        serde_json::to_string(&raw_request).map_err(|error| format!("failed to serialize request as JSON: {error}"))
    }

    async fn send_impl(
        &self,
        endpoint: &Endpoint,
        stream: bool,
    ) -> Result<(Result<reqwest::Response, reqwest::Error>, Arc<String>), String> {
        let raw_request_s = self.serialize_request(endpoint, stream)?;
        let url = match self.kind {
            RequestKind::Completion(..) => endpoint.completion_url(),
            RequestKind::Chat(..) => endpoint.chat_url(),
        };

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

        Response::from_raw(&response, Some(raw_request))
    }

    pub async fn send_streaming(
        &self,
        endpoint: &Endpoint,
        enable_reconstruction: bool,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Response>>>, String> {
        let (client, raw_request) = match self.send_impl(endpoint, true).await {
            Ok((Ok(response), raw_request)) => (response, raw_request),
            Ok((Err(error), _)) => {
                return Err(format!("HTTP error: {error}"));
            }
            Err(error) => return Err(error),
        };

        let mut delta_state = if enable_reconstruction { Some(DeltaState::default()) } else { None };
        let stream = handle_streaming(client)?.filter_map(move |chunk| {
            let chunk = match chunk {
                StreamingChunk::Payload(payload) => Some(Response::from_raw(&payload, Some(raw_request.clone()))),
                StreamingChunk::Chunk(payload) => Some(parse_response(&payload, Some(raw_request.clone()), delta_state.as_mut())),
                StreamingChunk::Finish => {
                    if let Some(delta_state) = delta_state.take() {
                        match delta_state.finalize() {
                            Ok(response) => {
                                let response = serde_json::to_string(&response).unwrap();
                                let mut response = parse_response(&response, Some(raw_request.clone()), None);
                                if let Ok(Ok(ref mut response)) = response.obj {
                                    response.kind = ResponseKind::Reconstructed;
                                }

                                Some(response)
                            }
                            Err(error) => Some(Response {
                                obj: Err(format!("failed to reconstruct a response from streaming chunks: {error}")),
                                raw: None,
                                original_request: Some(raw_request.clone()),
                            }),
                        }
                    } else {
                        None
                    }
                }
                StreamingChunk::Error(error) => Some(Response {
                    obj: Err(error),
                    raw: None,
                    original_request: Some(raw_request.clone()),
                }),
            };

            futures::future::ready(chunk)
        });

        Ok(Box::pin(stream))
    }
}

#[test]
fn test_parse_response_error_01() {
    let raw_response = include_str!("test-data/test-reply-01.json");
    let response = Response::from_raw(&raw_response, Default::default());
    assert_eq!(
        response.obj.unwrap_err(),
        "response returned an error: code 502: Network connection lost."
    );
}

#[derive(Default)]
pub struct DeltaState {
    object: serde_json::Map<String, serde_json::Value>,
    choice: serde_json::Map<String, serde_json::Value>,
    role: Option<String>,
    content: Option<String>,
    reasoning: Option<String>,
}

impl DeltaState {
    pub fn apply(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let serde_json::Value::Object(map) = value else {
            return Err("value is not a map".into());
        };

        for (key, value) in map {
            if key == "choices" {
                let serde_json::Value::Array(choices) = value else {
                    return Err("choices is not an array".into());
                };

                if choices.len() != 1 {
                    return Err("choices has unexpected length".into());
                }

                let serde_json::Value::Object(choice) = &choices[0] else {
                    return Err("choices is not an object".into());
                };

                if let Some(delta) = choice.get("delta").and_then(|delta| delta.as_object()) {
                    if let Some(role) = delta.get("role").and_then(|role| role.as_str()) {
                        self.role = Some(role.to_owned());
                    }

                    if let Some(s) = delta.get("content").and_then(|s| s.as_str()) {
                        self.content.get_or_insert_default().push_str(&s);
                    }

                    if let Some(s) = delta
                        .get("reasoning")
                        .and_then(|s| s.as_str())
                        .or_else(|| delta.get("reasoning_content").and_then(|s| s.as_str()))
                    {
                        self.reasoning.get_or_insert_default().push_str(&s);
                    }
                }

                for (subkey, subvalue) in choice {
                    if subkey == "delta" || (subvalue.is_null() && self.choice.contains_key(&*subkey)) {
                        continue;
                    }

                    self.choice.insert(subkey.into(), subvalue.clone());
                }

                continue;
            }
            self.object.insert(key.clone(), value.clone());
        }

        Ok(())
    }

    pub fn finalize(mut self) -> Result<serde_json::Value, String> {
        let Some(role) = self.role else {
            return Err("missing 'role'".into());
        };

        let Some(content) = self.content else {
            return Err("missing 'content'".into());
        };

        let mut message = serde_json::json! {{
            "role": role,
            "content": content,
            "refusal": null,
            "reasoning": null,
        }};

        if let Some(reasoning) = self.reasoning {
            message.as_object_mut().unwrap().insert("reasoning".into(), reasoning.into());
        }

        self.choice.insert("message".into(), message);
        self.object.insert("choices".into(), serde_json::json! { [self.choice] });
        self.object.insert("object".into(), "chat.completion".into());
        Ok(serde_json::Value::Object(self.object))
    }
}

#[cfg(test)]
fn test_streaming_reconstruction(streaming_data: &str, non_streaming_data: &str, modify_non_streaming: fn(&mut serde_json::Value)) {
    let mut state = DeltaState::default();

    for line in streaming_data.lines() {
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        state.apply(&value).unwrap();
    }

    let response_streaming = state.finalize().unwrap();
    let mut response_non_streaming: serde_json::Value = serde_json::from_str(&non_streaming_data).unwrap();
    modify_non_streaming(&mut response_non_streaming);

    if response_streaming != response_non_streaming {
        eprintln!(
            "Failed to reconstruct response!\nExpected response:\n{}\n\nReconstructed response:\n{}\n\n",
            serde_json::to_string_pretty(&response_non_streaming).unwrap(),
            serde_json::to_string_pretty(&response_streaming).unwrap()
        );
        panic!();
    }
}

#[cfg(test)]
fn clean_openrouter_response_to_match_streaming(response: &mut serde_json::Value) {
    response.as_object_mut().unwrap().get_mut("choices").unwrap()[0]
        .as_object_mut()
        .unwrap()
        .get_mut("message")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove("reasoning_details");
}

#[cfg(test)]
fn clean_local_response_to_match_streaming(response: &mut serde_json::Value) {
    let obj = response.as_object_mut().unwrap();
    for key in [
        "service_tier",
        "kv_transfer_params",
        "system_fingerprint",
        "usage",
        "prompt_logprobs",
    ] {
        obj.remove(key);
    }

    let message = obj.get_mut("choices").unwrap().as_array_mut().unwrap()[0]
        .as_object_mut()
        .unwrap()
        .get_mut("message")
        .unwrap()
        .as_object_mut()
        .unwrap();

    message.remove("annotations");
    message.remove("audio");
    message.remove("function_call");
    message.remove("tool_calls");
    message.remove("reasoning_content");
}

#[test]
fn test_reconstruct_reply_from_streaming_openrouter() {
    let streaming_data = include_str!("test-data/01-openrouter-streaming.jsonl");
    let non_streaming_data = include_str!("test-data/01-openrouter-non-streaming.json");
    test_streaming_reconstruction(streaming_data, non_streaming_data, clean_openrouter_response_to_match_streaming);
}

#[test]
fn test_reconstruct_reply_from_streaming_local() {
    let streaming_data = include_str!("test-data/01-local-streaming.jsonl");
    let non_streaming_data = include_str!("test-data/01-local-non-streaming.json");
    test_streaming_reconstruction(streaming_data, non_streaming_data, clean_local_response_to_match_streaming);
}
