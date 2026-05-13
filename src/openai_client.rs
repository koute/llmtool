use futures::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
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

    pub fn is_openrouter(&self) -> bool {
        self.url == "https://openrouter.ai/api"
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
pub struct RawReasoning {
    // Can be one or the other.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
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
    pub messages: Vec<RawMessageOut>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<RawToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<RawResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<RawStructuredOutputs>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<RawReasoning>,

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
    ToolCalls,
}

#[derive(Clone, serde::Deserialize, Debug)]
pub struct Usage {
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    // pub total_tokens: u64,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize)]
pub struct RawFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize)]
pub struct RawToolCall {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    // index: u64,
    pub function: Option<RawFunctionCall>,
}

impl RawToolCall {
    pub fn parse(&self) -> Result<ToolCallRequest, String> {
        let Some(kind) = self.kind.as_ref() else {
            return Err(format!("'type' not found"));
        };

        let Some(id) = self.id.as_ref() else {
            return Err(format!("'id' not found"));
        };

        match kind.as_str() {
            "function" => {
                let Some(ref raw_function) = self.function else {
                    return Err(format!("has 'function' type but no 'function' field"));
                };

                let Some(args) = raw_function.arguments.as_ref() else {
                    return Err(format!("'function.arguments' not found"));
                };

                let args: serde_json::Value = match serde_json::from_str(&args) {
                    Ok(arguments) => arguments,
                    Err(_) => {
                        return Err(format!("failed to parse 'function.arguments'"));
                    }
                };

                let Some(name) = raw_function.name.as_ref() else {
                    return Err(format!("'function.name' not found"));
                };

                Ok(ToolCallRequest {
                    raw: self.clone(),
                    id: id.to_owned(),
                    kind: ToolCallRequestKind::Function {
                        name: name.to_owned(),
                        args,
                    },
                })
            }
            kind => Err(format!("unsupported tool call type: '{kind}'")),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct RawMessageIn {
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
}

#[derive(serde::Deserialize, Debug)]
struct RawDelta {
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
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
    message: Option<RawMessageIn>,
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

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(String),
    AnyOf(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct ToolCallRequest {
    pub id: String,
    pub kind: ToolCallRequestKind,
    pub raw: RawToolCall,
}

impl ToolCallRequest {
    pub fn is_function_named(&self, tool_name: &str) -> bool {
        let ToolCallRequestKind::Function { ref name, .. } = self.kind;
        name == tool_name
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ToolCallRequestKind {
    Function { name: String, args: serde_json::Value },
}

#[derive(Clone, Debug)]
pub struct ResponseOk {
    pub finish_reason: Option<FinishReason>,
    pub text: String,
    pub reasoning_content: Option<String>,
    pub usage: Option<Usage>,
    pub model: String,
    pub kind: ResponseKind,
    pub tool_calls: Vec<ToolCallRequest>,
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

            let (is_streaming, text, reasoning_content, raw_tool_calls) = if let Some(message) = choice.message {
                (
                    false,
                    message.content.unwrap_or(String::new()),
                    message.reasoning_content.or(message.reasoning),
                    message.tool_calls,
                )
            } else if let Some(text) = choice.text {
                (false, text, None, None)
            } else if let Some(delta) = choice.delta {
                (
                    true,
                    delta.content.unwrap_or(String::new()),
                    delta.reasoning_content.or(delta.reasoning),
                    None,
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

            let mut tool_calls = Vec::new();
            if let Some(raw_tool_calls) = raw_tool_calls {
                for raw_tool_call in raw_tool_calls {
                    match raw_tool_call.parse() {
                        Ok(tool_call) => tool_calls.push(tool_call),
                        Err(error) => return create_response(Err(format!("malformed tool call found: {error}"))),
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
                tool_calls,
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
    pub top_k: Option<u32>,
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
        raw.top_k = args.top_k;
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

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<RawToolCall>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: String, content: String) -> Self {
        Self {
            role,
            content,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawMessageOut {
    pub role: String,
    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<RawToolCall>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawFunctionDef {
    pub name: String,
    pub description: String,
    #[serde(rename = "parameters")]
    pub args_schema: serde_json::Value,
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct RawToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<RawFunctionDef>,
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
#[serde(rename_all = "snake_case")]
pub enum ArgKind {
    Bool,
    String,
    Number,
    Array(Box<ArgKind>),
    Object(BTreeMap<String, Argument>),
    Enum(Vec<String>),
}

#[derive(Clone, serde::Serialize)]
pub struct Argument {
    pub description: String,
    pub kind: ArgKind,
    pub is_required: bool,
}

#[derive(Clone, serde::Serialize)]
pub enum ToolDef {
    Function {
        name: String,
        description: String,
        args: BTreeMap<String, Argument>,
    },
}

#[derive(Copy, Clone, serde::Serialize, clap::ValueEnum, Default)]
pub enum Thinking {
    #[default]
    Auto,
    Enable,
    Disable,
}

fn is_auto(thinking: &Thinking) -> bool {
    matches!(thinking, Thinking::Auto)
}

#[derive(Clone, serde::Serialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "is_auto")]
    pub thinking: Thinking,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Schema>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
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
                thinking,
                ref reasoning_effort,
                ref schema,
                ref tools,
                ref tool_choice,
            }) => {
                raw_request.messages = messages
                    .iter()
                    .map(|message| RawMessageOut {
                        role: message.role.clone(),
                        content: message.content.clone(),
                        reasoning: message.reasoning_content.clone(),
                        reasoning_content: message.reasoning_content.clone(),
                        tool_calls: message.tool_calls.clone(),
                        tool_call_id: message.tool_call_id.clone(),
                    })
                    .collect();

                if endpoint.is_openrouter() {
                    // Force it to send us the reasoning traces, if any.
                    raw_request.reasoning.get_or_insert_default();
                }

                let thinking = match thinking {
                    Thinking::Auto => None,
                    Thinking::Enable => Some(true),
                    Thinking::Disable => Some(false),
                };

                if let Some(thinking) = thinking {
                    let chat_template_kwargs = raw_request.chat_template_kwargs.get_or_insert(Value::Object(Default::default()));
                    let Value::Object(kwargs) = chat_template_kwargs else {
                        unreachable!()
                    };
                    kwargs.insert("enable_thinking".into(), thinking.into());
                    kwargs.insert(
                        "thinking".into(),
                        Value::Object({
                            let mut map = serde_json::Map::new();
                            map.insert("type".into(), if thinking { "enabled" } else { "disabled" }.into());
                            map
                        }),
                    );
                    raw_request.reasoning.get_or_insert_default().enabled = Some(thinking);
                }

                if thinking.unwrap_or(true) && raw_request.messages.iter().any(|message| message.reasoning.is_some()) {
                    let chat_template_kwargs = raw_request.chat_template_kwargs.get_or_insert(Value::Object(Default::default()));
                    let Value::Object(kwargs) = chat_template_kwargs else {
                        unreachable!()
                    };

                    // For GLM-4.7/GLM-4.7-Flash. See: https://docs.z.ai/guides/capabilities/thinking-mode
                    kwargs.insert("clear_thinking".into(), false.into());
                }

                if let Some(reasoning_effort) = reasoning_effort {
                    let chat_template_kwargs = raw_request.chat_template_kwargs.get_or_insert(Value::Object(Default::default()));
                    let Value::Object(kwargs) = chat_template_kwargs else {
                        unreachable!()
                    };
                    kwargs.insert("reasoning_effort".into(), reasoning_effort.clone().into());
                    if reasoning_effort.chars().all(|ch| ch.is_numeric()) {
                        let Ok(max_tokens) = reasoning_effort.parse() else {
                            return Err("cannot parse 'reasoning_effort'".into());
                        };

                        raw_request.reasoning.get_or_insert_default().max_tokens = Some(max_tokens);
                    } else {
                        raw_request.reasoning.get_or_insert_default().effort = Some(reasoning_effort.clone());
                    }
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

                for tool in tools {
                    fn emit_ty(obj: &mut serde_json::Map<String, serde_json::Value>, kind: ArgKind) {
                        let kind = match kind {
                            ArgKind::Bool => "boolean",
                            ArgKind::String => "string",
                            ArgKind::Enum(options) => {
                                obj.insert("enum".into(), options.into());
                                "string"
                            }
                            ArgKind::Number => "number",
                            ArgKind::Array(subkind) => {
                                let mut subobj = serde_json::Map::new();
                                emit_ty(&mut subobj, *subkind);
                                obj.insert("items".into(), subobj.into());
                                "array"
                            }
                            ArgKind::Object(properties) => {
                                let mut required: Vec<String> = Vec::new();
                                let mut props = serde_json::Map::new();
                                for (subkey, subarg) in properties {
                                    let mut subobj = serde_json::Map::new();
                                    emit_ty(&mut subobj, subarg.kind);
                                    subobj.insert("description".into(), subarg.description.into());
                                    if subarg.is_required {
                                        required.push(subkey.clone().into());
                                    }
                                    props.insert(subkey, serde_json::Value::Object(subobj));
                                }
                                obj.insert("properties".into(), props.into());
                                obj.insert("required".into(), required.into());
                                obj.insert("additionalProperties".into(), false.into());
                                "object"
                            }
                        };

                        obj.insert("type".into(), kind.into());
                    }

                    match tool {
                        ToolDef::Function { name, description, args } => {
                            let mut args_schema = serde_json::Map::new();
                            args_schema.insert("$schema".into(), "https://json-schema.org/draft/2020-12/schema".into());
                            emit_ty(&mut args_schema, ArgKind::Object(args.clone()));
                            let args_schema = args_schema.into();

                            raw_request.tools.push(RawToolDef {
                                kind: "function".into(),
                                function: Some(RawFunctionDef {
                                    name: name.clone(),
                                    description: description.clone(),
                                    args_schema,
                                }),
                            })
                        }
                    }
                }

                if let Some(tool_choice) = tool_choice {
                    let tool_choice: serde_json::Value = match tool_choice {
                        ToolChoice::None => "none".into(),
                        ToolChoice::Auto => "auto".into(),
                        ToolChoice::Required => "required".into(),
                        ToolChoice::Function(name) => serde_json::json! {{
                            "type": "function",
                            "function": {
                                "name": name,
                            }
                        }},
                        ToolChoice::AnyOf(names) => serde_json::json! {{
                            "type": "allowed_tools",
                            "allowed_tools": names.into_iter().map(|name| serde_json::json! {{
                                "type": "function",
                                "function": {
                                    "name": name,
                                }
                            }}).collect::<Vec<_>>()
                        }},
                    };

                    raw_request.tool_choice = Some(tool_choice);
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
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Response> + Send + 'static>>, String> {
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
struct DeltaToolCall {
    function_name: Option<String>,
    function_arguments: Option<String>,
    object: serde_json::Map<String, serde_json::Value>,
}

#[derive(Default)]
struct DeltaChoice {
    role: Option<String>,
    content: Option<String>,
    reasoning: Option<String>,
    object: serde_json::Map<String, serde_json::Value>,

    tool_calls: BTreeMap<u64, DeltaToolCall>,
}

#[derive(Default)]
pub struct DeltaState {
    object: serde_json::Map<String, serde_json::Value>,
    choices: BTreeMap<u64, DeltaChoice>,
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

                for choice in choices {
                    let serde_json::Value::Object(choice) = &choice else {
                        return Err("choice is not an object".into());
                    };

                    let Some(index) = choice.get("index") else {
                        return Err("choice object doesn't contain an 'index'".into());
                    };

                    let serde_json::Value::Number(index) = index else {
                        return Err("choice object contains an 'index' which isn't a number".into());
                    };

                    let Some(index) = index.as_u64() else {
                        return Err("choice object contains an 'index' which is not castable to u64".into());
                    };

                    let choice_state = self.choices.entry(index).or_insert_with(DeltaChoice::default);

                    if let Some(delta) = choice.get("delta").and_then(|delta| delta.as_object()) {
                        if let Some(role) = delta.get("role").and_then(|role| role.as_str()) {
                            choice_state.role = Some(role.to_owned());
                        }

                        if let Some(s) = delta.get("content").and_then(|s| s.as_str()) {
                            choice_state.content.get_or_insert_default().push_str(&s);
                        }

                        if let Some(s) = delta
                            .get("reasoning")
                            .and_then(|s| s.as_str())
                            .or_else(|| delta.get("reasoning_content").and_then(|s| s.as_str()))
                        {
                            choice_state.reasoning.get_or_insert_default().push_str(&s);
                        }

                        if let Some(tool_calls) = delta.get("tool_calls") {
                            let serde_json::Value::Array(tool_calls) = tool_calls else {
                                return Err("choice object contains a 'tool_calls' which is not an array".into());
                            };

                            for tool_call in tool_calls {
                                let serde_json::Value::Object(tool_call) = &tool_call else {
                                    return Err("tool call is not an object".into());
                                };

                                let Some(index) = tool_call.get("index") else {
                                    return Err("tool call object doesn't contain an 'index'".into());
                                };

                                let serde_json::Value::Number(index) = index else {
                                    return Err("tool call object contains an 'index' which isn't a number".into());
                                };

                                let Some(index) = index.as_u64() else {
                                    return Err("tool call object contains an 'index' which is not castable to u64".into());
                                };

                                let tool_call_state = choice_state.tool_calls.entry(index).or_insert_with(DeltaToolCall::default);
                                if let Some(function) = tool_call.get("function") {
                                    let Some(function) = function.as_object() else {
                                        return Err("tool call 'function' is not an object".into());
                                    };

                                    if let Some(name) = function.get("name") {
                                        if !name.is_null() {
                                            let Some(name) = name.as_str() else {
                                                return Err("tool call 'function.name' is not a string".into());
                                            };

                                            tool_call_state.function_name.get_or_insert_default().push_str(&name);
                                        }
                                    }

                                    if let Some(arguments) = function.get("arguments") {
                                        if !arguments.is_null() {
                                            let Some(arguments) = arguments.as_str() else {
                                                return Err("tool call 'function.arguments' is not a string".into());
                                            };

                                            tool_call_state.function_arguments.get_or_insert_default().push_str(&arguments);
                                        }
                                    }
                                } else {
                                    if let Some(kind) = tool_call.get("type").and_then(|kind| kind.as_str()) {
                                        return Err(format!("unsupported tool call type: '{kind}'"));
                                    } else {
                                        return Err("unsupported tool call type".into());
                                    }
                                }

                                for (subkey, subvalue) in tool_call {
                                    if subkey == "function" {
                                        continue;
                                    }

                                    tool_call_state.object.insert(subkey.into(), subvalue.clone());
                                }
                            }
                        }
                    }

                    for (subkey, subvalue) in choice {
                        if subkey == "delta" || (subvalue.is_null() && choice_state.object.contains_key(&*subkey)) {
                            continue;
                        }

                        choice_state.object.insert(subkey.into(), subvalue.clone());
                    }
                }

                continue;
            }

            self.object.insert(key.clone(), value.clone());
        }

        Ok(())
    }

    pub fn finalize(mut self) -> Result<serde_json::Value, String> {
        let mut choices = Vec::new();
        for mut choice in self.choices.into_values() {
            let Some(role) = choice.role else {
                return Err("missing 'role'".into());
            };

            let Some(content) = choice.content else {
                return Err("missing 'content'".into());
            };

            let mut message = serde_json::json! {{
                "role": role,
                "content": content,
                "refusal": null,
                "reasoning": null,
            }};

            if let Some(reasoning) = choice.reasoning {
                message.as_object_mut().unwrap().insert("reasoning".into(), reasoning.into());
            }

            if !choice.tool_calls.is_empty() {
                let mut tool_calls = Vec::new();
                for mut tool_call in choice.tool_calls.into_values() {
                    if tool_call.function_name.is_some() || tool_call.function_arguments.is_some() {
                        if !tool_call.object.contains_key("function") {
                            tool_call
                                .object
                                .insert("function".into(), serde_json::Value::Object(Default::default()));
                        } else if tool_call.object.get("function").unwrap().as_object().is_none() {
                            return Err("tool call 'function' is not an object".into());
                        }
                    }

                    if let Some(function_name) = tool_call.function_name {
                        tool_call
                            .object
                            .get_mut("function")
                            .unwrap()
                            .as_object_mut()
                            .unwrap()
                            .insert("name".into(), function_name.into());
                    }

                    if let Some(function_arguments) = tool_call.function_arguments {
                        tool_call
                            .object
                            .get_mut("function")
                            .unwrap()
                            .as_object_mut()
                            .unwrap()
                            .insert("arguments".into(), function_arguments.into());
                    }

                    tool_calls.push(tool_call.object);
                }

                message.as_object_mut().unwrap().insert("tool_calls".into(), tool_calls.into());
            }

            choice.object.insert("message".into(), message);
            choices.push(choice.object);
        }

        self.object.insert("choices".into(), choices.into());
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

#[test]
fn test_streaming_tool_call() {
    const EXPECTED_RESPONSE: &'static str = r#"
{
  "id": "chatcmpl-aa2ce11d16105ca0",
  "object": "chat.completion",
  "created": 1769351762,
  "model": "gpt-oss-120b",
  "prompt_token_ids": null,
  "usage": {
    "prompt_tokens": 9713,
    "total_tokens": 9826,
    "completion_tokens": 113
  },
  "choices": [
    {
      "index": 0,
      "logprobs": null,
      "finish_reason": "tool_calls",
      "token_ids": null,
      "stop_reason": 200012,
      "message": {
        "role": "assistant",
        "content": "",
        "refusal": null,
        "reasoning": "The user asks to list files in current directory. We need to use Bash tool to run ls (or we could use Glob). Since they want list files, easiest is Bash ls.\n\nWe'll run command `ls -1` maybe just `ls`. Use bash tool.",
        "tool_calls": [
          {
            "id": "chatcmpl-tool-a04db918809b96ad",
            "type": "function",
            "index": 0,
            "function": {
              "name": "bash",
              "arguments": "{\n  \"command\": \"ls -1\",\n  \"description\": \"List files in current directory\",\n  \"timeout\": 120000,\n  \"workdir\": \"/tmp/dummy\"\n}"
            }
          }
        ]
      }
    }
  ]
}
"#;

    let data = include_str!("test-data/02-local-streaming-tool-call.jsonl");
    let mut state = DeltaState::default();
    for line in data.lines() {
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        state.apply(&value).unwrap();
    }

    let response = state.finalize().unwrap();
    let response = serde_json::to_string_pretty(&response).unwrap();
    assert_eq!(EXPECTED_RESPONSE.trim(), response.trim());
}
