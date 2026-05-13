use crate::agent_tools::*;
use crate::agent_utils::{Child, Mutex};
use crate::openai_client;
use crate::openai_client::{Endpoint, GenerationArgs, RawFunctionCall, RawToolCall, ToolCallRequest, ToolCallRequestKind, ToolDef};
use crate::utils::extract_response;
use async_trait::async_trait;
use futures::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

const SYSTEM_PROMPT_TEMPLATE: &str = r#"
You are an interactive CLI tool that helps the user with anything he desires. Use the instructions below and the tools available to you to assist the user.

You should be concise, direct, and to the point. Avoid using emojis when communicating with the user.
Avoid pleasantries. If the user tells you that you're wrong don't say "I'm sorry" or "you're absolutely right". Do not sugarcoat. Be brutally honest.
Show, don't tell. Don't tell the user what you're doing. Just do it.
Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request.

Be proactive in completing the task you're given, and take follow-up actions when necessary without asking the user, as long as it is within the scope of the request.
Be relentless. You have unlimited time. Doing a good job and producing a high quality result is more important than doing it fast.
Never speculate or assume what the user wants; if something is unclear then always ask the user!

When asked to do something try to adhere to the following formula:
1. Analyze the request in detail and come up with a high-level, step-by-step plan on how to achieve it.
2. If necessary ask the user for any clarifications and to make important high-level decisions. Repeat this process until it's clear what you're supposed to do, and how to achieve it.
3. Start executing. Split the task into small, ideally easily solvable subtasks.
4. Go step by step and complete each subtask. Always verify that you've successfully finished a subtask, ideally by writing or running a unit test if possible.
5. Continue until you've finished the task.
{%- if 'Finish' in tools %}
6. Call the `{{ tools.Finish.name }}` tool.
{%- endif %}

Never use code comments as means to communicate with the user. Do not add comments to the code unless asked.
Adhere to the existing code style, use existing libraries and utilities, and follow existing patterns.
{%- if 'Bash' in tools %}
Only use the `{{ tools.Bash.name }}` tool as a last resort if there is no other tool for a given action.
Do NOT edit files with the `{{ tools.Bash.name }}` tool! Use the `{{ tools.EditFile.name }}` and `{{ tools.AppendToFile.name }}` tools to edit files.
{% endif %}

{%- if 'Finish' in tools %}
When you have finished the task, answered the user's final question or provided the requested output, immediately call the `{{ tools.Finish.name }}` tool.
Do NOT print out the final answer; always use the `{{ tools.Finish.name }}` to provide it.
{%- if 'Ask' in tools %}
If you are unsure whether the task is complete, you may ask the user for confirmation with the `{{ tools.Ask.name }}` tool before calling `{{ tools.Finish.name }}`.
ALWAYS use the `{{ tools.Ask.name }}` tool if you wish to ask the user a question.
{%- endif %}
{% endif %}

{%- if 'Python' in tools %}
NEVER do calculations manually. For every calculation:
1. Write a short Python program that computes the value.
2. Execute it with the `{{ tools.Python.name }}` tool.
3. Use the exact output from step 2 as the numeric answer.

Do not perform any arithmetic in your own reasoning. If the Python run fails, retry until you have a result.
I repeat, NEVER DO MATH YOURSELF.
{%- endif %}

Current working directory: {{ cwd }}
"#;

type AgentRef = Arc<Mutex<Agent>>;

#[derive(Clone, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
struct AssistantMessage {
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<RawToolCall>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemRemarkKind {
    AreYouFinished,
    DuplicateToolCallResult,
    DoYouWantToConfirm,
}

impl core::fmt::Display for SystemRemarkKind {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
        let s = match self {
            SystemRemarkKind::AreYouFinished => {
                "<system-remark>Are you finished? Verify that the solution meets ALL of the user's requirements; if yes then CALL the 'finish' tool, if not then iterate. If you need to ask the user a question then use the 'ask' tool.</system-remark>"
            }
            SystemRemarkKind::DuplicateToolCallResult => {
                "<system-remark>Hey, take note: you already did this with the same result; this isn't necessarily wrong so feel free to continue, but think carefully about what you're doing.</system-remark>"
            }
            SystemRemarkKind::DoYouWantToConfirm => {
                "<system-remark>Do you want to confirm? If you want to confirm you must CALL the 'confirm' tool; if not then carry on.</system-remark>"
            }
        };

        fmt.write_str(s)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnyMessage {
    kind: AnyMessageKind,
    is_hidden: bool,
}

impl From<AnyMessageKind> for AnyMessage {
    fn from(kind: AnyMessageKind) -> Self {
        Self { kind, is_hidden: false }
    }
}

impl AnyMessage {
    pub fn tool_calls(&self) -> Vec<ToolCallRequest> {
        let mut out = Vec::new();
        match &self.kind {
            AnyMessageKind::Assistant(AssistantMessage { tool_calls, .. }) => {
                for tool_call in tool_calls {
                    if let Ok(tool_call) = tool_call.parse() {
                        out.push(tool_call);
                    }
                }
            }
            _ => {}
        }

        out
    }
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnyMessageKind {
    User(String),
    SystemRemark(SystemRemarkKind),
    Assistant(AssistantMessage),
    ToolResult {
        id: String,
        #[serde(flatten)]
        result: ToolResult,
    },
}

impl AnyMessage {
    fn is_assistant(&self) -> bool {
        matches!(self.kind, AnyMessageKind::Assistant(..))
    }

    fn is_tool_result(&self) -> bool {
        matches!(self.kind, AnyMessageKind::ToolResult { .. })
    }

    fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self.kind {
            AnyMessageKind::Assistant(ref message) => Some(message),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<(&str, &ToolResult)> {
        match self.kind {
            AnyMessageKind::ToolResult { ref id, ref result } => Some((id.as_str(), result)),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Session {
    cwd: PathBuf,
    system_prompt: String,
    pub messages: Vec<AnyMessage>,
}

pub struct Agent {
    endpoint: Endpoint,
    generation_args: GenerationArgs,
    tools: Vec<Box<dyn Tool>>,
    session: Session,
    yolo: bool,
    whitelisted_paths: Vec<PathBuf>,
    whitelisted_tools: HashSet<String>,
    children: BTreeMap<i32, Child>,

    queued_user: Vec<String>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns a copy of the tool handle.
    fn clone_boxed(&self) -> Box<dyn Tool>;

    /// The internal name of the tool; should never change.
    fn internal_id(&self) -> &str;

    /// The name of the tool given to the LLM.
    fn name(&self) -> &str;

    /// The definition of the tool given to the LLM.
    fn definition(&self) -> ToolDef;

    /// Runs the tool.
    async fn run(&self, agent: &Mutex<Agent>, tx: &Tx, args: serde_json::Value, is_confirmed: bool) -> ToolResult;
}

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResult {
    Ok { content: String },
    Fail { content: String },
    Unconfirmed { content: String },
    Rejected,
    InvalidArguments { tool: String, error: String },
    UnknownTool { tool: String },
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self::Ok { content: content.into() }
    }

    pub fn err(content: impl Into<String>) -> Self {
        Self::Fail { content: content.into() }
    }

    pub fn unconfirmed(content: impl Into<String>) -> Self {
        Self::Unconfirmed { content: content.into() }
    }

    pub fn is_ok(&self) -> bool {
        match self {
            Self::Ok { .. } => true,
            Self::Fail { .. } | Self::Rejected | Self::Unconfirmed { .. } | Self::InvalidArguments { .. } | Self::UnknownTool { .. } => {
                false
            }
        }
    }

    pub fn is_broken(&self) -> bool {
        matches!(self, Self::InvalidArguments { .. } | Self::UnknownTool { .. })
    }

    pub fn is_unconfirmed(&self) -> bool {
        matches!(self, Self::Unconfirmed { .. })
    }

    pub fn invalid_arguments(tool: &dyn Tool, error: serde_json::Error) -> ToolResult {
        Self::InvalidArguments {
            tool: tool.name().to_owned(),
            error: error.to_string(),
        }
    }

    fn unknown_tool(tool: String) -> ToolResult {
        ToolResult::UnknownTool { tool }
    }

    pub fn permission_denied() -> ToolResult {
        Self::Rejected
    }

    pub fn to_string(&self) -> String {
        let (prefix, content) = match self {
            Self::Ok { content } => ("Status: OK", content),
            Self::Fail { content } => ("Status: FAIL", content),
            Self::Unconfirmed { content } => ("Status: NEEDS CONFIRMATION", content),
            Self::InvalidArguments { tool, error } => {
                return format!(
                    "Status: FAIL; INVALID ARGUMENTS\nThe arguments you've passed to tool '{tool}' are not valid; here's the error: {error}"
                );
            }
            Self::UnknownTool { tool } => {
                return format!(
                    "Status: FAIL; UNKNOWN TOOL\nTHE FOLLOWING TOOL IS NOT SUPPORTED: '{tool}'; SEE IF YOU'VE MADE A TYPO; IF NOT THEN DO NOT USE IT AGAIN"
                );
            }
            Self::Rejected => {
                return format!("Status: FAIL; PERMISSION DENIED BY THE USER\nYour call was correct, but it was rejected by the user");
            }
        };

        if content.is_empty() {
            prefix.to_owned()
        } else {
            format!("{prefix}\n{content}")
        }
    }
}

type ResponseStream = Pin<Box<dyn Stream<Item = openai_client::Response> + Send>>;

pub struct Tx(tokio::sync::mpsc::Sender<AgentEvent>);

impl core::ops::Deref for Tx {
    type Target = tokio::sync::mpsc::Sender<AgentEvent>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tx {
    async fn tool_req(&self, name: String, args: serde_json::Map<String, serde_json::Value>) {
        let _ = self.0.send(AgentEvent::ToolCallRequest { name, args }).await;
    }

    async fn tool_result(&self, name: String, result: ToolResult) {
        let _ = self.0.send(AgentEvent::ToolCallResult { name, result }).await;
    }

    async fn system_remark(&self, kind: SystemRemarkKind) {
        let _ = self.0.send(AgentEvent::SystemRemark { kind }).await;
    }

    async fn ask_for_permission(&self, tool: &dyn Tool, path: Option<PathBuf>) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self
            .0
            .send(AgentEvent::Permission {
                tool: tool.clone_boxed(),
                path,
                tx,
            })
            .await;

        match rx.await {
            Ok(result) => result,
            Err(_) => false,
        }
    }
}

#[derive(serde::Serialize)]
struct SystemPromptTool {
    name: String,
}

#[derive(serde::Serialize)]
struct SystemPromptEnvironment {
    cwd: String,
    tools: BTreeMap<String, SystemPromptTool>,
}

impl Agent {
    pub fn new(endpoint: Endpoint, generation_args: GenerationArgs) -> AgentRef {
        let cwd = std::env::current_dir().unwrap();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(Tool_Pwd),
            Box::new(Tool_Ls),
            Box::new(Tool_Cd),
            Box::new(Tool_Mkdir),
            Box::new(Tool_CreateFile),
            Box::new(Tool_AppendToFile),
            Box::new(Tool_ReadFile),
            Box::new(Tool_EditFile),
            Box::new(Tool_Bash),
            Box::new(Tool_Finish),
            Box::new(Tool_Confirm),
            Box::new(Tool_Ask),
            Box::new(Tool_Confirm),
            Box::new(Tool_Python),
            // Box::new(Tool_WaitForChild),
            // Box::new(Tool_KillChild),
            // Box::new(Tool_ChildStatus),
        ];

        let system_prompt = {
            let mut env = minijinja::Environment::new();
            env.add_template("system_prompt.txt", SYSTEM_PROMPT_TEMPLATE.trim()).unwrap();
            let template = env.get_template("system_prompt.txt").unwrap();
            template
                .render(minijinja::Value::from_serialize(SystemPromptEnvironment {
                    cwd: cwd.display().to_string(),
                    tools: tools
                        .iter()
                        .map(|tool| {
                            (
                                tool.internal_id().to_owned(),
                                SystemPromptTool {
                                    name: tool.name().to_owned(),
                                },
                            )
                        })
                        .collect(),
                }))
                .unwrap()
        };

        let state = Agent {
            yolo: false,
            endpoint,
            generation_args,
            tools,
            children: BTreeMap::new(),
            whitelisted_paths: Vec::new(),
            whitelisted_tools: HashSet::new(),
            queued_user: Vec::new(),
            session: Session {
                cwd,
                system_prompt,
                messages: Vec::new(),
            },
        };

        Arc::new(Mutex::new(state))
    }

    fn append_user(&mut self, prompt: String) {
        self.queued_user.push(prompt);
    }

    fn flush_output_queue(&mut self) {
        for prompt in self.queued_user.drain(..) {
            self.session.messages.push(AnyMessageKind::User(prompt).into());
        }
    }

    fn is_path_whitelisted(&self, path: &Path) -> bool {
        self.whitelisted_paths.iter().any(|root| root == path || path.starts_with(&root))
    }

    fn whitelist_path(&mut self, path: PathBuf) {
        if self.is_path_whitelisted(&path) {
            return;
        }

        self.whitelisted_paths.push(path);
    }

    fn is_finished(&self) -> bool {
        let Some(last) = self.session.messages.last() else { return false };
        if last
            .tool_calls()
            .into_iter()
            .any(|tool_call| tool_call.is_function_named(Tool_Finish.name()))
        {
            return true;
        }

        if let Some((id, _)) = last.as_tool_result() {
            if let Some(prev) = self.session.messages.iter().rev().skip(1).next() {
                if prev
                    .tool_calls()
                    .into_iter()
                    .any(|tool_call| tool_call.id == id && tool_call.is_function_named(Tool_Finish.name()))
                {
                    return true;
                }
            }
        }

        false
    }
}

impl Mutex<Agent> {
    pub fn is_yolo(&self) -> bool {
        self.0.lock().yolo
    }

    pub fn set_yolo(&self, value: bool) {
        self.0.lock().yolo = value;
    }

    pub fn is_finished(&self) -> bool {
        self.0.lock().is_finished()
    }

    pub fn append_user(&self, prompt: String) {
        self.0.lock().append_user(prompt);
    }

    pub fn cwd(&self) -> PathBuf {
        self.0.lock().session.cwd.clone()
    }

    pub fn set_cwd(&self, path: PathBuf) {
        self.0.lock().session.cwd = path;
    }

    fn is_path_whitelisted(&self, path: &Path) -> bool {
        self.0.lock().is_path_whitelisted(path)
    }

    pub fn whitelist_path(&self, path: PathBuf) {
        self.0.lock().whitelist_path(path);
    }

    pub fn whitelisted_paths(&self) -> Vec<PathBuf> {
        self.0.lock().whitelisted_paths.clone()
    }

    pub fn whitelist_tool(&self, tool: &dyn Tool) {
        self.0.lock().whitelisted_tools.insert(tool.internal_id().to_owned());
    }

    pub async fn ask_for_permission(&self, tx: &Tx, tool: &dyn Tool) -> bool {
        let state = self.0.lock();
        if state.yolo || state.whitelisted_tools.contains(tool.internal_id()) {
            return true;
        }

        tx.ask_for_permission(tool, None).await
    }

    pub async fn ask_for_permission_with_file(&self, tx: &Tx, tool: &dyn Tool, path: PathBuf) -> bool {
        if self.0.lock().yolo {
            return true;
        }

        if self.is_path_whitelisted(&path) {
            return true;
        }

        tx.ask_for_permission(tool, Some(path)).await
    }

    async fn run_inner(&self) -> Result<ResponseStream, String> {
        let mut state = self.0.lock();
        state.flush_output_queue();

        let mut messages = Vec::new();
        messages.push(openai_client::Message::new("system".into(), state.session.system_prompt.clone()));
        for message in &state.session.messages {
            if message.is_hidden {
                continue;
            }

            match &message.kind {
                AnyMessageKind::User(content) => {
                    messages.push(openai_client::Message::new("user".into(), content.clone()));
                }
                AnyMessageKind::SystemRemark(kind) => {
                    messages.push(openai_client::Message::new("user".into(), kind.to_string()));
                }
                AnyMessageKind::Assistant(message) => messages.push(openai_client::Message {
                    role: "assistant".into(),
                    content: message.content.clone(),
                    reasoning_content: message.reasoning.clone(),
                    tool_calls: if message.tool_calls.is_empty() {
                        None
                    } else {
                        Some(message.tool_calls.clone())
                    },
                    tool_call_id: None,
                }),
                AnyMessageKind::ToolResult { id, result } => {
                    messages.push(openai_client::Message {
                        role: "tool".into(),
                        content: result.to_string(),
                        reasoning_content: None,
                        tool_calls: Default::default(),
                        tool_call_id: Some(id.clone()),
                    });
                }
            }
        }

        let request = openai_client::ChatRequest {
            messages,
            thinking: openai_client::Thinking::Enable,
            reasoning_effort: None,
            schema: None,
            tools: state.tools.iter().map(|tool| tool.definition()).collect(),
            tool_choice: Some(openai_client::ToolChoice::Auto),
        };

        let request = openai_client::Request {
            args: state.generation_args.clone(),
            kind: openai_client::RequestKind::Chat(request),
        };

        let endpoint = state.endpoint.clone();
        core::mem::drop(state);

        request.send_streaming(&endpoint, true).await.map_err(|error| error.to_string())
    }

    fn assistant_mut(&self, callback: impl FnOnce(&mut AssistantMessage)) {
        let mut state = self.0.lock();
        if !state.session.messages.last().map(|message| message.is_assistant()).unwrap_or(false) {
            state.session.messages.push(AnyMessageKind::Assistant(Default::default()).into());
        }

        let AnyMessageKind::Assistant(ref mut message) = state.session.messages.last_mut().unwrap().kind else {
            unreachable!()
        };

        callback(message);
    }

    pub fn tool_by_name(&self, name: &str) -> Option<Box<dyn Tool>> {
        self.0
            .lock()
            .tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(|tool| tool.clone_boxed())
    }

    async fn main_loop(&self, tx: Tx, mut stream: ResponseStream) {
        loop {
            let Some(chunk) = stream.next().await else {
                let mut should_continue = false;
                {
                    let mut state = self.0.lock();
                    if !state.queued_user.is_empty() {
                        state.flush_output_queue();
                        should_continue = true;
                    } else if !state.is_finished() {
                        let mut kind = SystemRemarkKind::AreYouFinished;
                        if let Some(last) = state.session.messages.last_mut() {
                            if let AnyMessageKind::Assistant(AssistantMessage {
                                ref mut content,
                                ref mut tool_calls,
                                ..
                            }) = last.kind
                            {
                                if let Some(result) = Tool_Finish::try_parse_non_empty(&content) {
                                    // gpt-oss sometimes responds with the payload instead of calling the tool
                                    content.clear();

                                    use rand::Rng;
                                    let random: String = rand::rng()
                                        .sample_iter(&rand::distr::Alphanumeric)
                                        .take(12)
                                        .map(char::from)
                                        .collect();

                                    let id = format!("finish-{random}");
                                    tool_calls.push(RawToolCall {
                                        id: Some(random.clone()),
                                        kind: Some("function".into()),
                                        function: Some(RawFunctionCall {
                                            name: Some(Tool_Finish.name().into()),
                                            arguments: Some(serde_json::to_string(&result).unwrap()),
                                        }),
                                    });

                                    let result_value = serde_json::to_value(&result).unwrap();
                                    let serde_json::Value::Object(ref args_map) = result_value else {
                                        unreachable!()
                                    };

                                    tx.tool_req(Tool_Finish.name().into(), args_map.clone()).await;
                                    let result = Tool_Finish.run(self, &tx, result_value, true).await;

                                    state.session.messages.push(
                                        AnyMessageKind::ToolResult {
                                            id,
                                            result: result.clone(),
                                        }
                                        .into(),
                                    );

                                    tx.tool_result(Tool_Finish.name().into(), result.clone()).await;
                                    break;
                                }

                                if content.to_lowercase().contains("confirm") {
                                    if let Some(prev) = state.session.messages.iter().skip(1).next() {
                                        if let Some((_, result)) = prev.as_tool_result() {
                                            if result.is_unconfirmed() {
                                                kind = SystemRemarkKind::DoYouWantToConfirm;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        tx.system_remark(kind).await;
                        state.session.messages.push(AnyMessageKind::SystemRemark(kind).into());
                        should_continue = true;
                    }
                }

                if should_continue {
                    stream = match self.run_inner().await {
                        Ok(stream) => stream,
                        Err(error) => {
                            let _ = tx.send(AgentEvent::Error(error)).await;
                            return;
                        }
                    };
                    continue;
                }

                break;
            };

            let response = match extract_response(&chunk) {
                Ok(response) => response,
                Err(error) => {
                    let _ = tx.send(AgentEvent::Error(error)).await;
                    return;
                }
            };

            if !response.is_reconstructed() {
                if let Some(ref reasoning_content) = response.reasoning_content {
                    self.assistant_mut(|message| message.reasoning.get_or_insert_default().push_str(&reasoning_content));

                    if tx
                        .send(AgentEvent::Info {
                            kind: InfoKind::Reasoning,
                            text: reasoning_content.clone(),
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }

                if !response.content.is_empty() {
                    self.assistant_mut(|message| message.content.push_str(&response.content));

                    if tx
                        .send(AgentEvent::Info {
                            kind: InfoKind::Text,
                            text: response.content.clone(),
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }

            if !response.tool_calls.is_empty() {
                for tool_call in &response.tool_calls {
                    self.assistant_mut(|message| message.tool_calls.push(tool_call.raw.clone()));
                }

                for tool_call in &response.tool_calls {
                    let ToolCallRequestKind::Function { ref name, ref args } = tool_call.kind;
                    let tool = self.tool_by_name(name.as_str());

                    let result = match tool {
                        Some(tool) => {
                            let serde_json::Value::Object(args_map) = args else { todo!() };
                            tx.tool_req(name.clone(), args_map.clone()).await;
                            tool.run(self, &tx, args.clone(), false).await
                        }
                        None => ToolResult::unknown_tool(name.to_owned()),
                    };

                    tx.tool_result(name.to_owned(), result.clone()).await;

                    if name != Tool_Finish.name() {
                        let mut state = self.0.lock();

                        let mut dupe_count = 0;
                        for message in state
                            .session
                            .messages
                            .iter()
                            .take(16)
                            .filter(|message| message.is_tool_result())
                            .take(2)
                        {
                            if let Some((_, old_result)) = message.as_tool_result() {
                                if *old_result == result {
                                    dupe_count += 1;
                                }
                            }
                        }

                        state.session.messages.push(
                            AnyMessageKind::ToolResult {
                                id: tool_call.id.clone(),
                                result: result.clone(),
                            }
                            .into(),
                        );

                        if dupe_count >= 4 {
                            let kind = SystemRemarkKind::DuplicateToolCallResult;
                            state.session.messages.push(AnyMessageKind::SystemRemark(kind).into());
                            core::mem::drop(state);
                            tx.system_remark(kind).await;
                        }
                    }

                    if name == Tool_Finish.name() {
                        break;
                    }

                    stream = match self.run_inner().await {
                        Ok(stream) => stream,
                        Err(error) => {
                            let _ = tx.send(AgentEvent::Error(error)).await;
                            return;
                        }
                    };
                }
            }
        }
    }

    pub async fn run(self: &Arc<Self>) -> Result<Pin<Box<dyn Stream<Item = AgentEvent> + Send + 'static>>, String> {
        let stream = self.run_inner().await?;
        let itself = self.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tokio::spawn(async move { itself.main_loop(Tx(tx), stream).await });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    pub fn session(&self) -> Session {
        self.0.lock().session.clone()
    }

    pub fn set_session(&self, session: Session) -> Result<Pin<Box<dyn Stream<Item = AgentEvent> + Send + 'static>>, String> {
        let mut state = self.0.lock();
        let mut events = Vec::new();
        let mut call_id_to_tool_name = HashMap::new();
        for message in &session.messages {
            match &message.kind {
                AnyMessageKind::User(text) => {
                    events.push(AgentEvent::User { text: text.clone() });
                }
                AnyMessageKind::SystemRemark(kind) => {
                    events.push(AgentEvent::SystemRemark { kind: *kind });
                }
                AnyMessageKind::Assistant(AssistantMessage {
                    content,
                    reasoning,
                    tool_calls,
                }) => {
                    if !content.is_empty() {
                        events.push(AgentEvent::Info {
                            kind: InfoKind::Text,
                            text: content.clone(),
                        });
                    }

                    if let Some(reasoning) = reasoning {
                        events.push(AgentEvent::Info {
                            kind: InfoKind::Reasoning,
                            text: reasoning.clone(),
                        });
                    }

                    for tool_call in tool_calls {
                        let tool_call = tool_call.parse().map_err(|error| format!("failed to parse tool call: {error}"))?;
                        let ToolCallRequestKind::Function { ref name, ref args } = tool_call.kind;
                        call_id_to_tool_name.insert(tool_call.id.clone(), name.clone());

                        let serde_json::Value::Object(args) = args else {
                            return Err("found a tool call with a non-object payload".into());
                        };

                        events.push(AgentEvent::ToolCallRequest {
                            name: name.clone(),
                            args: args.clone(),
                        });
                    }
                }
                AnyMessageKind::ToolResult { id, result } => {
                    let name = call_id_to_tool_name.get(&*id).cloned().unwrap_or_else(|| "UNKNOWN TOOL".into());
                    events.push(AgentEvent::ToolCallResult {
                        name: name.clone(),
                        result: result.clone(),
                    })
                }
            }
        }

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tokio::spawn(async move {
            for event in events {
                if tx.send(event).await.is_err() {
                    break;
                }
            }
        });

        state.session = session;
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

pub enum InfoKind {
    Text,
    Reasoning,
}

pub enum AgentEvent {
    User {
        text: String,
    },
    Info {
        kind: InfoKind,
        text: String,
    },
    SystemRemark {
        kind: SystemRemarkKind,
    },
    ToolCallRequest {
        name: String,
        args: serde_json::Map<String, serde_json::Value>,
    },
    ToolCallResult {
        name: String,
        result: ToolResult,
    },
    Permission {
        tool: Box<dyn Tool>,
        path: Option<PathBuf>,
        tx: tokio::sync::oneshot::Sender<bool>,
    },
    Ask {
        tx: tokio::sync::oneshot::Sender<String>,
    },
    Error(String),
}
