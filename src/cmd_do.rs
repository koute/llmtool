use crate::CommonArgs;
use futures::prelude::*;
use std::io::Write;
use std::path::PathBuf;

use crate::agent::{Agent, AgentEvent, InfoKind, Session};
use crate::agent_utils::Mutex;

const VT_RED: &str = "\x1b[1;31m";
const VT_GREEN: &str = "\x1b[1;32m";
const VT_YELLOW: &str = "\x1b[1;33m";
const VT_MAGENTA: &str = "\x1b[1;35m";
const VT_CYAN: &str = "\x1b[1;36m";
const VT_DARK: &str = "\x1b[1;30m";
const VT_DARK_BLUE: &str = "\x1b[0;34m";
const VT_RESET: &str = "\x1b[0m";

#[derive(clap::Args)]
pub struct DoArgs {
    #[clap(flatten)]
    common_args: CommonArgs,

    /// Exit after the model considers the task done.
    #[clap(long)]
    oneshot: bool,

    /// Never ask the user for anything; implies '--oneshot'
    #[clap(long)]
    unattended: bool,

    /// Disable permission checks and all sandboxing, allowing the model to do anything; UNSAFE
    #[clap(long)]
    yolo: bool,

    /// Load and save the session into a given file.
    #[clap(long)]
    session: Option<PathBuf>,

    /// Path to whitelist.
    #[clap(long, short = 'w')]
    whitelist: Vec<PathBuf>,

    query: Vec<String>,
}

fn select(prompt: &str, choices: &[&str]) -> Option<usize> {
    dialoguer::Select::with_theme(&dialoguer::theme::ColorfulTheme::default())
        .with_prompt(prompt)
        .default(0)
        .items(&choices[..])
        .interact()
        .ok()
}

fn grab_input() -> Option<String> {
    dialoguer::Input::with_theme(&dialoguer::theme::SimpleTheme)
        .with_prompt("")
        .interact_text()
        .ok()
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    None,
    Reasoning,
    Text,
    ToolCallRequest,
    ToolCallSuccess,
    ToolCallFail,
    ToolCallBroken,
    Prompt,
    Choice,
    Error,
    SystemRemark,
}

struct ConsoleWriter {
    mode: Mode,
    ends_with_newline: bool,
}

impl ConsoleWriter {
    fn injected_newline(&mut self) {
        self.ends_with_newline = true;
    }

    fn change(&mut self, new_mode: Mode) {
        self.write(new_mode, "");
    }

    fn write(&mut self, new_mode: Mode, text: &str) {
        if self.mode == new_mode && text.is_empty() {
            return;
        }

        let out = std::io::stdout();
        let mut out = out.lock();

        if self.mode != new_mode {
            let color = match new_mode {
                Mode::Reasoning => VT_DARK,
                Mode::ToolCallRequest => VT_CYAN,
                Mode::ToolCallSuccess => VT_GREEN,
                Mode::ToolCallFail => VT_YELLOW,
                Mode::ToolCallBroken => VT_RED,
                Mode::SystemRemark => VT_YELLOW,
                Mode::Error => VT_RED,
                Mode::None | Mode::Prompt | Mode::Choice | Mode::Text => VT_RESET,
            };

            if self.ends_with_newline {
                let _ = out.write_all(b"\n");
            } else {
                let _ = out.write_all(b"\n\n");
            }

            let _ = out.write_all(color.as_bytes());
            self.mode = new_mode;
        }

        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
        self.ends_with_newline = text.ends_with('\n');
    }
}

fn flush_session(saved_session: &mut Option<Session>, agent: &Mutex<Agent>, session_path: Option<&(PathBuf, PathBuf)>) {
    let Some((session_path, tmp_path)) = session_path else { return };
    let session = agent.session();
    if saved_session
        .as_ref()
        .map(|saved_session| *saved_session == session)
        .unwrap_or(false)
    {
        return;
    }

    if let Ok(session_json) = serde_json::to_string_pretty(&session) {
        if let Ok(()) = std::fs::write(&tmp_path, &session_json) {
            if let Ok(()) = std::fs::rename(&tmp_path, &session_path) {
                *saved_session = Some(session);
            }
        }
    }
}

pub async fn main_do(
    DoArgs {
        mut common_args,
        query,
        oneshot,
        yolo,
        unattended,
        session: session_path,
        whitelist,
    }: DoArgs,
) -> Result<(), String> {
    let syntax_set = syntect::parsing::SyntaxSet::load_defaults_newlines();
    let mut theme = syntect::highlighting::ThemeSet::load_defaults().themes["base16-eighties.dark"].clone();
    theme.settings.background = Some(syntect::highlighting::Color::BLACK);

    let mut initial_prompt = query.join(" ").replace("\\n", "\n");

    if initial_prompt.is_empty() && unattended {
        return Err("no prompt was given".into());
    }

    let endpoint = common_args.common_setup().await?;
    if endpoint.is_local() && common_args.niceness.is_none() {
        common_args.niceness = Some(-1);
    }

    if common_args.repetition_penalty.is_none() {
        common_args.repetition_penalty = Some(1.0);
    }

    if common_args.frequency_penalty.is_none() {
        common_args.frequency_penalty = Some(0.0);
    }

    if common_args.presence_penalty.is_none() {
        common_args.presence_penalty = Some(0.0);
    }

    if common_args.temperature.is_none() {
        common_args.temperature = Some(1.0);
    }

    if common_args.min_p.is_none() {
        common_args.min_p = Some(0.125);
    }

    let agent = Agent::new(endpoint, common_args.get_generation_args()?);

    let mut stream = None;
    let mut saved_session = if let Some(ref session_path) = session_path {
        match std::fs::read_to_string(&session_path) {
            Ok(blob) if !blob.is_empty() => {
                let session: Session = serde_json::from_str(&blob)
                    .map_err(|error| format!("failed to parse the session from '{}': {error}", session_path.display()))?;
                stream = Some(
                    agent
                        .set_session(session.clone())
                        .map_err(|error| format!("failed to restore the session from '{}': {error}", session_path.display()))?,
                );
                Some(session)
            }
            Ok(_) => None,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(format!("failed to read the session from '{}': {error}", session_path.display()));
            }
        }
    } else {
        None
    };

    let session_path = if let Some(session_path) = session_path {
        let mut tmp_path = session_path.clone();
        let Some(filename) = tmp_path.file_name() else {
            return Err("invalid session path".into());
        };
        let Some(filename) = filename.to_str() else {
            return Err("invalid session path".into());
        };
        let filename = format!("{}.tmp", filename);
        tmp_path.set_file_name(filename);
        Some((session_path, tmp_path))
    } else {
        None
    };

    agent.set_yolo(yolo);
    for mut path in whitelist {
        if path.is_relative() {
            path = std::env::current_dir().unwrap().join(path);
        }

        let mut chunks = Vec::new();
        let mut path = loop {
            break match path.canonicalize() {
                Ok(path) => path,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound && path.file_name().is_some() && path.parent().is_some() => {
                    chunks.push(path.file_name().unwrap().to_owned());
                    path = path.parent().unwrap().into();
                    continue;
                }
                Err(error) => {
                    return Err(format!("failed to canonicalize path '{}': {}", path.display(), error));
                }
            };
        };

        for chunk in chunks.into_iter().rev() {
            path = path.join(chunk);
        }

        assert!(path.is_absolute());
        agent.whitelist_path(path);
    }

    let mut writer = ConsoleWriter {
        mode: Mode::None,
        ends_with_newline: true,
    };

    let (ctrlc_tx, mut ctrlc_rx) = tokio::sync::mpsc::channel(8);
    ctrlc::set_handler(move || {
        let _ = ctrlc_tx.try_send(());
    })
    .unwrap();

    let unattended_prompt = "The user is running you in unattended mode and is not here to answer any queries; if there is still work to be done then use your best judgement and continue, and if you're finished then call 'finish'.";

    let mut is_running = true;
    'outer_loop: loop {
        if (oneshot || unattended) && agent.is_finished() && stream.is_none() {
            is_running = false;
        }

        if !is_running {
            writer.change(Mode::Prompt);
            flush_session(&mut saved_session, &agent, session_path.as_ref());
            return Ok(());
        }

        let Some(mut stream) = stream.take() else {
            let prompt = if !initial_prompt.is_empty() {
                core::mem::take(&mut initial_prompt)
            } else {
                writer.change(Mode::Prompt);
                if unattended {
                    writer.write(Mode::Prompt, unattended_prompt);
                    writer.write(Mode::Prompt, "\n");
                    unattended_prompt.into()
                } else {
                    let Some(input) = grab_input() else {
                        return Ok(());
                    };
                    writer.injected_newline();
                    input
                }
            };

            agent.append_user(prompt);
            let new_stream = match agent.run().await {
                Ok(new_stream) => new_stream,
                Err(error) => {
                    writer.change(Mode::Prompt);
                    flush_session(&mut saved_session, &agent, session_path.as_ref());
                    return Err(error);
                }
            };
            stream = Some(new_stream);
            continue;
        };

        let mut timeout = Box::pin(tokio::time::interval(core::time::Duration::from_secs(10)));
        loop {
            let chunk = tokio::select! {
                chunk = stream.next() => {
                    let Some(chunk) = chunk else {
                        break;
                    };

                    chunk
                }
                _ = ctrlc_rx.recv() => {
                    is_running = false;
                    continue 'outer_loop;
                }
                _ = timeout.tick() => {
                    flush_session(&mut saved_session, &agent, session_path.as_ref());
                    continue;
                }
            };

            match chunk {
                AgentEvent::User { text } => {
                    writer.write(Mode::Prompt, &format!(": {}\n", &text));
                }
                AgentEvent::Info { text, kind } => {
                    let new_mode = match kind {
                        InfoKind::Text => Mode::Text,
                        InfoKind::Reasoning => Mode::Reasoning,
                    };

                    writer.write(new_mode, &text);
                }
                AgentEvent::Error(error) => {
                    writer.write(Mode::Error, &error);
                    break;
                }
                AgentEvent::Ask { tx } => {
                    if unattended {
                        writer.write(Mode::Prompt, unattended_prompt);
                        writer.write(Mode::Prompt, "\n");
                        let _ = tx.send(unattended_prompt.to_owned());
                        continue;
                    }

                    writer.change(Mode::Prompt);
                    let Some(text) = grab_input() else {
                        is_running = false;
                        continue 'outer_loop;
                    };
                    let _ = tx.send(text);
                    continue;
                }
                AgentEvent::SystemRemark { kind } => {
                    writer.write(Mode::SystemRemark, &format!("{}\n", kind));
                }
                AgentEvent::ToolCallRequest { name, args } => {
                    use core::fmt::Write;
                    let mut buffer = format!("{}:\n", name);
                    let mut multiline = Vec::new();
                    for (key, value) in args {
                        match value {
                            serde_json::Value::String(string) => {
                                if string.contains("\n") {
                                    multiline.push((key, string));
                                } else {
                                    let _ = writeln!(&mut buffer, "  {key} = {string:?}");
                                }
                            }
                            _ => {
                                let _ = write!(&mut buffer, "  {key} = ");
                                for (n, line) in serde_json::to_string_pretty(&value).unwrap().lines().enumerate() {
                                    if n > 0 {
                                        buffer.push_str("    ");
                                    }

                                    let _ = writeln!(&mut buffer, "{line}");
                                }
                            }
                        }
                    }

                    for (key, value) in multiline {
                        let _ = writeln!(&mut buffer, "  {key} = ...\n```");

                        use crate::agent::Tool;
                        if name == crate::agent_tools::Tool_Python.name() && key == "code" {
                            let syntax = syntax_set.find_syntax_by_extension("py").unwrap();
                            let mut h = syntect::easy::HighlightLines::new(syntax, &theme);
                            for line in syntect::util::LinesWithEndings::from(&value) {
                                let ranges: Result<Vec<(syntect::highlighting::Style, &str)>, _> = h.highlight_line(line, &syntax_set);
                                if let Ok(ranges) = ranges {
                                    let escaped = syntect::util::as_24_bit_terminal_escaped(&ranges[..], true);
                                    let _ = write!(&mut buffer, "{escaped}");
                                } else {
                                    let _ = write!(&mut buffer, "{line}");
                                }
                            }
                        } else {
                            let _ = writeln!(&mut buffer, "\n{value}");
                        }

                        let _ = writeln!(&mut buffer, "\n{VT_CYAN}```");
                    }
                    buffer.pop();
                    writer.write(Mode::ToolCallRequest, &buffer);
                }
                AgentEvent::ToolCallResult { name, result } => {
                    let mode = if result.is_ok() {
                        Mode::ToolCallSuccess
                    } else if result.is_broken() {
                        Mode::ToolCallBroken
                    } else {
                        Mode::ToolCallFail
                    };

                    writer.write(mode, &format!("{name}:\n{}", result.to_string()));
                }
                AgentEvent::Permission { tool, path, tx } => {
                    if unattended {
                        let _ = tx.send(false);
                        continue;
                    }

                    writer.change(Mode::Choice);
                    let response = match select("Allow this action?", &["Yes", "No", "More..."]) {
                        Some(0) => true,
                        Some(1) => false,
                        Some(2) => {
                            let mut actions: Vec<(String, Box<dyn FnOnce(&Mutex<Agent>) -> bool>)> = Vec::new();
                            actions.push(("Yes".into(), Box::new(|_| true)));
                            actions.push(("No".into(), Box::new(|_| false)));
                            if let Some(path) = path {
                                let path_clone = path.clone();
                                actions.push((
                                    format!("Always allow path for ANY tool: {}", path.display()),
                                    Box::new(|agent| {
                                        agent.whitelist_path(path_clone);
                                        true
                                    }),
                                ));

                                if let Some(parent) = path.parent() {
                                    let parent = parent.to_owned();
                                    actions.push((
                                        format!("Always allow path for ANY tool: {}", parent.display()),
                                        Box::new(|agent| {
                                            agent.whitelist_path(parent);
                                            true
                                        }),
                                    ));
                                }
                            }

                            actions.push((
                                "Always allow this tool".into(),
                                Box::new(|agent| {
                                    agent.whitelist_tool(&*tool);
                                    true
                                }),
                            ));
                            actions.push((
                                "Always allow ANY tool (YOLO mode)".into(),
                                Box::new(|agent| {
                                    agent.set_yolo(true);
                                    true
                                }),
                            ));

                            let mut choices = Vec::new();
                            for (choice, _) in &actions {
                                choices.push(choice.as_str());
                            }

                            let Some(index) = select("Allow this action?", &choices) else {
                                is_running = false;
                                continue 'outer_loop;
                            };
                            (actions.swap_remove(index).1)(&agent)
                        }
                        None => {
                            is_running = false;
                            continue 'outer_loop;
                        }
                        _ => unreachable!(),
                    };

                    let _ = tx.send(response);
                    writer.injected_newline();
                    continue;
                }
            };
        }
    }
}
