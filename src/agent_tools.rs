#![allow(non_camel_case_types)]

use crate::agent_utils::{Child, Mutex, SubprocessBuilder, WaitResult, WaitStatus, signal_name};
use crate::openai_client::{ArgKind, Argument, ToolCallRequestKind, ToolDef};
use crate::utils::{Lines, mmap_read};
use async_trait::async_trait;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use crate::agent::{Agent, AgentEvent, Tool, ToolResult, Tx};

macro_rules! parse_args {
    ($itself:expr, $args_ty:ident, $args:expr) => {
        match serde_json::from_value::<$args_ty>($args) {
            Ok(args) => args,
            Err(error) => return ToolResult::invalid_arguments($itself, error),
        }
    };
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_CreateFile {
    path: String,
}

#[derive(Copy, Clone)]
pub struct Tool_CreateFile;

#[async_trait]
impl Tool for Tool_CreateFile {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "CreateFile"
    }
    fn name(&self) -> &str {
        "create_file"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Creates a new blank file; will return the absolute path of the new file, or an error if the file already exists and is non-empty".into(),
            args: [
                (
                    "path".to_owned(),
                    Argument {
                        description: "The path of the new file; can be relative or absolute".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                )
            ].into()
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        fn already_exists(path: &Path) -> ToolResult {
            ToolResult::ok(format!("An empty file already exists at: {}", path.display()))
        }

        let args = parse_args!(self, ToolArgs_CreateFile, args);

        let path = agent.cwd().join(PathBuf::from(&args.path));
        let Some(dirname) = path.parent() else {
            return ToolResult::err(format!("Invalid path: {}", args.path));
        };

        if let Ok(metadata) = path.metadata() {
            if metadata.len() == 0 && metadata.is_file() {
                return already_exists(&path);
            }
        }

        if !dirname.exists() {
            return ToolResult::err(format!(
                "Cannot create an empty file at '{}': the parent directory '{}' doesn't exist",
                path.display(),
                dirname.display()
            ));
        }

        if !agent.ask_for_permission_with_file(&user_tx, self, path.clone()).await {
            return ToolResult::permission_denied();
        }

        let result = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map(|_| ());

        if let Err(ref error) = result {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                if let Ok(metadata) = path.metadata() {
                    if metadata.len() == 0 && metadata.is_file() {
                        return already_exists(&path);
                    }
                }
            }
        }

        match result {
            Ok(()) => ToolResult::ok(format!("Empty file created at: {}", path.display())),
            Err(error) => ToolResult::err(format!("Cannot create an empty file at '{}': {}", path.display(), error)),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_AppendToFile {
    path: String,
    text: String,
}

#[derive(Copy, Clone)]
pub struct Tool_AppendToFile;

#[async_trait]
impl Tool for Tool_AppendToFile {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "AppendToFile"
    }
    fn name(&self) -> &str {
        "append_to_file"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Appends text to a file; the appended text will always be put on a new line".into(),
            args: [
                (
                    "path".to_owned(),
                    Argument {
                        description: "The path to the file; must be absolute".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
                (
                    "text".to_owned(),
                    Argument {
                        description: "The text to append to the file; can be multiline".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
            ]
            .into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_AppendToFile, args);
        let absolute_path = match std::fs::canonicalize(&args.path) {
            Ok(path) => path,
            Err(error) => {
                return ToolResult::err(format!("Cannot append to '{}': {}", args.path, error));
            }
        };

        if !args.path.starts_with("/") {
            return ToolResult::err(format!(
                "Given path is not absolute: {}\nDid you mean this: {}",
                args.path,
                absolute_path.display()
            ));
        }

        if !agent.ask_for_permission_with_file(&user_tx, self, absolute_path.clone()).await {
            return ToolResult::permission_denied();
        }

        let mut fp = match std::fs::OpenOptions::new().read(true).append(true).open(&absolute_path) {
            Ok(fp) => fp,
            Err(error) => {
                return ToolResult::err(format!("Failed to open file '{}': {}", absolute_path.display(), error));
            }
        };

        let mut append_newline = false;
        if let Ok(_) = fp.seek(std::io::SeekFrom::End(-1)) {
            let mut buffer = [0];
            if let Ok(1) = fp.read(&mut buffer) {
                if buffer[0] != b'\n' {
                    append_newline = true;
                }
            }
        }

        let result = fp
            .seek(std::io::SeekFrom::End(0))
            .and_then(|_| if append_newline { fp.write_all(&[b'\n']) } else { Ok(()) })
            .and_then(|_| fp.write_all(args.text.as_bytes()));

        match result {
            Ok(()) => ToolResult::ok(format!("Text appended to file at: {}", absolute_path.display())),
            Err(error) => ToolResult::err(format!("Failed to write to file '{}': {}", absolute_path.display(), error)),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_ReadFile {
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_start: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_end: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_count: Option<u64>,
}

#[derive(Copy, Clone)]
pub struct Tool_ReadFile;

#[async_trait]
impl Tool for Tool_ReadFile {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "ReadFile"
    }
    fn name(&self) -> &str {
        "read_file"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Reads a given file and displays its contents with line numbers (the line numbers at the start of each line are NOT part of the file!)".into(),
            args: [
                (
                    "path".to_owned(),
                    Argument {
                        description: "The path to the file; MUST be absolute".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
                (
                    "line_start".to_owned(),
                    Argument {
                        description: "The first line from which to start reading (1-based indexing); optional, '1' by default".into(),
                        kind: ArgKind::Number,
                        is_required: false,
                    },
                ),
                (
                    "line_end".to_owned(),
                    Argument {
                        description: "The last line of the file to read (inclusive, 1-based indexing); optional".into(),
                        kind: ArgKind::Number,
                        is_required: false,
                    },
                ),
                (
                    "line_count".to_owned(),
                    Argument {
                        description: "The maximum number of lines to read; optional".into(),
                        kind: ArgKind::Number,
                        is_required: false,
                    },
                ),
            ]
            .into(),
        }
    }

    async fn run(&self, _agent: &Mutex<Agent>, _user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_ReadFile, args);
        let absolute_path = match std::fs::canonicalize(&args.path) {
            Ok(path) => path,
            Err(error) => {
                return ToolResult::err(format!("Cannot read file '{}': {}", args.path, error));
            }
        };

        if !args.path.starts_with("/") {
            return ToolResult::err(format!(
                "Given path is not absolute: {}\nDid you mean this: {}",
                args.path,
                absolute_path.display()
            ));
        }

        let mmap = match mmap_read(&absolute_path) {
            Ok(mmap) => mmap,
            Err(error) => {
                return ToolResult::err(format!("Failed to open file '{}': {}", absolute_path.display(), error));
            }
        };

        let starting_line = args.line_start.unwrap_or(1).saturating_sub(1);
        let line_count = args.line_count.unwrap_or(u64::MAX);

        use std::fmt::Write;
        let mut buffer = String::new();
        let mut lines_read = 0;
        let mut lines_total = 0;
        let mut first_line = 0;
        let mut last_line = 1;
        for (line_number, line) in Lines::new(&mmap, false).enumerate() {
            lines_total += 1;

            let line_number = line_number as u64;
            if lines_read >= line_count {
                continue;
            }

            if line_number < starting_line {
                continue;
            }

            if let Ok(line) = std::str::from_utf8(&line) {
                writeln!(&mut buffer, "{}:{line}", line_number + 1).unwrap();
            } else {
                todo!();
            }

            if first_line == 0 {
                first_line = line_number + 1;
            }
            last_line = line_number + 1;
            lines_read += 1;

            if let Some(boundary) = args.line_end {
                if line_number + 1 >= boundary {
                    break;
                }
            }
        }

        if lines_read == lines_total {
            ToolResult::ok(format!(
                "Contents of '{}' (whole file) between line {first_line} and {last_line} (inclusive):\n{buffer}",
                absolute_path.display()
            ))
        } else {
            ToolResult::ok(format!(
                "Contents of '{}' between line {first_line} and {last_line} (inclusive; {lines_read} line(s) shown; whole file is {lines_total} line(s) long):\n{buffer}",
                absolute_path.display()
            ))
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_EditFile {
    path: String,
    line_start: u64,
    line_end: u64,
    contents: String,
}

#[derive(Copy, Clone)]
pub struct Tool_EditFile;

#[async_trait]
impl Tool for Tool_EditFile {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "EditFile"
    }
    fn name(&self) -> &str {
        "edit_file"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: format!(
                "Edits a given file, replacing the contents of given line range; use this tool only to EDIT files, and if you want to append to a file then prefer the '{}' tool",
                Tool_AppendToFile.name()
            ),
            args: [
                (
                    "path".to_owned(),
                    Argument {
                        description: "The path to the file".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
                (
                    "line_start".to_owned(),
                    Argument {
                        description: "The first line of the file to edit (1-based indexing)".into(),
                        kind: ArgKind::Number,
                        is_required: true,
                    },
                ),
                (
                    "line_end".to_owned(),
                    Argument {
                        description: "The last line of the file to edit (inclusive, 1-based indexing)".into(),
                        kind: ArgKind::Number,
                        is_required: true,
                    },
                ),
                (
                    "contents".to_owned(),
                    Argument {
                        description: "The contents with which to replace the given line range; replaces the WHOLE range of lines".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
            ]
            .into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, is_confirmed: bool) -> ToolResult {
        use core::fmt::Write;

        let args = parse_args!(self, ToolArgs_EditFile, args);
        let absolute_path = match std::fs::canonicalize(&args.path) {
            Ok(path) => path,
            Err(error) => {
                return ToolResult::err(format!("Cannot read file '{}': {}", args.path, error));
            }
        };

        if !args.path.starts_with("/") {
            return ToolResult::err(format!(
                "Given path is not absolute: {}\nDid you mean this: {}",
                args.path,
                absolute_path.display()
            ));
        }

        if let Err(error) = std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .truncate(false)
            .create(false)
            .create_new(false)
            .open(&absolute_path)
        {
            return ToolResult::err(format!("Failed to open file '{}' for writing: {}", absolute_path.display(), error));
        }

        let tmp_path = {
            let mut tmp_path = absolute_path.clone();
            let Some(filename) = tmp_path.file_name() else {
                return ToolResult::err(format!("Invalid path: {}", absolute_path.display()));
            };
            let Some(filename) = filename.to_str() else {
                return ToolResult::err(format!("Invalid path: {}", absolute_path.display()));
            };

            use rand::Rng;
            let random: String = rand::rng()
                .sample_iter(&rand::distr::Alphanumeric)
                .take(6)
                .map(char::from)
                .collect();

            let filename = format!("{filename}.{random}.tmp");
            tmp_path.set_file_name(filename);
            tmp_path
        };

        let mmap = match mmap_read(&absolute_path) {
            Ok(mmap) => mmap,
            Err(error) => {
                return ToolResult::err(format!("Failed to open file '{}': {}", absolute_path.display(), error));
            }
        };

        let mut lines = Vec::new();
        for line in Lines::new(&mmap, false) {
            if let Ok(line) = std::str::from_utf8(line) {
                lines.push(line);
            } else {
                return ToolResult::err(format!(
                    "Cannot edit file '{}': the file contains invalid UTF-8",
                    absolute_path.display()
                ));
            }
        }

        let line_start = args.line_start;
        let line_end = args.line_end;

        if line_start == 0 {
            return ToolResult::err(format!(
                "Cannot edit file '{}': the starting line is 0 (remember: the line numbers use indexing which starts at 1!)",
                absolute_path.display()
            ));
        }

        if line_end == 0 {
            return ToolResult::err(format!(
                "Cannot edit file '{}': the last line is 0 (remember: the line numbers use indexing which starts at 1!)",
                absolute_path.display()
            ));
        }

        if line_end < line_start {
            return ToolResult::err(format!(
                "Cannot edit file '{}': the starting line must always be greater or equal to the last line",
                absolute_path.display()
            ));
        }

        if (line_start - 1) as usize >= lines.len() {
            return ToolResult::err(format!(
                "Cannot edit file '{}': the starting line is out of bounds (you've specified line {}, while the file only has {} lines)",
                absolute_path.display(),
                line_start,
                lines.len()
            ));
        }

        if (line_end - 1) as usize >= lines.len() {
            return ToolResult::err(format!(
                "Cannot edit file '{}': the last line is out of bounds (you've specified line {}, while the file only has {} lines)",
                absolute_path.display(),
                line_end,
                lines.len()
            ));
        }

        if is_confirmed {
            let mut out_lines: Vec<&str> = Vec::new();
            out_lines.extend(lines[0..line_start as usize - 1].iter());
            out_lines.extend(args.contents.lines());
            out_lines.extend(lines[line_end as usize..].iter());
            let out_lines = out_lines.join("\n");

            if let Err(error) = std::fs::write(&tmp_path, &out_lines) {
                return ToolResult::err(format!("Failed to write to '{}': {}", absolute_path.display(), error));
            }

            if let Err(error) = std::fs::rename(&tmp_path, &absolute_path) {
                let _ = std::fs::remove_file(tmp_path);
                return ToolResult::err(format!("Failed to write to '{}': {}", absolute_path.display(), error));
            }

            ToolResult::ok(format!("File '{}' was successfully edited", absolute_path.display()))
        } else {
            if !agent.ask_for_permission_with_file(&user_tx, self, absolute_path.clone()).await {
                return ToolResult::permission_denied();
            }

            let old_lines = &lines[line_start as usize - 1..=line_end as usize - 1];
            let old_lines_joined = old_lines.join("\n");
            let new_lines_joined = args.contents;
            let new_lines: Vec<_> = new_lines_joined.lines().collect();

            let path = absolute_path.display();
            let mut old_contents = String::new();
            let mut new_contents = String::new();
            let mut diff = String::new();

            for (n, line) in old_lines.iter().enumerate() {
                let _ = writeln!(&mut old_contents, "{}:{}", line_start + n as u64, line);
            }

            for (n, line) in new_lines.iter().enumerate() {
                let _ = writeln!(&mut new_contents, "{}:{}", line_start + n as u64, line);
            }

            for d in diff::lines(&old_lines_joined, &new_lines_joined) {
                let _ = match d {
                    diff::Result::Left(l) => writeln!(&mut diff, "-{}", l),
                    diff::Result::Both(l, _) => writeln!(&mut diff, " {}", l),
                    diff::Result::Right(r) => writeln!(&mut diff, "+{}", r),
                };
            }

            let mut output = String::new();
            let _ = write!(
                &mut output,
                "You are replacing lines {line_start}-{line_end} (inclusive) of file '{path}':\n```\n{old_contents}```\n\n"
            );
            let _ = write!(&mut output, "With these new lines:\n```\n{new_contents}```\n\n");
            let _ = write!(&mut output, "Here's the diff for your reference:\n```\n{diff}```\n\n");
            let _ = write!(
                &mut output,
                "The edit was NOT applied YET! Double-check and *verify* that this is actually want you want to do, and that you haven't made a mistake! Compare the lines you're replacing with the new lines, and also look at the diff. If you're ABSOLUTELY sure then confirm this edit by calling the 'confirm' tool."
            );

            ToolResult::unconfirmed(output)
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Bash {
    command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    workdir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asynchronous: Option<bool>,
}

#[derive(Copy, Clone)]
pub struct Tool_Bash;

#[async_trait]
impl Tool for Tool_Bash {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Bash"
    }
    fn name(&self) -> &str {
        "bash"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Executes a given command in a subprocess using a shell, with an optional timeout, and returns its output".into(),
            args: [
                (
                    "command".to_owned(),
                    Argument {
                        description: "The command to run".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
                (
                    "workdir".to_owned(),
                    Argument {
                        description:
                            "The working directory to run the command in; optional, will by default run in the current working directory"
                                .into(),
                        kind: ArgKind::String,
                        is_required: false,
                    },
                ),
                (
                    "timeout".to_owned(),
                    Argument {
                        description: "Completion timeout, in seconds; optional".into(),
                        kind: ArgKind::Number,
                        is_required: false,
                    },
                ),
                (
                    "asynchronous".to_owned(),
                    Argument {
                        description:
                            "When 'true' launches the process in the background and gives you its PID; optional, 'false' by default".into(),
                        kind: ArgKind::Bool,
                        is_required: false,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_Bash, args);
        let workdir_path = if let Some(ref workdir) = args.workdir {
            match std::path::Path::new(workdir).canonicalize() {
                Ok(path) if !workdir.starts_with("/") => {
                    return ToolResult::err(format!(
                        "Given workdir is not absolute: {}\nDid you mean this: {}",
                        workdir,
                        path.display()
                    ));
                }
                Ok(path) => path,
                Err(error) => {
                    return ToolResult::err(format!("Invalid workdir '{}': {}", workdir, error));
                }
            }
        } else {
            agent.cwd()
        };

        if !agent.ask_for_permission(&user_tx, self).await {
            return ToolResult::permission_denied();
        }

        let sandboxed = if agent.is_yolo() { false } else { true };
        let timeout = std::time::Duration::from_secs(args.timeout.unwrap_or(5.0).ceil() as u64);
        let child = SubprocessBuilder {
            workdir: workdir_path,
            command: "bash".into(),
            args: vec!["-c".into(), args.command],
            stdin: None,
            sandboxed,
            whitelisted_paths: agent.whitelisted_paths(),
        }
        .spawn()
        .await;

        match handle_child(timeout, child).await {
            Err(error) => return error,
            Ok(finished) => {
                let mut result = handle_child_wait_status(&finished);
                match result {
                    ToolResult::Fail { ref mut content } if content.contains("Permission denied") && sandboxed => {
                        use core::fmt::Write;
                        let _ = writeln!(
                            content,
                            "\n\nNOTE: This tool is *sandboxed* and can only modify a whitelisted set of paths. Consider whether the error you got is the result of sandboxing, and try using other tools if possible!"
                        );
                        result
                    }
                    result => result,
                }
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Mkdir {
    path: String,
}

#[derive(Copy, Clone)]
pub struct Tool_Mkdir;

#[async_trait]
impl Tool for Tool_Mkdir {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Mkdir"
    }
    fn name(&self) -> &str {
        "mkdir"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description:
                "Creates a new directory; will return the absolute path of the newly created directory, or if the directory already exists"
                    .into(),
            args: [(
                "path".to_owned(),
                Argument {
                    description: "The path of the new directory".into(),
                    kind: ArgKind::String,
                    is_required: true,
                },
            )]
            .into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        fn already_exists(path: &Path) -> ToolResult {
            ToolResult::ok(format!("No action necessary; directory already exists at: {}", path.display()))
        }

        let args = parse_args!(self, ToolArgs_Mkdir, args);
        let path = agent.cwd().join(PathBuf::from(&args.path));
        if let Ok(metadata) = path.metadata() {
            if metadata.is_dir() {
                return already_exists(&path);
            }
        }

        let Some(dirname) = path.parent() else {
            return ToolResult::err(format!("Invalid path: {}", args.path));
        };

        if !dirname.exists() {
            return ToolResult::err(format!(
                "Cannot create a new directory at '{}': the parent directory '{}' doesn't exist",
                path.display(),
                dirname.display()
            ));
        }

        if !agent.ask_for_permission_with_file(&user_tx, self, path.clone()).await {
            return ToolResult::permission_denied();
        }

        match std::fs::create_dir(&args.path) {
            Ok(()) => ToolResult::ok(format!("New directory created at: {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => already_exists(&path),
            Err(error) => ToolResult::err(format!("Cannot create a new directory at '{}': {}", path.display(), error)),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Ls {
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative: Option<bool>,
}

#[derive(Copy, Clone)]
pub struct Tool_Ls;

#[async_trait]
impl Tool for Tool_Ls {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Ls"
    }
    fn name(&self) -> &str {
        "ls"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Lists the files in a given directory".into(),
            args: [
                (
                    "path".to_owned(),
                    Argument {
                        description: "The directory to list the files in; optional, will by default use the current working directory".into(),
                        kind: ArgKind::String,
                        is_required: false,
                    },
                ),
                (
                    "relative".to_owned(),
                    Argument {
                        description: "When 'true' will print out relative paths; when 'false' will print out absolute paths; optional, 'false' by default".into(),
                        kind: ArgKind::Bool,
                        is_required: false,
                    }
                )
            ].into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, _user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        use core::fmt::Write;
        let mut args = parse_args!(self, ToolArgs_Ls, args);
        if let Some(ref mut path) = args.path {
            if path == "." || path == "./" {
                *path = "".into();
            }
        }

        let path = if let Some(path) = args.path {
            agent.cwd().join(PathBuf::from(&path))
        } else {
            agent.cwd()
        };

        if !path.exists() {
            return ToolResult::err(format!("Path doesn't exist: {}", path.display()));
        }

        if !path.is_dir() {
            return ToolResult::err(format!("Path is not a directory: {}", path.display()));
        }

        let dir = match std::fs::read_dir(&path) {
            Ok(dir) => dir,
            Err(error) => {
                return ToolResult::err(format!("Failed to read directory at '{}': {}", path.display(), error));
            }
        };

        let mut paths = Vec::new();
        for entry in dir {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    return ToolResult::err(format!("Failed to read directory at '{}': {}", path.display(), error));
                }
            };

            paths.push(entry.path());
        }

        if paths.is_empty() {
            return ToolResult::ok(format!("Directory at '{}' is empty", path.display()));
        }

        let mut output = format!(
            "Directory at '{}'{} contains {} file(s):\n\n",
            path.display(),
            if path == agent.cwd() {
                " (the current working directory)"
            } else {
                ""
            },
            paths.len()
        );
        for path in paths {
            if args.relative.unwrap_or(false) {
                let _ = writeln!(&mut output, "{}", path.file_name().unwrap().to_str().unwrap());
            } else {
                let _ = writeln!(&mut output, "{}", path.display());
            }
        }

        ToolResult::ok(output)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Cd {
    path: String,
}

#[derive(Copy, Clone)]
pub struct Tool_Cd;

#[async_trait]
impl Tool for Tool_Cd {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Cd"
    }
    fn name(&self) -> &str {
        "cd"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Changes the current working directory; will return the absolute path of the new working directory".into(),
            args: [(
                "path".to_owned(),
                Argument {
                    description: "The path to the new working directory; can be either relative or absolute".into(),
                    kind: ArgKind::String,
                    is_required: true,
                },
            )]
            .into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, _user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_Cd, args);
        let path = agent.cwd().join(PathBuf::from(&args.path));
        if !path.exists() {
            return ToolResult::err(format!("Path doesn't exist: {}", path.display()));
        }

        if !path.is_dir() {
            return ToolResult::err(format!("Path is not a directory: {}", path.display()));
        }

        agent.set_cwd(path.clone());
        ToolResult::ok(format!("The current working directory was changed to: {}", path.display()))
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Pwd {}

#[derive(Copy, Clone)]
pub struct Tool_Pwd;

#[async_trait]
impl Tool for Tool_Pwd {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Pwd"
    }
    fn name(&self) -> &str {
        "pwd"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Prints out the current working directory".into(),
            args: [].into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, _user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let _args = parse_args!(self, ToolArgs_Pwd, args);
        let cwd = agent.cwd();
        ToolResult::ok(cwd.display().to_string())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_WaitForChild {
    pid: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<f64>,
}

#[derive(Copy, Clone)]
pub struct Tool_WaitForChild;

#[async_trait]
impl Tool for Tool_WaitForChild {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "WaitForChild"
    }
    fn name(&self) -> &str {
        "wait_for_child"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Waits for a previously spawned child process to finish".into(),
            args: [
                (
                    "pid".to_owned(),
                    Argument {
                        description: "PID of the child".into(),
                        kind: ArgKind::Number,
                        is_required: true,
                    },
                ),
                (
                    "timeout".to_owned(),
                    Argument {
                        description: "Completion timeout, in seconds; optional".into(),
                        kind: ArgKind::Number,
                        is_required: false,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_WaitForChild, args);
        todo!()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_KillChild {
    pid: u64,
}

#[derive(Copy, Clone)]
pub struct Tool_KillChild;

#[async_trait]
impl Tool for Tool_KillChild {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "KillChild"
    }
    fn name(&self) -> &str {
        "kill_child"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Kills a previously spawned child process".into(),
            args: [(
                "pid".to_owned(),
                Argument {
                    description: "PID of the child".into(),
                    kind: ArgKind::Number,
                    is_required: true,
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_KillChild, args);
        todo!()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_ChildStatus {
    pid: u64,
}

#[derive(Copy, Clone)]
pub struct Tool_ChildStatus;

#[async_trait]
impl Tool for Tool_ChildStatus {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "ChildStatus"
    }
    fn name(&self) -> &str {
        "child_status"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Checks the status of a child process".into(),
            args: [(
                "pid".to_owned(),
                Argument {
                    description: "PID of the child".into(),
                    kind: ArgKind::Number,
                    is_required: true,
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_ChildStatus, args);
        todo!()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Ask {
    question: String,
}

#[derive(Copy, Clone)]
pub struct Tool_Ask;

#[async_trait]
impl Tool for Tool_Ask {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Ask"
    }
    fn name(&self) -> &str {
        "ask"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Asks the user an open-ended question.".into(),
            args: [(
                "question".to_owned(),
                Argument {
                    description: "The question to ask the user".into(),
                    kind: ArgKind::String,
                    is_required: true,
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, _agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let _args = parse_args!(self, ToolArgs_Ask, args);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = user_tx.send(AgentEvent::Ask { tx }).await;

        match rx.await {
            Ok(result) => ToolResult::ok(result),
            Err(_) => ToolResult::err("The user REJECTED giving you an answer; continue and use your best judgement."),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolArgs_Finish {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_result: Option<String>,
}

#[derive(Copy, Clone)]
pub struct Tool_Finish;

impl Tool_Finish {
    pub fn try_parse_non_empty(json: &str) -> Option<ToolArgs_Finish> {
        let args = serde_json::from_str::<ToolArgs_Finish>(json).ok()?;
        if args.final_result.is_none() {
            return None;
        }

        Some(args)
    }
}

#[async_trait]
impl Tool for Tool_Finish {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Finish"
    }
    fn name(&self) -> &str {
        "finish"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Signals to the user that you've finished your task. Be sure to *verify* that you've finished the task before using this tool!".into(),
            args: [
                (
                    "final_result".to_owned(),
                    Argument {
                        description: "The final result of the task, if applicable".into(),
                        kind: ArgKind::String,
                        is_required: false,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, _agent: &Mutex<Agent>, _user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let _args = parse_args!(self, ToolArgs_Finish, args);
        ToolResult::ok("")
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Confirm {}

#[derive(Copy, Clone)]
pub struct Tool_Confirm;

#[async_trait]
impl Tool for Tool_Confirm {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Confirm"
    }
    fn name(&self) -> &str {
        "confirm"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Confirms previous action; only use this when asked".into(),
            args: [].into(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let _args = parse_args!(self, ToolArgs_Confirm, args);
        let session = agent.session();
        let mut unconfirmed_id = None;
        for message in session.messages.iter().rev().skip(1) {
            if let Some((id, result)) = message.as_tool_result() {
                if result.is_unconfirmed() {
                    if unconfirmed_id.is_some() {
                        return ToolResult::err("no tool call found to confirm");
                    }

                    unconfirmed_id = Some(id);
                }

                continue;
            }

            let Some(unconfirmed_id) = unconfirmed_id else {
                continue;
            };

            let tool_calls = message.tool_calls();
            for tool_call in tool_calls {
                if tool_call.id == unconfirmed_id {
                    let ToolCallRequestKind::Function { ref name, ref args } = tool_call.kind;
                    let tool = agent.tool_by_name(name.as_str());
                    match tool {
                        Some(tool) => return tool.run(agent, &user_tx, args.clone(), true).await,
                        None => {
                            return ToolResult::err("failed to confirm tool call: previous unconfirmed tool call is invalid");
                        }
                    };
                }
            }
        }

        ToolResult::err("no tool call found to confirm")
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Python {
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    description: Option<String>, // gpt-oss likes to add it sometimes
    code: String,
}

#[derive(Copy, Clone)]
pub struct Tool_Python;

fn handle_child_wait_status(wait_status: &WaitStatus) -> ToolResult {
    use core::fmt::Write;
    let mut result = if let Some(code) = wait_status.code {
        format!("The command finished with status code: {code}")
    } else if let Some(signal) = wait_status.signal {
        if let Some(name) = signal_name(signal) {
            format!("The command received a signal: {signal} ({name})")
        } else {
            format!("The command received a signal: {signal}")
        }
    } else {
        unreachable!();
    };

    result.push('\n');
    if wait_status.stderr.is_empty() {
        result.push_str("Its stderr is EMPTY.");
    } else {
        let _ = write!(&mut result, "Here's its stderr:\n```\n{}\n```\n", wait_status.stderr);
    }
    result.push('\n');
    if wait_status.stdout.is_empty() {
        result.push_str("Its stdout is EMPTY.");
    } else {
        let _ = write!(&mut result, "Here's its stdout:\n```\n{}```\n", wait_status.stdout);
    }
    if let Some(code) = wait_status.code {
        if code == 0 {
            ToolResult::ok(result)
        } else {
            ToolResult::err(result)
        }
    } else if wait_status.signal.is_some() {
        ToolResult::err(result)
    } else {
        unreachable!();
    }
}

async fn handle_child(timeout: core::time::Duration, child: Result<Child, String>) -> Result<WaitStatus, ToolResult> {
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            return Err(ToolResult::err(format!("Failed to spawn child process: {}", error)));
        }
    };

    match child.wait(timeout).await {
        WaitResult::Timeout => Err(ToolResult::err("Command timed-out")),
        WaitResult::WaitFailed(error) => Err(ToolResult::err("Internal error: failed to wait for the command")),
        WaitResult::WaitOk(wait_status) => Ok(wait_status),
    }
}

#[async_trait]
impl Tool for Tool_Python {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Python"
    }
    fn name(&self) -> &str {
        "python"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Runs the given program in an embedded Python interpreter and returns its stdout; you MUST use the 'print' function to get output from this tool".into(),
            args: [
                (
                    "code".to_owned(),
                    Argument {
                        description: "The code to run".into(),
                        kind: ArgKind::String,
                        is_required: true,
                    },
                ),
                (
                    "timeout".to_owned(),
                    Argument {
                        description: "Completion timeout, in seconds; optional, 5s by default".into(),
                        kind: ArgKind::Number,
                        is_required: false,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_Python, args);
        if !agent.ask_for_permission(&user_tx, self).await {
            return ToolResult::permission_denied();
        }

        let timeout = std::time::Duration::from_secs(args.timeout.unwrap_or(5.0).ceil() as u64);
        let child = SubprocessBuilder {
            workdir: agent.cwd(),
            command: "/usr/bin/env".into(),
            args: vec!["python".into(), "-".into()],
            stdin: Some(args.code.clone()),
            sandboxed: if agent.is_yolo() { false } else { true },
            whitelisted_paths: agent.whitelisted_paths(),
        }
        .spawn()
        .await;

        match handle_child(timeout, child).await {
            Err(error) => return error,
            Ok(wait_status) => {
                if let Some(code) = wait_status.code {
                    if code == 0 {
                        return ToolResult::ok(wait_status.stdout.clone());
                    }
                }

                handle_child_wait_status(&wait_status)
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArgs_Query {
    prompt: String,
}

#[derive(Copy, Clone)]
pub struct Tool_Query;

#[async_trait]
impl Tool for Tool_Query {
    fn clone_boxed(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
    fn internal_id(&self) -> &str {
        "Query"
    }
    fn name(&self) -> &str {
        "query"
    }
    fn definition(&self) -> ToolDef {
        ToolDef::Function {
            name: self.name().into(),
            description: "Launches a subagent with a given prompt to answer a question or execute a task".into(),
            args: [(
                "prompt".to_owned(),
                Argument {
                    description: "The prompt to give to the subagent".into(),
                    kind: ArgKind::String,
                    is_required: true,
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    async fn run(&self, agent: &Mutex<Agent>, user_tx: &Tx, args: serde_json::Value, _is_confirmed: bool) -> ToolResult {
        let args = parse_args!(self, ToolArgs_Query, args);
        if !agent.ask_for_permission(&user_tx, self).await {
            return ToolResult::permission_denied();
        }

        todo!()
    }
}
