use futures::prelude::*;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub enum AsyncOption<R> {
    Some(R),
    None,
}

impl<R> AsyncOption<R> {
    fn get(self: Pin<&mut Self>) -> Option<Pin<&mut R>> {
        if matches!(*self, AsyncOption::None) {
            return None;
        }

        let pin = unsafe {
            self.map_unchecked_mut(|opt| match opt {
                AsyncOption::Some(reader) => reader,
                AsyncOption::None => unreachable!(),
            })
        };

        Some(pin)
    }
}

impl<R> From<Option<R>> for AsyncOption<R> {
    fn from(reader: Option<R>) -> Self {
        match reader {
            Some(reader) => Self::Some(reader),
            None => Self::None,
        }
    }
}

impl<R: tokio::io::AsyncRead> tokio::io::AsyncRead for AsyncOption<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf,
    ) -> std::task::Poll<std::io::Result<()>> {
        let Some(pin) = self.get() else { return std::task::Poll::Pending };
        tokio::io::AsyncRead::poll_read(pin, cx, buf)
    }
}

impl<R: tokio::io::AsyncWrite> tokio::io::AsyncWrite for AsyncOption<R> {
    fn poll_write(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>, buf: &[u8]) -> std::task::Poll<std::io::Result<usize>> {
        let Some(pin) = self.get() else { return std::task::Poll::Pending };
        tokio::io::AsyncWrite::poll_write(pin, cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<std::io::Result<()>> {
        let Some(pin) = self.get() else { return std::task::Poll::Pending };
        tokio::io::AsyncWrite::poll_flush(pin, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<std::io::Result<()>> {
        let Some(pin) = self.get() else { return std::task::Poll::Pending };
        tokio::io::AsyncWrite::poll_shutdown(pin, cx)
    }
}

impl<R: Future> Future for AsyncOption<R> {
    type Output = R::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        if matches!(*self, AsyncOption::None) {
            return std::task::Poll::Pending;
        }

        let pin = unsafe {
            self.map_unchecked_mut(|opt| match opt {
                AsyncOption::Some(reader) => reader,
                AsyncOption::None => unreachable!(),
            })
        };

        Future::poll(pin, cx)
    }
}

pub struct Child {
    stdin: AsyncOption<tokio::process::ChildStdin>,
    stdout: AsyncOption<tokio::process::ChildStdout>,
    stderr: AsyncOption<tokio::process::ChildStderr>,
    child: tokio::process::Child,
    stdin_pending: bytes::Bytes,
    stdout_buffer: Vec<u8>,
    stderr_buffer: Vec<u8>,
    whole_stdout: Vec<u8>,
    whole_stderr: Vec<u8>,
}

pub struct WaitStatus {
    pub code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

pub fn signal_name(signal: i32) -> Option<&'static str> {
    macro_rules! lookup_signals {
        ($($name:ident),+) => {
            $(
                if signal == libc::$name {
                    return Some(stringify!($name));
                }
            )+
        }
    }

    lookup_signals! {
        SIGABRT,
        SIGALRM,
        SIGBUS,
        SIGCHLD,
        SIGCONT,
        SIGFPE,
        SIGHUP,
        SIGILL,
        SIGINT,
        SIGIO,
        SIGIOT,
        SIGKILL,
        SIGPIPE,
        SIGPOLL,
        SIGPWR,
        SIGQUIT,
        SIGSEGV,
        SIGSTKFLT,
        SIGSTOP,
        SIGTSTP,
        SIGSYS,
        SIGTERM,
        SIGTRAP,
        SIGTTIN,
        SIGTTOU,
        SIGURG,
        SIGUSR1,
        SIGUSR2,
        SIGXCPU,
        SIGXFSZ,
        SIGWINCH
    };

    None
}

enum ChildResult {
    Pending,
    Timeout,
    WaitOk(WaitStatus),
    WaitFailed(std::io::Error),
}

pub enum WaitResult {
    Timeout,
    WaitOk(WaitStatus),
    WaitFailed(std::io::Error),
}

impl Child {
    pub async fn kill(&mut self) {
        let _ = self.child.kill().await;
    }

    pub async fn wait(&mut self, timeout: core::time::Duration) -> WaitResult {
        let mut global_timeout = Box::pin(tokio::time::sleep(timeout));
        loop {
            match self.poll(Some(&mut global_timeout)).await {
                ChildResult::Pending => continue,
                ChildResult::Timeout => break WaitResult::Timeout,
                ChildResult::WaitOk(result) => break WaitResult::WaitOk(result),
                ChildResult::WaitFailed(error) => break WaitResult::WaitFailed(error),
            }
        }
    }

    async fn poll(&mut self, timeout: Option<&mut Pin<Box<tokio::time::Sleep>>>) -> ChildResult {
        let timeout = AsyncOption::from(timeout);
        tokio::select! {
            status = self.child.wait() => {
                let status = match status {
                    Ok(status) => status,
                    Err(error) => {
                        return ChildResult::WaitFailed(error);
                    }
                };

                let stderr = String::from_utf8_lossy(&self.whole_stderr).into_owned();
                let stdout = String::from_utf8_lossy(&self.whole_stdout).into_owned();
                use std::os::unix::process::ExitStatusExt;

                ChildResult::WaitOk(WaitStatus {
                    code: status.code(),
                    signal: status.signal(),
                    stdout,
                    stderr,
                })
            }
            count = self.stdout.read(&mut self.stdout_buffer) => {
                match count {
                    Ok(count) => self.whole_stdout.extend_from_slice(&self.stdout_buffer[..count]),
                    Err(_) => {
                        self.stdout = AsyncOption::None;
                    }
                }

                ChildResult::Pending
            }
            count = self.stderr.read(&mut self.stderr_buffer) => {
                match count {
                    Ok(count) => self.whole_stderr.extend_from_slice(&self.stderr_buffer[..count]),
                    Err(_) => {
                        self.stderr = AsyncOption::None;
                    }
                }

                ChildResult::Pending
            }
            count = self.stdin.write(&mut self.stdin_pending) => {
                use bytes::Buf;
                match count {
                    Ok(count) => {
                        self.stdin_pending.advance(count);
                        if self.stdin_pending.is_empty() {
                            self.stdin = AsyncOption::None;
                        }
                    },
                    Err(_) => {
                        self.stdin = AsyncOption::None;
                    }
                }

                ChildResult::Pending
            }
            _ = timeout => {
                ChildResult::Timeout
            }
            else => {
                ChildResult::Pending
            }
        }
    }
}

pub struct SubprocessBuilder {
    pub workdir: PathBuf,
    pub command: String,
    pub args: Vec<String>,
    pub stdin: Option<String>,
    pub sandboxed: bool,
    pub whitelisted_paths: Vec<PathBuf>,
}

impl SubprocessBuilder {
    pub async fn spawn(self) -> Result<Child, String> {
        let mut child = tokio::process::Command::new(self.command);
        let child = child
            .args(self.args)
            .current_dir(&self.workdir)
            .stdin(if self.stdin.is_some() {
                std::process::Stdio::piped()
            } else {
                std::process::Stdio::null()
            })
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = if self.sandboxed {
            use landlock::{Access, AccessFs, RulesetAttr, RulesetCreatedAttr, make_bitflags};
            let abi = landlock::ABI::V6;
            let ruleset = landlock::Ruleset::default()
                .handle_access(landlock::AccessFs::from_all(abi))
                .map_err(|error| format!("failed to set up sandboxing: failed to call 'handle_access' on a ruleset: {error}"))?;

            let mut ruleset = ruleset
                .create()
                .map_err(|error| format!("failed to set up sandboxing: failed to create a Landlock ruleset: {error}"))?;

            ruleset = ruleset
                .add_rule(landlock::PathBeneath::new(
                    landlock::PathFd::new("/")
                        .map_err(|error| format!("failed to set up sandboxing: failed to open a path fd to '/': {error}"))?,
                    make_bitflags! { AccessFs::{ReadFile | ReadDir | Execute} },
                ))
                .map_err(|error| format!("failed to set up sandboxing: failed to add '/' to Landlock ruleset: {error}"))?;

            for path in self.whitelisted_paths {
                let Ok(fd) = landlock::PathFd::new(&path) else {
                    continue;
                };

                ruleset = ruleset
                    .add_rule(landlock::PathBeneath::new(fd, AccessFs::from_all(abi)))
                    .map_err(|error| {
                        format!(
                            "failed to set up sandboxing: failed to add whitelisted path '{}' to ruleset: {error}",
                            path.display()
                        )
                    })?;
            }
            let ruleset = Arc::new(Mutex::new(Some(ruleset)));

            child.spawn_with(move |command| {
                unsafe fn launder_lifetime<T>(x: &mut T) -> &'static mut T {
                    unsafe { core::mem::transmute(x) }
                }

                let ruleset = ruleset.0.lock().take().unwrap();
                let command = unsafe { launder_lifetime(command) };
                let result = std::thread::spawn(move || {
                    ruleset.restrict_self().map_err(|error| {
                        std::io::Error::new(std::io::ErrorKind::Other, format!("failed to sandbox the child process: {error}"))
                    })?;
                    command.spawn()
                })
                .join();

                match result {
                    Ok(result) => result,
                    Err(_) => return Err(std::io::Error::new(std::io::ErrorKind::Other, "failed to join on a new thread")),
                }
            })
        } else {
            child.spawn()
        }
        .map_err(|error| error.to_string())?;

        Ok(Child {
            stdin: AsyncOption::from(child.stdin.take()),
            stdout: AsyncOption::from(child.stdout.take()),
            stderr: AsyncOption::from(child.stderr.take()),
            child,
            stdin_pending: bytes::Bytes::from_owner(self.stdin.unwrap_or(String::new())),
            stdout_buffer: vec![0_u8; 64 * 1024],
            stderr_buffer: vec![0_u8; 64 * 1024],
            whole_stdout: Vec::new(),
            whole_stderr: Vec::new(),
        })
    }
}

pub struct Mutex<T>(pub parking_lot::Mutex<T>);

impl<T> Mutex<T> {
    pub fn new(value: T) -> Self {
        Self(parking_lot::Mutex::new(value))
    }
}
