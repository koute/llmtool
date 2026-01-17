use clap::Parser;

use ahash::AHashMap as HashMap;
use ahash::AHashSet as HashSet;
use core::time::Duration;
use futures::prelude::*;
use openai_client::Endpoint;
use rayon::prelude::*;
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

mod openai_client;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(clap::Args)]
pub struct CommonArgs {
    #[clap(long)]
    pub model: Option<String>,

    #[clap(long)]
    pub seed: Option<u32>,

    #[clap(short = 'l', long)]
    pub max_tokens: Option<u32>,

    #[clap(short = 't', long)]
    pub temperature: Option<f32>,

    #[clap(long)]
    pub frequency_penalty: Option<f32>,

    #[clap(long)]
    pub presence_penalty: Option<f32>,

    #[clap(long)]
    pub repetition_penalty: Option<f32>,

    #[clap(long)]
    pub repetition_penalty_range: Option<u32>,

    #[clap(long)]
    pub request_prompt_caching: bool,

    #[clap(short = 'r', long)]
    pub reproducible: bool,

    #[clap(long)]
    pub url: Option<String>,

    #[clap(long)]
    pub api_key: Option<String>,

    #[clap(long)]
    pub provider: Option<String>,

    #[clap(long)]
    pub niceness: Option<i64>,

    #[clap(long)]
    pub logprobs: bool,

    #[clap(long)]
    pub top_logprobs: Option<u32>,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum OutputFormat {
    JsonObject,
    JsonArrayOfStrings,
    JsonArrayOfObjects,
    JsonArrayOfArrays,
}

#[derive(clap::Args)]
pub struct ChatArgs {
    #[clap(long)]
    disable_thinking: bool,

    #[clap(long)]
    reasoning_effort: Option<String>,

    #[clap(long, short = 's')]
    system_prompt: Option<String>,

    #[clap(long)]
    output_format_choice: Option<String>,

    #[clap(long)]
    output_format: Option<OutputFormat>,

    #[clap(long)]
    json_schema: Option<PathBuf>,
}

#[derive(Copy, Clone, Default, clap::ValueEnum)]
enum IsEnabled {
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Copy, Clone, Default, clap::ValueEnum)]
enum Thinking {
    #[default]
    Auto,
    Show,
    Hide,
}

#[derive(clap::Args)]
pub struct SingleRequestArgs {
    #[clap(long, short = 'v')]
    verbose: bool,

    #[clap(long, default_value = "auto")]
    streaming: IsEnabled,

    #[clap(long, default_value = "auto")]
    stdin: IsEnabled,
}

#[derive(clap::Parser)]
enum Args {
    /// Sends a single chat completion query.
    Q {
        #[clap(long, default_value = "auto")]
        thinking: Thinking,

        #[clap(flatten)]
        common_args: CommonArgs,

        #[clap(flatten)]
        chat_args: ChatArgs,

        #[clap(flatten)]
        single_request_args: SingleRequestArgs,

        query: Vec<String>,
    },
    /// Sends a single completion query.
    Complete {
        #[clap(flatten)]
        common_args: CommonArgs,

        #[clap(flatten)]
        single_request_args: SingleRequestArgs,

        query: Vec<String>,
    },
    /// Batch query many requests.
    BatchQuery {
        #[clap(flatten)]
        common_args: CommonArgs,

        #[clap(flatten)]
        chat_args: ChatArgs,

        #[clap(long, short = 'i')]
        input: PathBuf,

        #[clap(long, short = 'o')]
        output: PathBuf,

        #[clap(long)]
        save_raw: bool,

        #[clap(long, short = 'j', default_value_t = 16)]
        jobs: u32,

        #[clap(long)]
        quiet: bool,
    },
    /// Lists all available models.
    ListModels {
        #[clap(long)]
        url: Option<String>,
    },
    /// Starts a cache server.
    CacheServer {
        #[clap(long, default_value = "127.0.0.1")]
        host: String,

        #[clap(long, default_value_t = 9999)]
        port: u32,

        #[clap(long)]
        cache_path: Option<PathBuf>,
    },
}

fn split_blob_approximate(blob: &[u8], separator: &[u8], count: usize) -> Vec<Range<usize>> {
    if count == 0 {
        return vec![];
    }

    if blob.len() < separator.len() * 2 || count == 1 {
        return vec![0..blob.len()];
    }

    let chunk_size = blob.len() / count;
    let mut output = Vec::new();
    let mut last_index = 0;
    for nth in 0..count - 1 {
        let guess = std::cmp::max(std::cmp::max(separator.len(), chunk_size * (nth + 1)) - separator.len(), last_index);
        if let Some(index) = memchr::memmem::find(&blob[guess..], separator).map(|offset| guess + offset) {
            output.push(last_index..index);
            last_index = index + separator.len();
        } else {
            break;
        }
    }

    if last_index < blob.len() {
        output.push(last_index..blob.len());
    }

    output
}

struct Lines<'a> {
    slice: &'a [u8],
}

impl<'a> Iterator for Lines<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        if let Some(index) = memchr::memchr(b'\n', self.slice) {
            let line = &self.slice[..index];
            self.slice = &self.slice[index + 1..];
            Some(line)
        } else {
            let line = self.slice;
            self.slice = &[];
            Some(line)
        }
    }
}

impl<'a> Lines<'a> {
    fn new(mut slice: &'a [u8]) -> Self {
        while !slice.is_empty() && slice[0] == b'\n' {
            slice = &slice[1..];
        }

        while !slice.is_empty() && slice[slice.len() - 1] == b'\n' {
            slice = &slice[..slice.len() - 1];
        }

        Self { slice }
    }
}

struct LinesWithRange<'a> {
    slice: &'a [u8],
    position: usize,
}

impl<'a> Iterator for LinesWithRange<'a> {
    type Item = (&'a [u8], Range<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        let slice = &self.slice[self.position..];
        if slice.is_empty() {
            return None;
        }

        if let Some(index) = memchr::memchr(b'\n', slice) {
            let line = &slice[..index];
            let range = self.position..self.position + line.len();
            self.position += index + 1;
            Some((line, range))
        } else {
            let range = self.position..self.slice.len();
            self.position = self.slice.len();
            Some((slice, range))
        }
    }
}

impl<'a> LinesWithRange<'a> {
    fn new(mut slice: &'a [u8]) -> Self {
        while !slice.is_empty() && slice[slice.len() - 1] == b'\n' {
            slice = &slice[..slice.len() - 1];
        }

        let mut position = 0;
        while !slice[position..].is_empty() && slice[position] == b'\n' {
            position += 1;
        }

        Self { slice, position }
    }
}

#[derive(Default)]
struct Hasher(blake3::Hasher);

impl core::hash::Hasher for Hasher {
    fn finish(&self) -> u64 {
        unimplemented!();
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

type UniqueHash = [u8; 32];

fn hash_value<T>(value: &T) -> UniqueHash
where
    T: std::hash::Hash,
{
    let mut hasher = Hasher::default();
    value.hash(&mut hasher);
    *hasher.0.finalize().as_bytes()
}

#[derive(serde::Serialize)]
struct BatchOutputLine<'a> {
    response: &'a str,
    finish_reason: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    timestamp: String,
    request: serde_json::Value,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_request: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_response: Option<serde_json::Value>,
}

fn gather_hashes(raw_jsonl: &[u8], running: Option<Arc<AtomicBool>>) -> Result<HashSet<UniqueHash>, String> {
    let running = running.unwrap_or_else(|| Arc::new(AtomicBool::new(true)));
    let error_arc: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let thread_count = get_thread_count()?;
    let chunks = split_blob_approximate(&raw_jsonl, b"\n", thread_count * 16);
    let hashes: Vec<[u8; 32]> = chunks
        .into_par_iter()
        .flat_map(|chunk| {
            let mut hashes = Vec::new();
            for line in Lines::new(&raw_jsonl[chunk]) {
                if !running.load(Ordering::Relaxed) {
                    break;
                }

                if line.is_empty() {
                    continue;
                }

                let value: serde_json::Value = match serde_json::from_slice(line) {
                    Ok(value) => value,
                    Err(error) => {
                        *error_arc.lock().unwrap() = Some(format!("failed to parse jsonl file: {error}"));
                        running.store(false, Ordering::Relaxed);
                        break;
                    }
                };

                let serde_json::Value::Object(value) = value else {
                    *error_arc.lock().unwrap() = Some(format!("failed to parse jsonl file: line was not an object"));
                    running.store(false, Ordering::Relaxed);
                    break;
                };

                let Some(value) = value.get("request") else {
                    continue;
                };

                hashes.push(hash_value(&value));
            }
            hashes
        })
        .collect();

    if let Some(error) = error_arc.lock().unwrap().take() {
        return Err(error);
    }

    Ok(hashes.into_iter().collect())
}

fn get_thread_count() -> Result<usize, String> {
    Ok(std::thread::available_parallelism()
        .map_err(|error| format!("failed to get the number of threads: {error}"))?
        .get())
}

fn mmap_read(path: &Path) -> Result<memmap2::Mmap, String> {
    let output = std::fs::File::open(&path).map_err(|error| format!("failed to open {}: {}", path.display(), error))?;
    let output = unsafe { memmap2::Mmap::map(&output) }.map_err(|error| format!("failed to mmap {}: {}", path.display(), error))?;

    Ok(output)
}

async fn main_batch_query(
    mut common_args: CommonArgs,
    chat_args: ChatArgs,
    input_path: PathBuf,
    output_path: PathBuf,
    save_raw: bool,
    jobs: u32,
    quiet: bool,
) -> Result<(), String> {
    let endpoint = common_args.common_setup().await?;
    let generation_args = common_args.get_generation_args()?;
    let generation_args = Arc::new(generation_args);
    let chat_template = Arc::new(prepare_chat_request_template(&chat_args)?);

    if !quiet {
        print_logs(&endpoint, &generation_args);
    }

    let thread_count = get_thread_count()?;
    let input = mmap_read(&input_path)?;
    let input_count = if !quiet {
        let input_count: usize = {
            let chunks = split_blob_approximate(&input, b"\n", thread_count * 16);
            chunks
                .into_par_iter()
                .map(|chunk| Lines::new(&input[chunk]).filter(|line| !line.is_empty()).count())
                .sum()
        };

        eprintln!("INFO: Total input count: {input_count}");
        input_count
    } else {
        0
    };

    let soft_break = Arc::new(AtomicBool::new(false));
    let running = Arc::new(AtomicBool::new(true));
    let error_arc: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    {
        let soft_break = soft_break.clone();
        let running = running.clone();
        ctrlc::set_handler(move || {
            if soft_break.load(Ordering::Relaxed) {
                running.store(false, Ordering::Relaxed);
            } else {
                soft_break.store(true, Ordering::Relaxed);
            }
        })
        .unwrap();
    }

    let mut output_needs_newline = false;
    let hashes = if output_path.exists() {
        let output = mmap_read(&output_path)?;
        if output.last().copied() != Some(b'\n') {
            output_needs_newline = true;
        }

        if !quiet {
            eprintln!("INFO: Parsing existing outputs...");
        }

        let hashes = gather_hashes(&output, Some(running.clone()))?;
        if !running.load(Ordering::Relaxed) {
            if let Some(error) = error_arc.lock().unwrap().take() {
                return Err(error);
            } else {
                return Ok(());
            }
        }

        if !quiet {
            eprintln!("INFO: Existing outputs: found {} unique entries.", hashes.len());
        }

        hashes
    } else {
        HashSet::new()
    };

    let output_fp = std::fs::OpenOptions::new()
        .read(false)
        .write(true)
        .append(true)
        .truncate(false)
        .create(true)
        .open(&output_path)
        .map_err(|error| format!("failed to open {} for writing: {error}", output_path.display()))?;

    let mut output_fp = std::io::BufWriter::new(output_fp);
    if output_needs_newline {
        let _ = output_fp.write_all(b"\n");
    }

    let failure_threshold = u64::from(jobs) + 1;
    let failure_accumulator = Arc::new(AtomicU64::new(0));
    let skipped_count = Arc::new(AtomicU64::new(0));
    let job_count = Arc::new(AtomicU64::new(0));
    let (tx, mut rx) = tokio::sync::mpsc::channel(1024);
    let hashes = Arc::new(hashes);
    let input = Arc::new(input);
    let chunks = split_blob_approximate(&input, b"\n", jobs as usize * 16);
    let chunks: Arc<Mutex<Vec<Range<usize>>>> = Arc::new(Mutex::new(chunks));
    for _ in 0..jobs {
        let tx = tx.clone();
        let endpoint = endpoint.clone();
        let input = input.clone();
        let chunks = chunks.clone();
        let running = running.clone();
        let soft_break = soft_break.clone();
        let error_arc = error_arc.clone();
        let generation_args = generation_args.clone();
        let chat_template = chat_template.clone();
        let hashes = hashes.clone();
        let skipped_count = skipped_count.clone();
        let job_count = job_count.clone();
        let failure_accumulator = failure_accumulator.clone();

        struct DecrementOnDrop(Arc<AtomicU64>);
        impl Drop for DecrementOnDrop {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }

        job_count.fetch_add(1, Ordering::Relaxed);
        let job_guard = DecrementOnDrop(job_count);
        let task = async move {
            let tx = tx;
            let _job_guard = job_guard;
            let error = 'main_loop: loop {
                let Some(chunk) = chunks.lock().unwrap().pop() else {
                    return;
                };

                for line in Lines::new(&input[chunk]) {
                    if !running.load(Ordering::Relaxed) || soft_break.load(Ordering::Relaxed) {
                        return;
                    }

                    if line.is_empty() {
                        continue;
                    }

                    let request: serde_json::Value = match serde_json::from_slice(line) {
                        Ok(request) => request,
                        Err(error) => {
                            break 'main_loop format!("failed to parse input file: {error}");
                        }
                    };

                    let hash = hash_value(&request);
                    if hashes.contains(&hash) {
                        skipped_count.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    let serde_json::Value::Object(ref request_obj) = request else {
                        break 'main_loop format!("failed to parse input file: line was not an object");
                    };

                    let Some(prompt) = request_obj.get("prompt") else {
                        break 'main_loop format!("failed to parse input file: missing 'prompt' key");
                    };

                    let serde_json::Value::String(prompt) = prompt else {
                        break 'main_loop format!("failed to parse input file: 'prompt' key is not a string");
                    };

                    let mut kind = (*chat_template).clone();
                    kind.messages.push(openai_client::Message {
                        role: "user".into(),
                        content: prompt.clone(),
                    });

                    let req = openai_client::Request {
                        args: (*generation_args).clone(),
                        kind: openai_client::RequestKind::Chat(kind),
                    };

                    let raw_response = req.send(&endpoint).await;
                    let response = match extract_response(&raw_response) {
                        Ok(response) => response,
                        Err(error) => {
                            let failure_count = failure_accumulator.fetch_add(1, Ordering::Relaxed);
                            if failure_count >= failure_threshold {
                                break 'main_loop format!("failed to extract response: {error}");
                            } else {
                                let stderr = std::io::stderr();
                                let mut stderr = stderr.lock();
                                let _ = writeln!(&mut stderr, "\nERROR: {}", error);
                                continue;
                            }
                        }
                    };

                    failure_accumulator.store(0, Ordering::Relaxed);

                    let now = time::OffsetDateTime::now_local().unwrap();
                    let entry = BatchOutputLine {
                        response: &response.text,
                        finish_reason: match response.finish_reason {
                            None => {
                                let stderr = std::io::stderr();
                                let mut stderr = stderr.lock();
                                let _ = writeln!(&mut stderr, "\nERROR: missing finish reason!");
                                if let Some(raw) = raw_response.raw {
                                    let _ = writeln!(&mut stderr, "DEBUG: RAW RESPONSE:\n#{raw}");
                                }

                                break 'main_loop format!("missing finish reason");
                            }
                            Some(openai_client::FinishReason::Length) => "length",
                            Some(openai_client::FinishReason::Stop) => "stop",
                        },
                        reasoning_content: response.reasoning_content.as_ref().map(|text| &**text),
                        timestamp: now.format(&time::format_description::well_known::Rfc3339).unwrap(),
                        model: &response.model,
                        request,
                        raw_request: if save_raw { raw_response.raw_request_json() } else { None },
                        raw_response: if save_raw { raw_response.raw_json() } else { None },
                    };

                    let mut output_line = serde_json::to_string(&entry).unwrap();
                    output_line.push('\n');
                    if tx.send((output_line, response.usage.clone())).await.is_err() {
                        return;
                    }
                }
            };

            *error_arc.lock().unwrap() = Some(error);
            soft_break.store(true, Ordering::Relaxed);
        };

        tokio::task::spawn(task);
    }

    core::mem::drop(tx);

    if !quiet {
        eprintln!("INFO: Running...");
    }

    struct Stats {
        items: u64,
        tokens_prompt: u64,
        tokens_completion: u64,
    }

    let error_result = error_arc.clone();
    let skipped_count_clone = skipped_count.clone();
    let spawn_result = tokio::task::spawn(async move {
        let mut flush_interval = tokio::time::interval(Duration::from_secs(3));
        let mut progress_interval = tokio::time::interval(Duration::from_secs(1));
        let mut stats = Stats {
            items: 0,
            tokens_prompt: 0,
            tokens_completion: 0,
        };
        struct History {
            items: u64,
            timestamp: std::time::Instant,
            tokens_prompt: u64,
            tokens_completion: u64,
        }
        let mut history: std::collections::VecDeque<History> = Default::default();
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }

            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some((output_line, usage)) => {
                            if let Some(usage) = usage {
                                stats.tokens_prompt += usage.prompt_tokens;
                                stats.tokens_completion += usage.completion_tokens;
                            }
                            stats.items += 1;
                            if let Err(error) = output_fp.write_all(output_line.as_bytes()) {
                                *error_arc.lock().unwrap() = Some(format!("failed to write to {}: {error}", output_path.display()));
                                running.store(false, Ordering::Relaxed);
                                return stats;
                            }
                        },
                        None => {
                            if !quiet {
                                eprint!("\r\x1b[2K");
                            }

                            break;
                        },
                    }
                }

                 _ = flush_interval.tick() => {
                    if let Err(error) = output_fp.flush() {
                        *error_arc.lock().unwrap() = Some(format!("failed to flush {}: {error}", output_path.display()));
                        running.store(false, Ordering::Relaxed);
                        return stats;
                    }
                }

                _ = progress_interval.tick() => {
                    if !quiet {
                        let stderr = std::io::stderr();
                        let mut stderr = stderr.lock();
                        let completed = skipped_count_clone.load(Ordering::Relaxed) + stats.items;
                        let job_count = job_count.load(Ordering::Relaxed);
                        let mut stats_changed = true;
                        let now = std::time::Instant::now();

                        if history.len() >= 2 {
                            if let Some(last_history) = history.back_mut() {
                                if last_history.items == stats.items && last_history.tokens_prompt == stats.tokens_prompt && last_history.tokens_completion == stats.tokens_completion {
                                    last_history.timestamp = now;
                                    stats_changed = false;
                                }
                            }
                        }

                        if stats_changed {
                            history.push_back(History {
                                timestamp: now,
                                items: stats.items,
                                tokens_prompt: stats.tokens_prompt,
                                tokens_completion: stats.tokens_completion,
                            });
                        }

                        while history.len() >= 32 {
                            history.pop_front();
                        }

                        let progress = (completed as f32 / input_count as f32 * 100.0) as u32;
                        let _ = write!(&mut stderr, "\r\x1b[2K[{progress}%] {completed}/{input_count}");
                        if job_count == 1 {
                            let _ = write!(&mut stderr, ", 1 job");
                        } else if job_count > 1 {
                            let _ = write!(&mut stderr, ", {job_count} jobs");
                        }

                        if history.len() >= 2 {
                            let start = history.front().unwrap();
                            let end = history.back().unwrap();
                            let elapsed = end.timestamp - start.timestamp;
                            let items = end.items - start.items;
                            let tokens_prompt = end.tokens_prompt - start.tokens_prompt;
                            let tokens_completion = end.tokens_completion - start.tokens_completion;
                            let pp_per_sec = (tokens_prompt as f32 / elapsed.as_secs_f32()) as u32;
                            let tg_per_sec = (tokens_completion as f32 / elapsed.as_secs_f32()) as u32;
                            let items_per_sec = items as f32 / elapsed.as_secs_f32();
                            let _ = write!(&mut stderr, ", {items_per_sec:.01} it/s, {pp_per_sec} pp/s, {tg_per_sec} tg/s");

                            if completed > 0 {
                                let remaining = (input_count as i64 - completed as i64).max(0);
                                let seconds = remaining as f32 / items_per_sec;
                                let (minutes, seconds) = if seconds <= 60.0 {
                                    (0, seconds as u32)
                                } else {
                                    let minutes = (seconds / 60.0) as u32;
                                    let seconds = (seconds - minutes as f32 * 60.0) as u32;
                                    (minutes, seconds)
                                };

                                let _ = write!(&mut stderr, ", ETA: {minutes}m {seconds}s");
                            }
                        }

                        if soft_break.load(Ordering::Relaxed) {
                            let _ = write!(&mut stderr, " SHUTTING DOWN");
                        }
                    }
                }
            }
        }

        if let Err(error) = output_fp.flush() {
            *error_arc.lock().unwrap() = Some(format!("failed to flush {}: {error}", output_path.display()));
            running.store(false, Ordering::Relaxed);
        }

        stats
    }).await;

    if !quiet {
        eprintln!("\r");
    }

    match spawn_result {
        Ok(stats) => {
            if !quiet {
                eprintln!(
                    "INFO: Processed {}/{} items(s) ({} processed, {} skipped), {} tokens ({} pp, {} tg)",
                    stats.items + skipped_count.load(Ordering::Relaxed),
                    input_count,
                    stats.items,
                    skipped_count.load(Ordering::Relaxed),
                    stats.tokens_prompt + stats.tokens_completion,
                    stats.tokens_prompt,
                    stats.tokens_completion
                );
            }
        }
        Err(error) => {
            return Err(format!("failed to spawn a task: {error}"));
        }
    }

    if let Some(error) = error_result.lock().unwrap().take() {
        return Err(error);
    }

    if !quiet {
        eprintln!("INFO: Finished!");
    }

    Ok(())
}

const DEFAULT_LOCAL_PORT: u32 = 9001;

fn extract_response(response: &openai_client::Response) -> Result<&openai_client::ResponseOk, String> {
    match response.obj {
        Ok(Ok(ref response)) => return Ok(response),
        Ok(Err(ref error)) => {
            return Err({
                use std::fmt::Write;
                let mut buffer = String::new();

                let _ = write!(&mut buffer, "received an error response: code={}, ", error.code);
                if let Some(ref kind) = error.kind {
                    let _ = write!(&mut buffer, "type=\"{}\", ", kind);
                }

                let _ = write!(&mut buffer, "error message: {:?}", error.message);
                buffer
            });
        }
        Err(ref error) => {
            if let Some(raw) = response.raw_json() {
                eprintln!("DEBUG: raw response (JSON): {}", serde_json::to_string_pretty(&raw).unwrap());
            } else if let Some(ref raw) = response.raw {
                eprintln!("DEBUG: raw response (string): {raw}");
            }
            return Err(error.to_string());
        }
    }
}

enum RequestKind {
    Completion,
    Chat(ChatArgs),
}

impl CommonArgs {
    fn get_generation_args(&self) -> Result<openai_client::GenerationArgs, String> {
        Ok(openai_client::GenerationArgs {
            model: match self.model {
                Some(ref value) => value.clone(),
                None => return Err("no model specified".into()),
            },
            seed: self.seed,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_penalty: self.repetition_penalty,
            repetition_penalty_range: self.repetition_penalty_range,
            request_prompt_caching: self.request_prompt_caching,
            priority: self.niceness,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
        })
    }

    async fn common_setup(&mut self) -> Result<Endpoint, String> {
        if self.reproducible {
            if self.temperature.is_none() {
                self.temperature = Some(0.0);
            }
            if self.seed.is_none() {
                self.seed = Some(2349857);
            }
        }

        let mut endpoint = if let Some(ref url) = self.url {
            Endpoint {
                url: url.clone(),
                api_key: self.api_key.clone().unwrap_or(String::new()),
                providers: Vec::new(),
                allow_fallbacks: true,
            }
        } else {
            if let Some(ref model) = self.model {
                if model.contains("/") {
                    if let Some(ref api_key) = self.api_key {
                        Endpoint::openrouter(api_key.clone())
                    } else if let Some(api_key) = std::env::var("HOME")
                        .ok()
                        .and_then(|home| std::fs::read_to_string(PathBuf::from(home).join(".openrouter-key.txt")).ok())
                    {
                        Endpoint::openrouter(api_key)
                    } else {
                        return Err("no API key specified".into());
                    }
                } else {
                    Endpoint::local(DEFAULT_LOCAL_PORT)
                }
            } else {
                let endpoint = Endpoint::local(DEFAULT_LOCAL_PORT);
                let models = openai_client::fetch_models(&endpoint).await?;
                let Some(model) = models.first() else {
                    return Err("no models found".into());
                };
                self.model = Some(model.name.clone());
                endpoint
            }
        };

        if let Some(ref provider) = self.provider {
            for provider in provider.split(',') {
                endpoint.providers.push(provider.to_owned());
            }
        }

        if !endpoint.providers.is_empty() {
            endpoint.allow_fallbacks = false;
        }

        Ok(endpoint)
    }
}

fn prepare_chat_request_template(chat_args: &ChatArgs) -> Result<openai_client::ChatRequest, String> {
    let ChatArgs {
        ref system_prompt,
        disable_thinking,
        ref reasoning_effort,
        ref output_format_choice,
        output_format,
        ref json_schema,
    } = *chat_args;
    let mut messages = Vec::new();
    if let Some(content) = system_prompt {
        messages.push(openai_client::Message {
            role: "system".into(),
            content: content.to_owned(),
        });
    }

    Ok(openai_client::ChatRequest {
        messages,
        disable_thinking,
        reasoning_effort: reasoning_effort.clone(),
        schema: if let Some(output_format_choice) = output_format_choice {
            Some(openai_client::Schema::Choice(
                output_format_choice.split(',').map(|s| s.to_owned()).collect(),
            ))
        } else if let Some(schema_path) = json_schema {
            match std::fs::read_to_string(&schema_path) {
                Ok(schema) => Some(openai_client::Schema::JsonSchema(schema)),
                Err(error) => {
                    return Err(format!("failed to read JSON schema from {}: {error}", schema_path.display()));
                }
            }
        } else {
            match output_format {
                Some(OutputFormat::JsonObject) => Some(openai_client::Schema::JsonObject),
                Some(OutputFormat::JsonArrayOfStrings) => Some(openai_client::Schema::JsonArrayOfStrings),
                Some(OutputFormat::JsonArrayOfObjects) => Some(openai_client::Schema::JsonArrayOfObjects),
                Some(OutputFormat::JsonArrayOfArrays) => Some(openai_client::Schema::JsonArrayOfArrays),
                None => None,
            }
        },
    })
}

fn print_logs(endpoint: &Endpoint, args: &openai_client::GenerationArgs) {
    eprintln!("INFO: URL: '{}'", endpoint.url);
    eprintln!("INFO: Model: '{}'", args.model);
    if let Some(seed) = args.seed {
        eprintln!("INFO: Seed: {}", seed);
    }
    if let Some(temperature) = args.temperature {
        eprintln!("INFO: Temperature: {}", temperature);
    }
    if let Some(frequency_penalty) = args.frequency_penalty {
        eprintln!("INFO: Frequency penalty: {}", frequency_penalty);
    }
    if let Some(presence_penalty) = args.presence_penalty {
        eprintln!("INFO: Presence penalty: {}", presence_penalty);
    }
    if let Some(repetition_penalty) = args.repetition_penalty {
        eprintln!("INFO: Repetition penalty: {}", repetition_penalty);
    }
    if let Some(repetition_penalty_range) = args.repetition_penalty_range {
        eprintln!("INFO: Frequency penalty range: {}", repetition_penalty_range);
    }
}

async fn main_single_request(
    mut common_args: CommonArgs,
    query: Vec<String>,
    kind: RequestKind,
    thinking: Thinking,
    single_request_args: SingleRequestArgs,
) -> Result<(), String> {
    let SingleRequestArgs { streaming, verbose, stdin } = single_request_args;

    use std::io::IsTerminal;
    let is_terminal = std::io::stdout().is_terminal();
    let streaming = match streaming {
        IsEnabled::On => true,
        IsEnabled::Off => false,
        IsEnabled::Auto => is_terminal,
    };

    let hide_thinking = match thinking {
        Thinking::Show => false,
        Thinking::Hide => true,
        Thinking::Auto => !is_terminal,
    };

    let mut prompt = query.join(" ").replace("\\n", "\n");
    let read_from_stdin = match stdin {
        IsEnabled::On => true,
        IsEnabled::Auto if !std::io::stdin().is_terminal() => true,
        IsEnabled::Off | IsEnabled::Auto => false,
    };

    if read_from_stdin {
        use std::io::Read;
        if std::io::stdin().read_to_string(&mut prompt).is_err() {
            return Err("failed to read from stdin".into());
        }
    }

    if prompt.is_empty() && matches!(kind, RequestKind::Completion) {
        return Err("missing prompt".into());
    }

    let endpoint = common_args.common_setup().await?;
    if is_terminal && streaming && endpoint.is_local() && common_args.niceness.is_none() {
        common_args.niceness = Some(-1);
    }

    let request = openai_client::Request {
        args: common_args.get_generation_args()?,
        kind: match kind {
            RequestKind::Completion => openai_client::RequestKind::Completion(openai_client::CompletionRequest { prompt }),
            RequestKind::Chat(ref chat_args) => {
                let mut req = prepare_chat_request_template(chat_args)?;
                req.messages.push(openai_client::Message {
                    role: "user".into(),
                    content: prompt,
                });
                openai_client::RequestKind::Chat(req)
            }
        },
    };

    let use_cache = !streaming;
    let cache_address = "127.0.0.1:9999";
    let mut cached_response: Option<openai_client::ResponseOk> = None;
    let request_for_cache = if use_cache {
        let mut request_for_cache = request.clone();
        request_for_cache.args.max_tokens = None;
        request_for_cache.args.priority = None;

        let request_for_cache = serde_json::to_value(&request_for_cache).unwrap();
        match cache_get(cache_address, &request_for_cache).await {
            Ok(Some(response)) => {
                // Converting it back to string is silly, but whatever.
                let response = openai_client::Response::from_raw(&serde_json::to_string(&response).unwrap(), Arc::new(String::new()));
                if let Ok(Ok(response_ok)) = response.obj {
                    cached_response = Some(response_ok);
                }

                Some(request_for_cache)
            }
            Ok(None) => Some(request_for_cache),
            Err(_) => None,
        }
    } else {
        None
    };

    if verbose {
        print_logs(&endpoint, &request.args);
    }

    if streaming {
        let mut is_first = true;
        let mut is_thinking = false;
        let mut stream = request.send_streaming(&endpoint).await.map_err(|error| error.to_string())?;

        while let Some(chunk) = stream.next().await {
            let response = extract_response(&chunk)?;
            let out = std::io::stdout();
            let mut out = out.lock();
            if is_first {
                match request.kind {
                    openai_client::RequestKind::Completion(ref request) => {
                        if out.write_all(request.prompt.as_bytes()).is_err() {
                            return Ok(());
                        };
                    }
                    openai_client::RequestKind::Chat(..) => {}
                }

                is_first = false;
            }

            if let Some(ref reasoning_content) = response.reasoning_content {
                if !hide_thinking {
                    if !is_thinking {
                        is_thinking = true;
                        if out.write_all("<think>".as_bytes()).is_err() {
                            return Ok(());
                        };
                    }

                    if out.write_all(reasoning_content.as_bytes()).is_err() {
                        return Ok(());
                    };
                }
            }

            if !response.text.is_empty() {
                if is_thinking && !hide_thinking {
                    is_thinking = false;
                    if out.write_all("</think>\n\n".as_bytes()).is_err() {
                        return Ok(());
                    };
                }

                if out.write_all(response.text.as_bytes()).is_err() {
                    return Ok(());
                };
            }

            if out.flush().is_err() {
                return Ok(());
            }
        }

        println!();
    } else {
        let response = if let Some(response_ok) = cached_response {
            response_ok
        } else {
            let response = request.send(&endpoint).await;
            let response_ok = extract_response(&response)?;
            if let Some(request_for_cache) = request_for_cache {
                if response_ok.finish_reason == Some(openai_client::FinishReason::Stop) {
                    if let Some(ref raw_response) = response.raw {
                        if let Ok(raw_response) = serde_json::from_str(&raw_response) {
                            if let Err(error) = cache_put(cache_address, &request_for_cache, &raw_response).await {
                                eprintln!("ERROR: Failed to cache request: {error}");
                            }
                        }
                    }
                }
            }

            response_ok.clone()
        };
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        match request.kind {
            openai_client::RequestKind::Completion(ref request) => {
                let _ = stdout.write_all(&request.prompt.as_bytes());
                let _ = stdout.write_all(&response.text.as_bytes());
            }
            openai_client::RequestKind::Chat(..) => {
                if !hide_thinking {
                    if let Some(ref reasoning_content) = response.reasoning_content {
                        let _ = writeln!(stdout, "<think>{}</think>\n", reasoning_content);
                    }
                }

                let _ = stdout.write_all(&response.text.as_bytes());
            }
        }
        if is_terminal && !response.text.ends_with("\n") {
            let _ = stdout.write_all("\n".as_bytes());
        }
        let _ = stdout.flush();
    }
    Ok(())
}

async fn main_list_models(url: Option<String>) -> Result<(), String> {
    let endpoint = if let Some(url) = url {
        Endpoint::new(url)
    } else {
        Endpoint::openrouter("".into())
    };

    let models = openai_client::fetch_models(&endpoint).await?;
    let models: Vec<_> = models.into_iter().map(|info| info.raw_info).collect();
    println!("{}", serde_json::to_string_pretty(&models).unwrap());

    Ok(())
}

fn load_server_cache(blob: &[u8]) -> Result<HashMap<UniqueHash, Range<usize>>, String> {
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

        let Some(raw_request) = obj.get("request") else {
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

async fn main_cache_server_impl(
    address: &str,
    cache_path: Option<PathBuf>,
) -> Result<Pin<Box<dyn futures::Future<Output = ()> + Send + 'static>>, String> {
    struct Cache {
        blob: Option<memmap2::Mmap>,
        map_cold: RwLock<HashMap<UniqueHash, Range<usize>>>,
        map_hot: RwLock<HashMap<UniqueHash, serde_json::Value>>,
        fp: Option<Mutex<std::io::BufWriter<std::fs::File>>>,
    }

    let mut cache = Cache {
        blob: None,
        map_cold: RwLock::new(HashMap::new()),
        map_hot: RwLock::new(HashMap::new()),
        fp: None,
    };

    let verbose = false;
    if let Some(ref cache_path) = cache_path {
        let mut output_needs_newline = false;

        if cache_path.exists() {
            eprintln!("INFO: Loading cache: {}", cache_path.display());
            let blob = mmap_read(&cache_path)?;
            if !blob.is_empty() && blob.last().copied() != Some(b'\n') {
                output_needs_newline = true;
            }

            let map_cold = load_server_cache(&blob)?;
            eprintln!("INFO: Loaded cache: {} entries", map_cold.len());

            cache.map_cold = RwLock::new(map_cold);
            cache.blob = Some(blob);
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

        cache.fp = Some(Mutex::new(fp));
    } else {
        eprintln!("INFO: Running in memory-only mode");
    }

    let cache = Arc::new(cache);
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| format!("failed to bind to address: {}", error))?;

    eprintln!("INFO: Cache server listening on {address}");
    let task = async move {
        let mut flush_interval = tokio::time::interval(Duration::from_secs(3));
        loop {
            let accept_result = tokio::select! {
                client = listener.accept() => {
                    client
                }
                _ = flush_interval.tick() => {
                    if let Some(ref fp) = cache.fp {
                        let _ = fp.lock().unwrap().flush();
                    }

                    continue;
                }
            };

            let (mut socket, addr) = match accept_result {
                Ok(result) => result,
                Err(error) => {
                    eprintln!("ERROR: Failed to accept connection: {}", error);
                    continue;
                }
            };

            let cache = cache.clone();

            tokio::spawn(async move {
                if verbose {
                    eprintln!("INFO: Connection from {}", addr);
                }

                let mut buffer = vec![0u8; 1024];
                let mut request_data = Vec::new();

                loop {
                    match socket.read(&mut buffer).await {
                        Ok(0) => {
                            if verbose {
                                eprintln!("INFO: Connection from {} closed", addr);
                            }

                            return;
                        }
                        Ok(n) => {
                            request_data.extend_from_slice(&buffer[..n]);

                            // Check if we have a complete request (ends with newline)
                            if request_data.contains(&b'\n') {
                                break;
                            }
                        }
                        Err(error) => {
                            if verbose {
                                eprintln!("ERROR: Failed to read from {}: {}", addr, error);
                            }
                            return;
                        }
                    }
                }

                let request: CacheRequest = match serde_json::from_slice(&request_data) {
                    Ok(request) => request,
                    Err(error) => {
                        if verbose {
                            eprintln!("ERROR: Failed to parse request: {error}");
                        }

                        return;
                    }
                };

                if verbose {
                    eprintln!("INFO: Received request from {}: {:?}", addr, request);
                }

                let response = match request {
                    CacheRequest::Get { key } => {
                        let key_hash = hash_value(&key);

                        let mut value = {
                            let map = cache.map_hot.read().unwrap();
                            map.get(&key_hash).cloned()
                        };

                        if value.is_none() {
                            if let Some(ref blob) = cache.blob {
                                value = get_from_storage(key_hash, &blob, &cache.map_cold);
                            }
                        }

                        match value {
                            Some(value) => {
                                let response = CacheResponse::Found(value.clone());
                                match serde_json::to_string(&response) {
                                    Ok(mut json_str) => {
                                        json_str.push('\n');
                                        Ok(json_str)
                                    }
                                    Err(error) => Err(format!("failed to serialize response: {}", error)),
                                }
                            }
                            None => {
                                let response = CacheResponse::NotFound;
                                match serde_json::to_string(&response) {
                                    Ok(json_str) => Ok(format!("{}\n", json_str)),
                                    Err(error) => Err(format!("failed to serialize response: {}", error)),
                                }
                            }
                        }
                    }
                    CacheRequest::Put { key, value } => {
                        let key_hash = hash_value(&key);

                        let value_clone = value.clone();
                        let mut map_hot = cache.map_hot.write().unwrap();
                        let write_to_file = match map_hot.entry(key_hash) {
                            std::collections::hash_map::Entry::Occupied(_) => false,
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                entry.insert(value_clone);
                                true
                            }
                        };
                        core::mem::drop(map_hot);

                        if write_to_file {
                            if let Some(ref fp) = cache.fp {
                                let entry = serde_json::json!({
                                    "request": key,
                                    "raw_response": value,
                                });
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

                        Ok("".into())
                    }
                };

                let response = match response {
                    Ok(response) => {
                        if verbose {
                            eprintln!("INFO: Sending response: {}", response);
                        }

                        response
                    }
                    Err(error) => {
                        if verbose {
                            eprintln!("ERROR: Failed to process request: {}", error);
                        }

                        format!("ERROR: {}\n", error)
                    }
                };

                if let Err(error) = socket.write_all(response.as_bytes()).await {
                    if verbose {
                        eprintln!("ERROR: Failed to write to {}: {}", addr, error);
                    }
                }
            });
        }
    };

    Ok(Box::pin(task))
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CacheRequest {
    Get { key: serde_json::Value },
    Put { key: serde_json::Value, value: serde_json::Value },
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
enum CacheResponse {
    Found(serde_json::Value),
    NotFound,
}

async fn main_cache_server(address: &str, cache_path: Option<PathBuf>) -> Result<(), String> {
    Ok(main_cache_server_impl(address, cache_path).await?.await)
}

async fn cache_get(address: &str, key: &serde_json::Value) -> Result<Option<serde_json::Value>, String> {
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

async fn cache_put(addr: &str, key: &serde_json::Value, value: &serde_json::Value) -> Result<(), String> {
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

fn small_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_time()
        .enable_io()
        .build()
        .unwrap()
}

fn big_runtime() -> tokio::runtime::Runtime {
    let thread_count = get_thread_count().unwrap();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(thread_count + 2)
        .enable_time()
        .enable_io()
        .build()
        .unwrap()
}

fn main() {
    let args = Args::parse();
    let error = match args {
        Args::Complete {
            common_args,
            query,
            single_request_args,
        } => small_runtime().block_on(main_single_request(
            common_args,
            query,
            RequestKind::Completion,
            Thinking::Auto,
            single_request_args,
        )),
        Args::Q {
            thinking,
            common_args,
            chat_args,
            query,
            single_request_args,
        } => small_runtime().block_on(main_single_request(
            common_args,
            query,
            RequestKind::Chat(chat_args),
            thinking,
            single_request_args,
        )),
        Args::BatchQuery {
            common_args,
            chat_args,
            input,
            output,
            save_raw,
            jobs,
            quiet,
        } => big_runtime().block_on(main_batch_query(common_args, chat_args, input, output, save_raw, jobs, quiet)),
        Args::ListModels { url } => small_runtime().block_on(main_list_models(url)),
        Args::CacheServer { host, port, cache_path } => big_runtime().block_on(main_cache_server(&format!("{host}:{port}"), cache_path)),
    };

    if let Err(error) = error {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_server() {
        small_runtime().block_on(test_cache_server_impl()).unwrap();
    }

    async fn test_cache_server_impl() -> Result<(), String> {
        let address = "127.0.0.1:9998";
        let task = main_cache_server_impl(address, None).await?;
        tokio::spawn(task);

        let test_key = serde_json::json!("test_key");
        let test_value = serde_json::json!("test_value");

        cache_put(&address, &test_key, &test_value).await?;

        let retrieved_value = cache_get(&address, &test_key).await?;
        assert_eq!(retrieved_value, Some(test_value));

        let non_existent_key = serde_json::json!("non_existent_key");
        let retrieved_value = cache_get(&address, &non_existent_key).await?;
        assert_eq!(retrieved_value, None);

        Ok(())
    }
}
