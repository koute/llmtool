use ahash::AHashSet as HashSet;
use core::time::Duration;
use rayon::prelude::*;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::openai_client;
use crate::utils::{
    Lines, UniqueHash, extract_response, get_thread_count, hash_value, mmap_read, prepare_chat_request_template, print_logs,
    split_blob_approximate,
};
use crate::{ChatArgs, CommonArgs};

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

pub async fn main_batch_query(
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
