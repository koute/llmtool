use core::ops::Range;
use std::path::Path;

use crate::openai_client;
use crate::openai_client::Endpoint;
use crate::{ChatArgs, OutputFormat};

pub fn prepare_chat_request_template(chat_args: &ChatArgs) -> Result<openai_client::ChatRequest, String> {
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

pub fn extract_response(response: &openai_client::Response) -> Result<&openai_client::ResponseOk, String> {
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

pub fn get_thread_count() -> Result<usize, String> {
    Ok(std::thread::available_parallelism()
        .map_err(|error| format!("failed to get the number of threads: {error}"))?
        .get())
}

pub fn mmap_read(path: &Path) -> Result<memmap2::Mmap, String> {
    let output = std::fs::File::open(&path).map_err(|error| format!("failed to open {}: {}", path.display(), error))?;
    let output = unsafe { memmap2::Mmap::map(&output) }.map_err(|error| format!("failed to mmap {}: {}", path.display(), error))?;

    Ok(output)
}

pub fn split_blob_approximate(blob: &[u8], separator: &[u8], count: usize) -> Vec<Range<usize>> {
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

pub struct Lines<'a> {
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
    pub fn new(mut slice: &'a [u8]) -> Self {
        while !slice.is_empty() && slice[0] == b'\n' {
            slice = &slice[1..];
        }

        while !slice.is_empty() && slice[slice.len() - 1] == b'\n' {
            slice = &slice[..slice.len() - 1];
        }

        Self { slice }
    }
}

pub struct LinesWithRange<'a> {
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
    pub fn new(mut slice: &'a [u8]) -> Self {
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

pub fn print_logs(endpoint: &Endpoint, args: &openai_client::GenerationArgs) {
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

pub type UniqueHash = [u8; 32];

pub fn hash_value<T>(value: &T) -> UniqueHash
where
    T: std::hash::Hash,
{
    let mut hasher = Hasher::default();
    value.hash(&mut hasher);
    *hasher.0.finalize().as_bytes()
}
