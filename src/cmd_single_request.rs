use futures::prelude::*;
use std::io::Write;
use std::sync::Arc;

use crate::openai_client;
use crate::utils::{extract_response, prepare_chat_request_template, print_logs};
use crate::{CommonArgs, IsEnabled, RequestKind, SingleRequestArgs, Thinking};

pub async fn main_single_request(
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
        match crate::cache::cache_get(cache_address, &request_for_cache).await {
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
                            if let Err(error) = crate::cache::cache_put(cache_address, &request_for_cache, &raw_response).await {
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
