use futures::prelude::*;
use std::io::Write;

use crate::openai_client;
use crate::utils::{extract_response, prepare_chat_request_template, print_logs};
use crate::{CommonArgs, DisplayThinking, IsEnabled, MessageFormat, RequestKind, SingleRequestArgs};
use std::path::PathBuf;
use base64::Engine;

async fn cache_response(
    cache_address: &str,
    request_for_cache: &serde_json::Value,
    response: &openai_client::Response,
    response_ok: &openai_client::ResponseOk,
) -> Result<(), String> {
    if response_ok.finish_reason == Some(openai_client::FinishReason::Stop) {
        if let Some(ref raw_response) = response.raw {
            if let Ok(raw_response) = serde_json::from_str(&raw_response) {
                if let Err(error) = crate::cache_client::cache_put(cache_address, &request_for_cache, &raw_response).await {
                    eprintln!("ERROR: Failed to cache request: {error}");
                }
            }
        }
    }

    Ok(())
}

pub async fn main_single_request(
    mut common_args: CommonArgs,
    query: Vec<String>,
    kind: RequestKind,
    display_thinking: DisplayThinking,
    single_request_args: SingleRequestArgs,
    image: Option<PathBuf>,
    resize: Option<(u32, u32)>,
    contrast: Option<f32>,
    saturation: Option<f32>,
) -> Result<(), String> {
    let SingleRequestArgs {
        streaming,
        verbose,
        stdin,
        print_raw_request,
        print_raw_response,
        disable_cache,
    } = single_request_args;

    use std::io::IsTerminal;
    let is_terminal = std::io::stdout().is_terminal();
    let mut streaming = match streaming {
        IsEnabled::On => true,
        IsEnabled::Off => false,
        IsEnabled::Auto => is_terminal,
    };

    let hide_thinking = match display_thinking {
        DisplayThinking::Show => false,
        DisplayThinking::Hide => true,
        DisplayThinking::Auto => !is_terminal,
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

    let mut output_reply_as_json = false;
    let request = openai_client::Request {
        args: common_args.get_generation_args()?,
        kind: match kind {
            RequestKind::Completion => openai_client::RequestKind::Completion(openai_client::CompletionRequest { prompt }),
            RequestKind::Chat(ref chat_args, ref schema_args, input_message_format, output_message_format) => {
                let mut req = prepare_chat_request_template(chat_args, schema_args)?;
                match input_message_format {
                    MessageFormat::Text => {
                        req.messages.push(openai_client::Message::new("user".into(), prompt));
                        if let Some(image_path) = image {
                            let image_data = std::fs::read(&image_path)
                                .map_err(|e| format!("failed to read image '{}': {e}", image_path.display()))?;
                            let (image_data, mime) = if resize.is_some() || contrast.is_some() || saturation.is_some() {
                                let mut img = image::load_from_memory(&image_data)
                                    .map_err(|e| format!("failed to decode image: {e}"))?;

                                if let Some(contrast) = contrast {
                                    image::imageops::colorops::contrast_in_place(&mut img, contrast);
                                }

                                if let Some(saturation) = saturation {
                                    let mut rgb = img.into_rgb8();
                                    for px in rgb.pixels_mut() {
                                        let [r, g, b] = px.0;
                                        let (h, s, l) = rgb_to_hsl(r, g, b);
                                        let s = (s * (1.0 + saturation)).clamp(0.0, 1.0);
                                        let (r, g, b) = hsl_to_rgb(h, s, l);
                                        px.0 = [r, g, b];
                                    }
                                    img = rgb.into();
                                }

                                if let Some((w, h)) = resize {
                                    img = img.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
                                }

                                let mut buf = Vec::new();
                                let is_png = image_path.extension().and_then(|e| e.to_str()) == Some("png");
                                let format = if is_png { image::ImageFormat::Png } else { image::ImageFormat::Jpeg };
                                let mime = if is_png { "image/png" } else { "image/jpeg" };
                                img.write_to(&mut std::io::Cursor::new(&mut buf), format)
                                    .map_err(|e| format!("failed to encode image: {e}"))?;
                                (buf, mime)
                            } else {
                                let mime = match image_path.extension().and_then(|e| e.to_str()) {
                                    Some("png") => "image/png",
                                    Some("jpg") | Some("jpeg") => "image/jpeg",
                                    Some("webp") => "image/webp",
                                    Some("gif") => "image/gif",
                                    _ => return Err(format!("unsupported image format: {}", image_path.display())),
                                };
                                (image_data, mime)
                            };
                            let b64 = base64::engine::general_purpose::STANDARD.encode(&image_data);
                            let data_url = format!("data:{mime};base64,{b64}");
                            if let Some(last) = req.messages.last_mut() {
                                last.images.push(data_url);
                            }
                        }
                    }
                    MessageFormat::Json => {
                        let prompt: Result<Vec<openai_client::Message>, _> = serde_json::from_str(&prompt);
                        let prompt = match prompt {
                            Ok(prompt) => prompt,
                            Err(error) => {
                                return Err(format!("failed to parse prompt: {error}"));
                            }
                        };

                        req.messages = prompt;
                    }
                }

                match output_message_format {
                    MessageFormat::Text => {}
                    MessageFormat::Json => {
                        output_reply_as_json = true;
                        streaming = false; // TODO: Make it work?
                    }
                }

                openai_client::RequestKind::Chat(req)
            }
        },
    };

    let use_cache = !disable_cache;
    let cache_address = "127.0.0.1:9999";
    let mut cached_response: Option<(openai_client::ResponseOk, Option<String>)> = None;
    let request_for_cache = if use_cache {
        let mut request_for_cache = request.clone();
        request_for_cache.args.max_tokens = None;
        request_for_cache.args.priority = None;

        let request_for_cache = serde_json::to_value(&request_for_cache).unwrap();
        match crate::cache_client::cache_get(cache_address, &request_for_cache).await {
            Ok(Some(response)) => {
                // Converting it back to string is silly, but whatever.
                let response = openai_client::Response::from_raw(&serde_json::to_string(&response).unwrap(), None);
                if let Ok(Ok(response_ok)) = response.obj {
                    cached_response = Some((response_ok, response.raw));
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

    if print_raw_request {
        if let Ok(request) = request.serialize_request(&endpoint, false) {
            println!("{request}");
        }
    }

    if streaming && cached_response.is_none() {
        let mut is_first = true;
        let mut is_thinking = false;
        let mut stream = request
            .send_streaming(&endpoint, use_cache || print_raw_response)
            .await
            .map_err(|error| error.to_string())?;

        while let Some(chunk) = stream.next().await {
            if print_raw_response {
                if let Some(raw) = chunk.raw {
                    println!("{raw}");
                }
                continue;
            }

            let response = extract_response(&chunk)?;
            if response.is_reconstructed() {
                if let Some(request_for_cache) = request_for_cache {
                    cache_response(cache_address, &request_for_cache, &chunk, response).await?;
                }

                break;
            }

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

            if !response.content.is_empty() {
                if is_thinking && !hide_thinking {
                    is_thinking = false;
                    if out.write_all("</think>\n\n".as_bytes()).is_err() {
                        return Ok(());
                    };
                }

                if out.write_all(response.content.as_bytes()).is_err() {
                    return Ok(());
                };
            }

            if out.flush().is_err() {
                return Ok(());
            }
        }

        println!();
    } else {
        let (response, response_raw) = if let Some(response) = cached_response {
            response
        } else {
            let response = request.send(&endpoint).await;
            let response_ok = extract_response(&response)?;
            if let Some(request_for_cache) = request_for_cache {
                cache_response(cache_address, &request_for_cache, &response, response_ok).await?;
            }

            (response_ok.clone(), response.raw)
        };
        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();
        if print_raw_response {
            if let Some(raw) = response_raw {
                let _ = stdout.write_all(raw.as_bytes());
            } else {
                eprintln!("ERROR: Missing raw response!");
            }
        } else {
            match request.kind {
                openai_client::RequestKind::Completion(ref request) => {
                    let _ = stdout.write_all(&request.prompt.as_bytes());
                    let _ = stdout.write_all(&response.content.as_bytes());
                }
                openai_client::RequestKind::Chat(..) => {
                    if output_reply_as_json {
                        let _ = stdout.write_all(&serde_json::to_string(&response).unwrap().as_bytes());
                    } else {
                        if !hide_thinking {
                            if let Some(ref reasoning_content) = response.reasoning_content {
                                let _ = writeln!(stdout, "<think>{}</think>\n", reasoning_content);
                            }
                        }

                        let _ = stdout.write_all(&response.content.as_bytes());
                    }
                }
            }
        }
        if is_terminal && !response.content.ends_with("\n") {
            let _ = stdout.write_all("\n".as_bytes());
        }
        let _ = stdout.flush();
    }
    Ok(())
}

fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if max == min {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let mut h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h /= 6.0;

    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let (r, g, b);

    if s == 0.0 {
        r = l;
        g = l;
        b = l;
    } else {
        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;
        r = hue_to_rgb(p, q, h + 1.0 / 3.0);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    }

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

