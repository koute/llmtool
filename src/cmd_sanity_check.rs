use crate::openai_client;
use crate::utils::{extract_response, prepare_chat_request_template};
use crate::{ChatArgs, CommonArgs, SchemaArgs};
use futures::prelude::*;
use std::io::Write;

#[derive(clap::Args)]
pub struct SanityCheckArgs {
    #[clap(flatten)]
    common_args: CommonArgs,

    #[clap(flatten)]
    chat_args: ChatArgs,

    /// Tries the hard version of the task where the model has to decrypt the prompt itself.
    #[clap(long)]
    hard: bool,
}

const KEY_IN: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A',
    'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1',
    '2', '3', '4', '5', '6', '7', '8', '9',
];
const KEY_OUT: &[char] = &[
    'I', 'D', 'M', 'q', 'L', 's', 'Y', 'H', 't', '8', 'e', 'B', 'b', 'm', 'l', 'Z', 'X', 'N', 'w', 'f', 'y', 'J', 'n', '0', 'd', 'W', '9',
    'P', 'z', 'E', 'g', 'j', 'Q', 'T', 'x', '2', 'F', 'G', 'i', 'o', 'h', 'S', 'c', 'k', 'R', '5', '6', '7', '1', '3', 'A', 'p', 'V', 'K',
    'r', '4', 'O', 'U', 'C', 'v', 'a', 'u',
];

fn decrypt(string: &str) -> String {
    let map: std::collections::HashMap<char, char> = KEY_OUT.iter().copied().zip(KEY_IN.iter().copied()).collect();
    string.chars().map(|ch| map.get(&ch).copied().unwrap_or(ch)).collect()
}

fn hard_prompt() -> String {
    use std::fmt::Write;
    let mut buffer = String::new();
    buffer.push_str("Cipher: ");
    for (cin, cout) in KEY_OUT.iter().copied().zip(KEY_IN.iter().copied()) {
        write!(&mut buffer, "'{cin}' -> '{cout}';").unwrap();
    }
    write!(&mut buffer, "\n\nDecript and fulfill the following task:\n{PROMPT}").unwrap();
    buffer
}

// This comes from MMLU-Pro, but I changed the input numbers (which changed the results), changed the correct letter and "encrypted" (to minimize the risk of leakage).
const PROMPT: &'static str = r"
9mwnLN fHL slBBlntmY byBftZBL MHltML XyLwftlm. 5HL BIwf BtmL ls dlyN NLwZlmwL wHlyBq DL ls fHL slBBlntmY slNbIf: '9oR1gk: $Gg55gk' (ntfHlyf XylfLw) nHLNL Gg55gk tw lmL ls 9,P,z,E,g,j,Q,T,x,2. 5Htme wfLZ Dd wfLZ DLslNL ImwnLNtmY.
cyLwftlm: 9 K4V-JlBf wHymf blflN HIw Im INbIfyNL nHlwL NLwtwfImML tw V.OV lHb. 9wwybtmY I JlBfIYL IMNlww fHL DNywH MlmfIMfw ls 4 JlBfw, nHIf INbIfyNL MyNNLmf ntBB sBln (I) nHLm fHL MlymfLNLbstw KKV JlBfw ? (D) ts fHL blflN BlIq tw tmMNLIwLq wl fHIf fHL MlymfLNLbsqNlZw fl KVv JlBfw
hZftlmw: 9) OK.a IbZ Imq Ov.v IbZ P) Or.U IbZ Imq UV.V IbZ z) 4C.U IbZ Imq O4.4 IbZ E) 4U.O IbZ Imq Or.r IbZ g) 4u.O IbZ Imq OC.K IbZ j) 4O.4 IbZ Imq OK.4 IbZ Q) 4v.C IbZ Imq OO.O IbZ T) Or.K IbZ Imq Oa.u IbZ x) OV.v IbZ Imq Ov.U IbZ 2) 44.r IbZ Imq OV.V IbZ
";

const EXPECTED_ANSWER: &'static str = "9oR1gk: P";

pub async fn main_sanity_check(
    SanityCheckArgs {
        mut common_args,
        chat_args,
        hard,
    }: SanityCheckArgs,
) -> Result<(), String> {
    let prompt = if !hard { decrypt(&PROMPT.trim()) } else { hard_prompt() };

    let endpoint = common_args.common_setup().await?;
    let mut args = common_args.get_generation_args()?;
    args.request_stream_usage = true;

    let request = openai_client::Request {
        args,
        kind: {
            let mut req = prepare_chat_request_template(&chat_args, &SchemaArgs::default())?;
            req.messages.push(openai_client::Message::new("user".into(), prompt));
            openai_client::RequestKind::Chat(req)
        },
    };

    let mut output = String::new();
    let mut stream = request.send_streaming(&endpoint, false).await.map_err(|error| error.to_string())?;
    let mut is_thinking = false;
    let mut usage = None;
    let mut timestamp_first_token = None;
    while let Some(chunk) = stream.next().await {
        let response = extract_response(&chunk)?;

        if timestamp_first_token.is_none() {
            timestamp_first_token = Some(std::time::Instant::now());
        }

        if response.usage.is_some() {
            usage = response.usage.clone();
        }

        let out = std::io::stdout();
        let mut out = out.lock();
        if let Some(ref reasoning_content) = response.reasoning_content {
            if !is_thinking {
                let _ = out.write_all(b"<think>");
                is_thinking = true;
            }

            let _ = out.write_all(reasoning_content.as_bytes());
        }

        if !response.content.is_empty() {
            if is_thinking {
                let _ = out.write_all(b"</think>");
                is_thinking = false;
            }

            let _ = out.write_all(response.content.as_bytes());
            output.push_str(&response.content);
        }

        if out.flush().is_err() {
            return Ok(());
        }
    }

    const VT_RED: &str = "\x1b[1;31m";
    const VT_GREEN: &str = "\x1b[1;32m";
    const VT_RESET: &str = "\x1b[0m";

    let output = output.trim();
    let output = output.lines().last().unwrap_or("");
    let expected = decrypt(EXPECTED_ANSWER);
    println!("\n\n");

    if let (Some(usage), Some(timestamp)) = (usage, timestamp_first_token) {
        let tokens = usage.completion_tokens;
        let time = timestamp.elapsed().as_secs_f64();
        let speed = tokens as f64 / time;
        println!("Processing time: {time:.1}s ({speed:.1}) tg/s\n");
    }

    if output == expected || output == format!("**{expected}**") {
        println!("{VT_GREEN}SANITY CHECK OK!{VT_RESET}");
        Ok(())
    } else {
        println!("{VT_RED}SANITY CHECK FAIL! EXPECTED: \"{expected}\"{VT_RESET}");
        Err(format!("sanity check failed"))
    }
}
