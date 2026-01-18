use clap::Parser;

use openai_client::Endpoint;
use std::path::PathBuf;

mod cache;
mod cmd_batch_query;
mod cmd_cache_server;
mod cmd_single_request;
mod openai_client;
mod utils;

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

const DEFAULT_LOCAL_PORT: u32 = 9001;

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

pub(crate) fn small_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_time()
        .enable_io()
        .build()
        .unwrap()
}

fn big_runtime() -> tokio::runtime::Runtime {
    let thread_count = crate::utils::get_thread_count().unwrap();
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
        } => small_runtime().block_on(crate::cmd_single_request::main_single_request(
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
        } => small_runtime().block_on(crate::cmd_single_request::main_single_request(
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
        } => big_runtime().block_on(crate::cmd_batch_query::main_batch_query(
            common_args,
            chat_args,
            input,
            output,
            save_raw,
            jobs,
            quiet,
        )),
        Args::ListModels { url } => small_runtime().block_on(main_list_models(url)),
        Args::CacheServer { host, port, cache_path } => {
            big_runtime().block_on(crate::cmd_cache_server::main_cache_server(&format!("{host}:{port}"), cache_path))
        }
    };

    if let Err(error) = error {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}
