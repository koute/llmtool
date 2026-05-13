use clap::Parser;

use openai_client::Endpoint;
use std::path::PathBuf;

mod cache;
mod cache_client;
mod cmd_batch_query;
mod cmd_cache_server;
mod cmd_router;
mod cmd_sanity_check;
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

    /// The maximum number of tokens to generate.
    #[clap(short = 'l', long)]
    pub max_tokens: Option<u32>,

    #[clap(short = 't', long)]
    pub temperature: Option<f32>,

    /// Makes the model consider only the top K tokens.
    #[clap(long)]
    pub top_k: Option<u32>,

    /// Makes the model consider only the tokens within the given probability mass. For example, '0.1' means that only the tokens comprising the top 10% probability mass are considered.
    #[clap(long)]
    pub top_p: Option<f32>,

    // https://arxiv.org/abs/2407.01082
    /// Makes the model consider only the tokens which are at least as probable as the top token multiplied by this value. For example, '0.1' means that only the tokens at least 1/10th as probable as the top token will be considered.
    #[clap(long)]
    pub min_p: Option<f32>,

    /// Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[clap(long)]
    pub frequency_penalty: Option<f32>,

    /// Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[clap(long)]
    pub presence_penalty: Option<f32>,

    /// Penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
    #[clap(long)]
    pub repetition_penalty: Option<f32>,

    #[clap(long)]
    pub repetition_penalty_range: Option<u32>,

    #[clap(long)]
    pub request_prompt_caching: bool,

    #[clap(short = 'r', long)]
    pub reproducible: bool,

    /// The URL of the target endpoint.
    #[clap(long)]
    pub url: Option<String>,

    /// The API key to use.
    #[clap(long)]
    pub api_key: Option<String>,

    /// A comma-separated list of OpenRouter providers to use.
    #[clap(long)]
    pub provider: Option<String>,

    /// The priority of the request. Higher values mean lower priority.
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

#[derive(Default, clap::Args)]
pub struct ChatArgs {
    #[clap(long, default_value = "auto")]
    thinking: openai_client::Thinking,

    #[clap(long)]
    reasoning_effort: Option<String>,

    #[clap(long, short = 's')]
    system_prompt: Option<String>,
}

#[derive(Default, clap::Args)]
pub struct SchemaArgs {
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

#[derive(Copy, Clone, clap::ValueEnum)]
pub enum OnOff {
    On,
    Off,
}

impl From<OnOff> for bool {
    fn from(on_off: OnOff) -> bool {
        match on_off {
            OnOff::On => true,
            OnOff::Off => false,
        }
    }
}

#[derive(Copy, Clone, Default, clap::ValueEnum)]
enum DisplayThinking {
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

    #[clap(long)]
    print_raw_request: bool,

    #[clap(long)]
    print_raw_response: bool,

    #[clap(long)]
    disable_cache: bool,
}

#[derive(clap::Parser)]
enum Args {
    /// Sends a single chat completion query.
    Q {
        /// Whether to display thinking. (NOTE: This doesn't affect whether the model's thinking is enabled.)
        #[clap(long, default_value = "auto")]
        display_thinking: DisplayThinking,

        #[clap(flatten)]
        common_args: CommonArgs,

        #[clap(flatten)]
        chat_args: ChatArgs,

        #[clap(flatten)]
        schema_args: SchemaArgs,

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

        #[clap(flatten)]
        schema_args: SchemaArgs,

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

        #[clap(long)]
        total_request_limit: Option<i64>,
    },
    /// Sanity-checks the model.
    SanityCheck(crate::cmd_sanity_check::SanityCheckArgs),
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
    /// Starts an OpenAI-compatible HTTP proxy/router.
    Router(crate::cmd_router::RouterArgs),
}

const DEFAULT_LOCAL_PORT: u32 = 9001;

enum RequestKind {
    Completion,
    Chat(ChatArgs, SchemaArgs),
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
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
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
                require_parameters: false,
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
            DisplayThinking::Auto,
            single_request_args,
        )),
        Args::Q {
            display_thinking,
            common_args,
            chat_args,
            schema_args,
            query,
            single_request_args,
        } => small_runtime().block_on(crate::cmd_single_request::main_single_request(
            common_args,
            query,
            RequestKind::Chat(chat_args, schema_args),
            display_thinking,
            single_request_args,
        )),
        Args::BatchQuery {
            common_args,
            chat_args,
            schema_args,
            input,
            output,
            save_raw,
            jobs,
            quiet,
            total_request_limit,
        } => big_runtime().block_on(crate::cmd_batch_query::main_batch_query(
            common_args,
            chat_args,
            schema_args,
            input,
            output,
            save_raw,
            jobs,
            quiet,
            total_request_limit,
        )),
        Args::ListModels { url } => small_runtime().block_on(main_list_models(url)),
        Args::CacheServer { host, port, cache_path } => {
            big_runtime().block_on(crate::cmd_cache_server::main_cache_server(&format!("{host}:{port}"), cache_path))
        }
        Args::Router(args) => big_runtime().block_on(crate::cmd_router::main_proxy_server(args)),
        Args::SanityCheck(args) => small_runtime().block_on(crate::cmd_sanity_check::main_sanity_check(args)),
    };

    if let Err(error) = error {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}
