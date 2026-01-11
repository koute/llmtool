## llmtool

This is my personal command-line toolkit for interacting with LLMs.

### Features

  - Can send single completion and chat requests.
  - Can run batch queries, with fast suspend/resume support.
  - Supports request/response caching and logging with a custom cache server.
  - Made for interacting with local models *and* with models on Open Router.

### Usage examples

```
$ lt q What is 2+2?
<think>The user asks a simple question: "What is 2+2?" The answer is 4. Provide answer.</think>

2 + 2 = 4.
```

```
$ cat Cargo.toml | lt q -r --thinking=hide "Extract the list of dependencies from the following file as JSON, nicely formatted, with one line per dependency:\n\n" --output-format=json-array-of-objects
[
  {"name":"serde","version":"1","features":["derive"]},
  {"name":"serde_json","version":"1","features":["preserve_order","float_roundtrip"]},
  {"name":"tokio","version":"1","features":["full"]},
  {"name":"reqwest","version":"0.11","features":["json"]},
  {"name":"futures","version":"0.3"},
  {"name":"ctrlc","version":"3"},
  {"name":"clap","version":"4.5.34","features":["derive"]},
  {"name":"memmap2","version":"0.9.5"},
  {"name":"memchr","version":"2.7.5"},
  {"name":"rayon","version":"1.11.0"},
  {"name":"tikv-jemallocator","version":"0.6.1"},
  {"name":"blake3","version":"1.8.2"},
  {"name":"ahash","version":"0.8.12"},
  {"name":"time","version":"0.3","features":["formatting","local-offset"]}
]
```

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
