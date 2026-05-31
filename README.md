# lint-lang

Lint your grammar locally for free. Built with Rust and native llama.cpp.

`lint-lang` fixes grammar, spelling, accents, punctuation, and capitalization. It keeps your text local by running a GGUF model on your machine. The default setup uses Qwen3 8B with Metal acceleration on Apple Silicon.

## Features

- Local grammar fixes without an API bill.
- Rust CLI installed as `lint-lang`.
- Native GGUF inference through llama.cpp.
- Default Qwen3 8B model download.
- pt-BR and English prompts.
- Clipboard copy by default.
- Single-line loading spinner.

## Requirements

- Rust 1.91+
- macOS with Xcode Command Line Tools for the best supported path
- About 5 GB of disk space for the default Qwen3 8B GGUF model

Install Rust if needed:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Install from source

```sh
git clone git@github.com:ocodista/lint-lang.git
cd lint-lang
cargo install --path . --force
```

The command is now available globally:

```sh
lint-lang --help
```

## First run

Download the default local model and fix text:

```sh
lint-lang --download-model --pt-br "As verspera da prova, resolvi ir ao cinema. À propósito: penso em chegar tard"
```

After setup, run it directly:

```sh
lint-lang "As verspera da prova, resolvi ir ao cinema. À propósito: penso em chegar tard"
```

Expected output:

```txt
Às vésperas da prova, resolvi ir ao cinema. A propósito: penso em chegar tarde.
```

## Configure models

Open the model selector:

```sh
lint-lang --configure
```

Use a local GGUF file directly:

```sh
lint-lang --configure --model-path ~/models/Qwen3-8B-Q4_K_M.gguf
```

Print the active model path:

```sh
lint-lang --print-model-path
```

## Locales

Portuguese Brazil:

```sh
lint-lang --pt-br "eu vai no mercado"
```

English:

```sh
lint-lang --locale en "i has a apple"
```

## Clipboard

`lint-lang` copies the corrected text to your clipboard by default.

Disable clipboard writes:

```sh
lint-lang --no-clipboard "i has a apple"
```

## Development

```sh
make check
cargo test
cargo clippy -- -D warnings
```

Useful commands:

```sh
make run "As verspera da prova"
make reset-config
```

## Model storage

The default model downloads to:

```txt
~/Library/Application Support/com.caio.lint-lang/models/Qwen3-8B-Q4_K_M.gguf
```

Model files are ignored by git.
