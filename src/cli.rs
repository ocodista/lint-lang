use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::prompt::GrammarLocale;

#[derive(Debug, Parser)]
#[command(
    name = "lint-lang",
    version,
    about = "Lint grammar locally for free with Rust and native llama.cpp.",
    after_help = "Command:\n  doctor    Check config, models, clipboard, and native llama.cpp support"
)]
pub struct Cli {
    /// Text to grammar-fix. If omitted, stdin is used when piped.
    #[arg(value_name = "TEXT", trailing_var_arg = true)]
    pub text: Vec<String>,

    /// Override with an Ollama model name for this run.
    #[arg(short, long, conflicts_with = "model_path")]
    pub model: Option<String>,

    /// Override with a local model path (.gguf uses native llama.cpp unless --llama-cli is set).
    #[arg(long, value_name = "PATH", conflicts_with = "model")]
    pub model_path: Option<PathBuf>,

    /// Backend for --model-path. Auto infers from extension.
    #[arg(long, value_enum, requires = "model_path")]
    pub backend: Option<LocalBackend>,

    /// Path to llama.cpp's llama-cli binary for .gguf models.
    #[arg(long, value_name = "PATH")]
    pub llama_cli: Option<PathBuf>,

    /// Directory to scan for .llamafile and .gguf models. Can be repeated.
    #[arg(long, value_name = "DIR")]
    pub model_dir: Vec<PathBuf>,

    /// Download the default Qwen3 8B GGUF model, save it, and use it.
    #[arg(long)]
    pub download_model: bool,

    /// Download a custom model URL instead of the default Qwen3 8B GGUF.
    #[arg(long, value_name = "URL")]
    pub download_url: Option<String>,

    /// Download output file or directory. Defaults to lint-lang's app model directory.
    #[arg(long, value_name = "PATH")]
    pub download_output: Option<PathBuf>,

    /// Re-download even if the target model file already exists.
    #[arg(long)]
    pub force_download: bool,

    /// Print the selected local model path and exit when no text is provided.
    #[arg(long)]
    pub print_model_path: bool,

    /// Ollama HTTP endpoint, only used for Ollama fallback models.
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Prompt locale for grammar correction.
    #[arg(long, value_enum)]
    pub locale: Option<GrammarLocale>,

    /// Shortcut for --locale pt-br.
    #[arg(long = "pt-br", alias = "ptbr", conflicts_with = "locale")]
    pub pt_br: bool,

    /// Open the TUI model selector and save the selection.
    #[arg(long)]
    pub configure: bool,

    /// Print the config file path and exit, unless text/configuration was also requested.
    #[arg(long)]
    pub config_path: bool,

    /// Do not write the fixed string to the system clipboard.
    #[arg(long)]
    pub no_clipboard: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LocalBackend {
    Auto,
    Llamafile,
    LlamaCpp,
}

impl LocalBackend {
    pub fn config_key(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Llamafile => Some("llamafile"),
            Self::LlamaCpp => Some("llama-cpp"),
        }
    }
}
