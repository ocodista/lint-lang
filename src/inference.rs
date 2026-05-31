use std::{num::NonZeroU32, path::Path, process::Command};

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::{
    context::params::LlamaContextParams, llama_backend::LlamaBackend, llama_batch::LlamaBatch,
    model::AddBos, model::LlamaModel, model::params::LlamaModelParams, sampling::LlamaSampler,
};

use crate::{
    config::{ConfiguredModel, DEFAULT_ENDPOINT, DEFAULT_LLAMA_CLI, ModelBackend},
    ollama,
    prompt::{GrammarLocale, postprocess_correction, qwen_instruct_prompt},
};

const MAX_PREDICTIONS: usize = 160;
const CONTEXT_TOKENS: u32 = 2048;

pub async fn fix_grammar(
    model: &ConfiguredModel,
    input: &str,
    locale: GrammarLocale,
) -> Result<String> {
    let fixed = match &model.backend {
        ModelBackend::Ollama { model, endpoint } => {
            let endpoint = endpoint.as_deref().unwrap_or(DEFAULT_ENDPOINT);
            ollama::fix_grammar(endpoint, model, input, locale).await
        }
        ModelBackend::LlamaCpp {
            model_path,
            llama_cli: None,
        } => {
            let model_path = model_path.clone();
            let input = input.to_owned();
            tokio::task::spawn_blocking(move || run_native_llama_cpp(&model_path, &input, locale))
                .await
                .context("native llama.cpp inference task failed")?
        }
        ModelBackend::Llamafile { .. }
        | ModelBackend::LlamaCpp {
            llama_cli: Some(_), ..
        } => {
            let model = model.clone();
            let input = input.to_owned();
            tokio::task::spawn_blocking(move || run_local_command(&model, &input, locale))
                .await
                .context("local inference task failed")?
        }
    }?;

    Ok(postprocess_correction(&fixed, locale))
}

fn run_native_llama_cpp(model_path: &Path, input: &str, locale: GrammarLocale) -> Result<String> {
    let prompt = qwen_instruct_prompt(input, locale);

    let mut backend =
        LlamaBackend::init().context("failed to initialize native llama.cpp backend")?;
    backend.void_logs();

    let mut model_params = LlamaModelParams::default().with_use_mmap(true);
    if backend.supports_gpu_offload() {
        model_params = model_params.with_n_gpu_layers(999);
    }

    let model =
        LlamaModel::load_from_file(&backend, model_path, &model_params).map_err(|error| {
            anyhow!(
                "failed to load GGUF model with native llama.cpp from {}: {error}",
                model_path.display()
            )
        })?;

    let threads = std::thread::available_parallelism()
        .map(|threads| i32::try_from(threads.get()).unwrap_or(4))
        .unwrap_or(4);
    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS))
        .with_n_threads(threads)
        .with_n_threads_batch(threads);
    let mut context = model
        .new_context(&backend, context_params)
        .map_err(|error| anyhow!("failed to create native llama.cpp context: {error}"))?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| anyhow!("failed to tokenize native llama.cpp prompt: {error}"))?;

    let required_context = tokens.len().saturating_add(MAX_PREDICTIONS);
    if required_context > CONTEXT_TOKENS as usize {
        bail!(
            "prompt is too long for native llama.cpp context: needs {required_context} tokens, limit is {CONTEXT_TOKENS}"
        );
    }

    let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
    let last_index = i32::try_from(tokens.len().saturating_sub(1))
        .context("prompt token count does not fit in i32")?;
    for (index, token) in (0_i32..).zip(tokens.into_iter()) {
        batch.add(token, index, &[0], index == last_index)?;
    }

    context
        .decode(&mut batch)
        .map_err(|error| anyhow!("failed to evaluate native llama.cpp prompt: {error}"))?;

    let mut output = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut position = batch.n_tokens();

    for _ in 0..MAX_PREDICTIONS {
        let token = sampler.sample(&context, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|error| anyhow!("failed to decode native llama.cpp token: {error}"))?;
        output.push_str(&piece);

        if output.contains("<|im_end|>")
            || output.contains("<|endoftext|>")
            || (output.contains("</think>") && output.lines().count() > 5)
        {
            break;
        }

        batch.clear();
        batch.add(token, position, &[0], true)?;
        position += 1;

        context
            .decode(&mut batch)
            .map_err(|error| anyhow!("failed to evaluate native llama.cpp token: {error}"))?;
    }

    let fixed = clean_local_output(&output);
    if fixed.is_empty() {
        bail!("native llama.cpp returned an empty correction");
    }

    Ok(fixed)
}

fn run_local_command(
    model: &ConfiguredModel,
    input: &str,
    locale: GrammarLocale,
) -> Result<String> {
    let prompt = qwen_instruct_prompt(input, locale);
    let mut command = local_command(model)?;
    add_generation_args(&mut command, &prompt);

    let output = command
        .output()
        .with_context(|| command_start_error(model))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        bail!(
            "local model command failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout.trim(),
            stderr.trim()
        );
    }

    let fixed = clean_local_output(&stdout);
    if fixed.is_empty() {
        bail!(
            "local model command returned an empty correction\nstderr:\n{}",
            stderr.trim()
        );
    }

    Ok(fixed)
}

fn local_command(model: &ConfiguredModel) -> Result<Command> {
    match &model.backend {
        ModelBackend::Llamafile { path } => {
            let mut command = Command::new(path);
            command.arg("--cli").arg("--log-disable");
            Ok(command)
        }
        ModelBackend::LlamaCpp {
            model_path,
            llama_cli: Some(llama_cli),
        } => {
            let mut command = Command::new(llama_cli);
            command.arg("-m").arg(model_path);
            Ok(command)
        }
        ModelBackend::LlamaCpp {
            model_path,
            llama_cli: None,
        } => {
            let mut command = Command::new(DEFAULT_LLAMA_CLI);
            command.arg("-m").arg(model_path);
            Ok(command)
        }
        ModelBackend::Ollama { .. } => bail!("Ollama models do not use a local command"),
    }
}

fn add_generation_args(command: &mut Command, prompt: &str) {
    command
        .arg("-p")
        .arg(prompt)
        .arg("-n")
        .arg(MAX_PREDICTIONS.to_string())
        .arg("--temp")
        .arg("0")
        .arg("--top-p")
        .arg("0.1")
        .arg("--no-display-prompt");
}

fn command_start_error(model: &ConfiguredModel) -> String {
    match &model.backend {
        ModelBackend::Llamafile { path } => format!(
            "failed to run llamafile {}. If needed, make it executable with `chmod +x {}`",
            path.display(),
            path.display()
        ),
        ModelBackend::LlamaCpp { llama_cli, .. } => {
            let llama_cli = llama_cli
                .as_ref()
                .map_or(DEFAULT_LLAMA_CLI.to_owned(), |path| {
                    path.display().to_string()
                });
            format!(
                "failed to run {llama_cli}. Install llama.cpp or configure the binary with --llama-cli"
            )
        }
        ModelBackend::Ollama { .. } => "failed to start inference".to_owned(),
    }
}

fn clean_local_output(output: &str) -> String {
    let mut cleaned = output.replace("\r\n", "\n");

    if let Some((_, assistant_output)) = cleaned.rsplit_once("<|im_start|>assistant") {
        cleaned = assistant_output.to_owned();
    }

    if let Some((before_end, _)) = cleaned.split_once("<|im_end|>") {
        cleaned = before_end.to_owned();
    }

    if let Some((_, after_think)) = cleaned.rsplit_once("</think>") {
        cleaned = after_think.to_owned();
    }

    for token in ["<|endoftext|>", "<|im_end|>", "<think>", "</think>"] {
        cleaned = cleaned.replace(token, "");
    }

    let cleaned = cleaned
        .trim_start_matches(|character: char| character.is_whitespace() || character == '\n')
        .trim();

    ollama::clean_model_output(cleaned)
}

#[cfg(test)]
mod tests {
    use super::clean_local_output;

    #[test]
    fn removes_qwen_chat_tokens() {
        let output = "<|im_start|>assistant\nI have an apple.<|im_end|>";

        assert_eq!(clean_local_output(output), "I have an apple.");
    }

    #[test]
    fn removes_prompt_echo_when_present() {
        let output = "prompt text <|im_start|>assistant\nFixed: I have an apple.<|im_end|>";

        assert_eq!(clean_local_output(output), "I have an apple.");
    }
}
