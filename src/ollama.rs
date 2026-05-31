use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::prompt::{GrammarLocale, ollama_user_prompt, system_prompt};

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    stream: bool,
    options: ChatOptions,
}

#[derive(Debug, Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    temperature: f32,
    top_p: f32,
    num_predict: u16,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: String,
}

pub async fn list_models(endpoint: &str) -> Result<Vec<String>> {
    let client = client(Duration::from_secs(5))?;
    let response = client
        .get(format!("{}/api/tags", endpoint.trim_end_matches('/')))
        .send()
        .await
        .with_context(|| format!("failed to connect to Ollama at {endpoint}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("Ollama model list request failed with {status}: {body}");
    }

    let tags: TagsResponse = response
        .json()
        .await
        .context("failed to decode Ollama model list response")?;

    let mut models: Vec<String> = tags.models.into_iter().map(|model| model.name).collect();
    models.sort();
    models.dedup();
    Ok(models)
}

pub async fn fix_grammar(
    endpoint: &str,
    model: &str,
    input: &str,
    locale: GrammarLocale,
) -> Result<String> {
    let client = client(Duration::from_secs(120))?;
    let user_prompt = ollama_user_prompt(input, locale);
    let request = ChatRequest {
        model,
        messages: vec![
            Message {
                role: "system",
                content: system_prompt(locale),
            },
            Message {
                role: "user",
                content: &user_prompt,
            },
        ],
        stream: false,
        options: ChatOptions {
            temperature: 0.0,
            top_p: 0.1,
            num_predict: 512,
        },
    };

    let response = client
        .post(format!("{}/api/chat", endpoint.trim_end_matches('/')))
        .json(&request)
        .send()
        .await
        .with_context(|| format!("failed to connect to Ollama at {endpoint}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("Ollama grammar request failed with {status}: {body}");
    }

    let response: ChatResponse = response
        .json()
        .await
        .context("failed to decode Ollama grammar response")?;

    let fixed = clean_model_output(&response.message.content);
    if fixed.is_empty() {
        bail!("Ollama returned an empty correction");
    }

    Ok(fixed)
}

fn client(timeout: Duration) -> Result<Client> {
    Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build HTTP client")
}

pub(crate) fn clean_model_output(output: &str) -> String {
    let mut cleaned = output.trim().to_owned();

    if let Some(stripped) = strip_code_fence(&cleaned) {
        cleaned = stripped.trim().to_owned();
    }

    if let Some((_, after_think)) = cleaned.rsplit_once("</think>") {
        cleaned = after_think.trim().to_owned();
    }

    for token in ["<think>", "</think>"] {
        cleaned = cleaned.replace(token, "").trim().to_owned();
    }

    for prefix in ["Corrected:", "Correction:", "Fixed:", "Output:"] {
        if let Some(rest) = cleaned.strip_prefix(prefix) {
            cleaned = rest.trim_start().to_owned();
            break;
        }
    }

    cleaned
}

fn strip_code_fence(output: &str) -> Option<&str> {
    let output = output.trim();
    if !output.starts_with("```") || !output.ends_with("```") {
        return None;
    }

    let without_start = output.trim_start_matches('`');
    let without_end = without_start.trim_end_matches('`').trim();
    if let Some((_, content)) = without_end.split_once('\n') {
        Some(content)
    } else {
        Some(without_end)
    }
}

#[cfg(test)]
mod tests {
    use super::clean_model_output;

    #[test]
    fn removes_common_labels() {
        assert_eq!(
            clean_model_output("Fixed: I have an apple."),
            "I have an apple."
        );
    }

    #[test]
    fn removes_code_fences() {
        assert_eq!(
            clean_model_output("```text\nI have an apple.\n```"),
            "I have an apple."
        );
    }

    #[test]
    fn keeps_quotes_because_they_may_belong_to_the_input() {
        assert_eq!(clean_model_output("\"Hello, world.\""), "\"Hello, world.\"");
    }
}
