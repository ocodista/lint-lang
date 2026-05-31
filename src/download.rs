use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use reqwest::{Client, header::USER_AGENT};
use tokio::{fs, io::AsyncWriteExt};

use crate::config::{
    ConfiguredModel, app_model_dir, configured_model_from_path, expand_home, extension_is,
};

pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf";
pub const DEFAULT_MODEL_FILENAME: &str = "Qwen3-8B-Q4_K_M.gguf";

#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub url: String,
    pub output: Option<PathBuf>,
    pub force: bool,
    pub llama_cli: Option<PathBuf>,
}

pub async fn download_model(options: DownloadOptions) -> Result<ConfiguredModel> {
    let target_path = target_path(options.output, &options.url)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(60 * 60))
        .build()
        .context("failed to build download HTTP client")?;

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create model directory {}", parent.display()))?;
    }

    let response = client
        .get(&options.url)
        .header(USER_AGENT, concat!("lint-lang/", env!("CARGO_PKG_VERSION")))
        .send()
        .await
        .with_context(|| format!("failed to start model download from {}", options.url))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("model download failed with {status}: {body}");
    }

    let expected_len = response.content_length();
    if !options.force
        && existing_file_is_complete(&target_path, expected_len)
            .await
            .unwrap_or(false)
    {
        make_executable_if_needed(&target_path).await?;
        eprintln!("model already downloaded: {}", target_path.display());
        return configured_model_from_path(target_path, None, options.llama_cli);
    }

    let part_path = part_path(&target_path)?;
    let mut file = fs::File::create(&part_path).await.with_context(|| {
        format!(
            "failed to create temporary download file {}",
            part_path.display()
        )
    })?;

    eprintln!("downloading model from {}", options.url);
    eprintln!("saving to {}", target_path.display());

    let mut downloaded = 0u64;
    let mut last_report = Instant::now();
    let mut response = response;

    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed while reading model download stream")?
    {
        file.write_all(&chunk).await.with_context(|| {
            format!("failed to write model download to {}", part_path.display())
        })?;
        downloaded += chunk.len() as u64;

        if last_report.elapsed() >= Duration::from_secs(1) {
            print_progress(downloaded, expected_len);
            last_report = Instant::now();
        }
    }

    file.flush()
        .await
        .with_context(|| format!("failed to flush model download to {}", part_path.display()))?;
    drop(file);

    if let Some(expected_len) = expected_len
        && downloaded != expected_len
    {
        bail!(
            "incomplete model download: got {}, expected {}",
            format_bytes(downloaded),
            format_bytes(expected_len)
        );
    }

    fs::rename(&part_path, &target_path)
        .await
        .with_context(|| {
            format!(
                "failed to move downloaded model from {} to {}",
                part_path.display(),
                target_path.display()
            )
        })?;

    make_executable_if_needed(&target_path).await?;
    print_progress(downloaded, expected_len);
    eprintln!("\ndownloaded model: {}", target_path.display());

    configured_model_from_path(target_path, None, options.llama_cli)
}

pub fn default_download_output() -> Result<PathBuf> {
    Ok(app_model_dir()?.join(DEFAULT_MODEL_FILENAME))
}

fn target_path(output: Option<PathBuf>, url: &str) -> Result<PathBuf> {
    let filename = file_name_from_url(url).unwrap_or(DEFAULT_MODEL_FILENAME);
    let Some(output) = output else {
        return default_download_output();
    };

    let output = expand_home(output);
    if output.exists() && output.is_dir() {
        return Ok(output.join(filename));
    }

    if looks_like_model_file(&output) {
        return Ok(output);
    }

    Ok(output.join(filename))
}

fn file_name_from_url(url: &str) -> Option<&str> {
    let without_query = url.split('?').next()?;
    without_query
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
}

fn looks_like_model_file(path: &Path) -> bool {
    extension_is(path, "llamafile") || extension_is(path, "gguf") || extension_is(path, "exe")
}

async fn existing_file_is_complete(path: &Path, expected_len: Option<u64>) -> Result<bool> {
    let metadata = match fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    };

    if !metadata.is_file() {
        return Ok(false);
    }

    match expected_len {
        Some(expected_len) => Ok(metadata.len() == expected_len),
        None => Ok(metadata.len() > 0),
    }
}

fn part_path(target_path: &Path) -> Result<PathBuf> {
    let file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow!(
                "download target has no file name: {}",
                target_path.display()
            )
        })?;
    Ok(target_path.with_file_name(format!("{file_name}.part")))
}

async fn make_executable_if_needed(path: &Path) -> Result<()> {
    if extension_is(path, "gguf") {
        return Ok(());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::metadata(path)
            .await
            .with_context(|| format!("failed to inspect downloaded model {}", path.display()))?;
        let mut permissions = metadata.permissions();
        permissions.set_mode(permissions.mode() | 0o755);
        fs::set_permissions(path, permissions)
            .await
            .with_context(|| {
                format!(
                    "failed to make downloaded model executable: {}",
                    path.display()
                )
            })?;
    }

    Ok(())
}

fn print_progress(downloaded: u64, expected_len: Option<u64>) {
    match expected_len {
        Some(total) if total > 0 => {
            let percent = downloaded as f64 / total as f64 * 100.0;
            eprint!(
                "\rdownloaded {} / {} ({percent:.1}%)",
                format_bytes(downloaded),
                format_bytes(total)
            );
        }
        _ => eprint!("\rdownloaded {}", format_bytes(downloaded)),
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0usize;

    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_MODEL_FILENAME, file_name_from_url, format_bytes, target_path};
    use std::path::PathBuf;

    #[test]
    fn extracts_file_name_from_url() {
        assert_eq!(
            file_name_from_url("https://example.com/models/qwen.llamafile?download=1"),
            Some("qwen.llamafile")
        );
    }

    #[test]
    fn treats_model_extension_as_output_file() {
        let path = target_path(
            Some(PathBuf::from("~/models/custom.llamafile")),
            "https://example.com/qwen.llamafile",
        )
        .unwrap();

        assert!(path.ends_with("custom.llamafile"));
    }

    #[test]
    fn treats_plain_output_as_directory() {
        let path = target_path(
            Some(PathBuf::from("models")),
            "https://example.com/qwen.llamafile",
        )
        .unwrap();

        assert!(path.ends_with("models/qwen.llamafile"));
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn default_filename_is_a_gguf() {
        assert!(DEFAULT_MODEL_FILENAME.ends_with(".gguf"));
    }
}
