use std::{fmt, fs, path::Path};

use anyhow::{Result, bail};
use arboard::Clipboard;
use llama_cpp_2::{LogOptions, llama_backend::LlamaBackend, send_logs_to_tracing};

use crate::{
    config::{
        AppConfig, ConfiguredModel, DEFAULT_ENDPOINT, ModelBackend, app_model_dir, config_path,
    },
    download, ollama,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DoctorStatus {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DoctorCheck {
    status: DoctorStatus,
    name: String,
    detail: String,
    fix: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct DoctorReport {
    checks: Vec<DoctorCheck>,
}

pub async fn run() -> Result<()> {
    let report = build_report().await;
    println!("{report}");

    if report.has_failures() {
        bail!("doctor found issues");
    }

    Ok(())
}

async fn build_report() -> DoctorReport {
    let mut report = DoctorReport::default();
    let config = inspect_config(&mut report);
    let endpoint = config
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned());

    inspect_locale(&mut report, &config);
    inspect_model(&mut report, config.configured_model().as_ref(), &endpoint).await;
    inspect_default_model_path(&mut report);
    inspect_native_backend(&mut report);
    inspect_clipboard(&mut report);

    report
}

fn inspect_config(report: &mut DoctorReport) -> AppConfig {
    match config_path() {
        Ok(path) => {
            report.ok("Config path", path.display().to_string());
            if path.exists() {
                report.ok("Config file", "found".to_owned());
            } else {
                report.warn(
                    "Config file",
                    "not created yet".to_owned(),
                    Some("Run `lint-lang --configure` or `lint-lang --download-model`."),
                );
            }
        }
        Err(error) => report.fail(
            "Config path",
            error.to_string(),
            Some("Check your platform config directory permissions."),
        ),
    }

    match AppConfig::load() {
        Ok(config) => config,
        Err(error) => {
            report.fail(
                "Config file",
                error.to_string(),
                Some("Fix or remove the config file, then run `lint-lang --configure`."),
            );
            AppConfig::default()
        }
    }
}

fn inspect_locale(report: &mut DoctorReport, config: &AppConfig) {
    let locale = config.locale.unwrap_or_default();
    report.ok("Locale", locale.label().to_owned());
}

async fn inspect_model(
    report: &mut DoctorReport,
    selected_model: Option<&ConfiguredModel>,
    endpoint: &str,
) {
    let Some(model) = selected_model else {
        report.fail(
            "Selected model",
            "none configured".to_owned(),
            Some("Run `lint-lang --configure` or `lint-lang --download-model`."),
        );
        return;
    };

    report.ok("Selected model", model.name.clone());
    report.ok("Backend", model.backend.kind_label().to_owned());

    match &model.backend {
        ModelBackend::Llamafile { path }
        | ModelBackend::LlamaCpp {
            model_path: path, ..
        } => {
            inspect_local_model_file(report, path);
        }
        ModelBackend::Ollama { model, .. } => inspect_ollama_model(report, endpoint, model).await,
    }
}

fn inspect_local_model_file(report: &mut DoctorReport, path: &Path) {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {
            report.ok(
                "Model file",
                format!("{} ({})", path.display(), format_bytes(metadata.len())),
            );
        }
        Ok(metadata) if metadata.is_file() => report.fail(
            "Model file",
            format!("{} is empty", path.display()),
            Some("Re-download the model with `lint-lang --download-model --force-download`."),
        ),
        Ok(_) => report.fail(
            "Model file",
            format!("{} is not a file", path.display()),
            Some("Configure a GGUF file with `lint-lang --configure --model-path <path>`."),
        ),
        Err(error) => report.fail(
            "Model file",
            format!("{} ({error})", path.display()),
            Some("Run `lint-lang --download-model` or configure an existing GGUF file."),
        ),
    }
}

async fn inspect_ollama_model(report: &mut DoctorReport, endpoint: &str, model_name: &str) {
    match ollama::list_models(endpoint).await {
        Ok(models) if models.iter().any(|model| model == model_name) => {
            report.ok("Ollama model", format!("{model_name} at {endpoint}"));
        }
        Ok(_) => report.fail(
            "Ollama model",
            format!("{model_name} is not installed at {endpoint}"),
            Some("Install it with `ollama pull <model>` or configure a local GGUF model."),
        ),
        Err(error) => report.warn(
            "Ollama endpoint",
            format!("{endpoint} ({error:#})"),
            Some("Ignore this if you use native GGUF. Otherwise start Ollama."),
        ),
    }
}

fn inspect_default_model_path(report: &mut DoctorReport) {
    match app_model_dir() {
        Ok(path) => report.ok("Default model directory", path.display().to_string()),
        Err(error) => report.warn(
            "Default model directory",
            error.to_string(),
            Some("Check your platform data directory permissions."),
        ),
    }

    match download::default_download_output() {
        Ok(path) if path.exists() => report.ok("Default Qwen3 model", path.display().to_string()),
        Ok(path) => report.warn(
            "Default Qwen3 model",
            format!("{} is not downloaded", path.display()),
            Some("Run `lint-lang --download-model`."),
        ),
        Err(error) => report.warn("Default Qwen3 model", error.to_string(), None),
    }
}

fn inspect_native_backend(report: &mut DoctorReport) {
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    match LlamaBackend::init() {
        Ok(backend) if backend.supports_gpu_offload() => {
            report.ok("Native llama.cpp", "GPU offload available".to_owned());
        }
        Ok(_) => report.warn(
            "Native llama.cpp",
            "GPU offload unavailable; CPU inference will work but may be slow".to_owned(),
            None,
        ),
        Err(error) => report.warn(
            "Native llama.cpp",
            error.to_string(),
            Some("Retry in a fresh process if another llama.cpp session is active."),
        ),
    }
}

fn inspect_clipboard(report: &mut DoctorReport) {
    match Clipboard::new() {
        Ok(_) => report.ok("Clipboard", "available".to_owned()),
        Err(error) => report.warn(
            "Clipboard",
            error.to_string(),
            Some("Use `--no-clipboard` to skip clipboard writes."),
        ),
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

impl DoctorReport {
    fn has_failures(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == DoctorStatus::Fail)
    }

    fn ok(&mut self, name: &str, detail: String) {
        self.checks.push(DoctorCheck {
            status: DoctorStatus::Ok,
            name: name.to_owned(),
            detail,
            fix: None,
        });
    }

    fn warn(&mut self, name: &str, detail: String, fix: Option<&str>) {
        self.checks.push(DoctorCheck {
            status: DoctorStatus::Warn,
            name: name.to_owned(),
            detail,
            fix: fix.map(str::to_owned),
        });
    }

    fn fail(&mut self, name: &str, detail: String, fix: Option<&str>) {
        self.checks.push(DoctorCheck {
            status: DoctorStatus::Fail,
            name: name.to_owned(),
            detail,
            fix: fix.map(str::to_owned),
        });
    }
}

impl fmt::Display for DoctorReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "lint-lang doctor")?;
        writeln!(formatter)?;

        for check in &self.checks {
            writeln!(
                formatter,
                "{} {}: {}",
                check.status.icon(),
                check.name,
                check.detail
            )?;
            if let Some(fix) = &check.fix {
                writeln!(formatter, "  → {fix}")?;
            }
        }

        Ok(())
    }
}

impl DoctorStatus {
    fn icon(self) -> &'static str {
        match self {
            Self::Ok => "✓",
            Self::Warn => "!",
            Self::Fail => "✗",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{DoctorReport, DoctorStatus, format_bytes, inspect_local_model_file};

    #[test]
    fn formats_report_with_fix_steps() {
        let mut report = DoctorReport::default();
        report.fail(
            "Selected model",
            "none configured".to_owned(),
            Some("Run `lint-lang --configure`."),
        );

        let rendered = report.to_string();

        assert!(rendered.contains("✗ Selected model: none configured"));
        assert!(rendered.contains("→ Run `lint-lang --configure`."));
        assert!(report.has_failures());
    }

    #[test]
    fn reports_existing_local_model_file() {
        let path = temporary_model_path("existing-model.gguf");
        fs::write(&path, b"GGUF").expect("write fake model");
        let mut report = DoctorReport::default();

        inspect_local_model_file(&mut report, &path);

        assert_eq!(report.checks[0].status, DoctorStatus::Ok);
        assert!(report.checks[0].detail.contains("existing-model.gguf"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn reports_missing_local_model_file() {
        let path = temporary_model_path("missing-model.gguf");
        let mut report = DoctorReport::default();

        inspect_local_model_file(&mut report, &path);

        assert_eq!(report.checks[0].status, DoctorStatus::Fail);
        assert!(report.checks[0].detail.contains("missing-model.gguf"));
    }

    #[test]
    fn formats_bytes_for_model_sizes() {
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.0 MB");
    }

    fn temporary_model_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lint-lang-doctor-{}-{name}", std::process::id()))
    }
}
