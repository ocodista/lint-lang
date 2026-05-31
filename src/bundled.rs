use std::path::PathBuf;

use anyhow::Result;

use crate::config::ConfiguredModel;

#[cfg(feature = "bundled-model")]
const BUNDLED_MODEL_BYTES: &[u8] = include_bytes!(env!(
    "LINT_LANG_BUNDLED_MODEL",
    "set LINT_LANG_BUNDLED_MODEL=/absolute/path/to/model.llamafile when building with --features bundled-model"
));

#[cfg(feature = "bundled-model")]
const BUNDLED_MODEL_SOURCE: &str = env!(
    "LINT_LANG_BUNDLED_MODEL",
    "set LINT_LANG_BUNDLED_MODEL=/absolute/path/to/model.llamafile when building with --features bundled-model"
);

#[cfg(feature = "bundled-model")]
pub fn configured_model(llama_cli: Option<PathBuf>) -> Result<Option<ConfiguredModel>> {
    use std::{fs, path::Path};

    use anyhow::{Context, anyhow};
    use directories::ProjectDirs;

    use crate::config::configured_model_from_path;

    let project_dirs = ProjectDirs::from("com", "caio", "lint-lang")
        .ok_or_else(|| anyhow!("could not find an app data directory for this platform"))?;
    let file_name = Path::new(BUNDLED_MODEL_SOURCE)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("bundled-model.llamafile");
    let model_path = project_dirs
        .data_dir()
        .join("bundled-model")
        .join(file_name);

    let should_write = model_path
        .metadata()
        .map(|metadata| metadata.len() != BUNDLED_MODEL_BYTES.len() as u64)
        .unwrap_or(true);

    if should_write {
        let parent = model_path
            .parent()
            .ok_or_else(|| anyhow!("bundled model path has no parent: {}", model_path.display()))?;
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create bundled model directory {}",
                parent.display()
            )
        })?;
        fs::write(&model_path, BUNDLED_MODEL_BYTES).with_context(|| {
            format!(
                "failed to extract bundled model to {}",
                model_path.display()
            )
        })?;

        #[cfg(unix)]
        make_executable_if_needed(&model_path)?;
    }

    configured_model_from_path(model_path, None, llama_cli).map(Some)
}

#[cfg(not(feature = "bundled-model"))]
pub fn configured_model(_llama_cli: Option<PathBuf>) -> Result<Option<ConfiguredModel>> {
    Ok(None)
}

#[cfg(all(feature = "bundled-model", unix))]
fn make_executable_if_needed(path: &std::path::Path) -> Result<()> {
    use std::{fs, os::unix::fs::PermissionsExt};

    use crate::config::extension_is;

    if extension_is(path, "gguf") {
        return Ok(());
    }

    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}
