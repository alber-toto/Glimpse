pub mod catalog;
pub mod engine;
pub mod install;
pub mod menu;
pub mod remote;

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use reqwest::Client;
use tauri::{AppHandle, Manager};

use crate::settings::UserSettings;
use crate::transcription_api::TranscriptionSuccess;
use crate::{AppRuntime, AppState};

pub use catalog::{SpeechModel, list_models};

pub const WHISPER_CHUNK_SECONDS: u32 = 28;
pub const WHISPER_CHUNK_OVERLAP_SECONDS: u32 = 2;
pub const WHISPER_LEADING_PAD_SECONDS: f32 = 0.2;
pub const PARAKEET_CHUNK_SECONDS: u32 = 180;
pub const PARAKEET_CHUNK_OVERLAP_SECONDS: u32 = 3;
pub const VAD_MIN_SPEECH_PERCENT_FILE: f32 = 2.0;
pub const VAD_MIN_SPEECH_PERCENT_CHUNK: f32 = 5.0;

pub fn selected_model(settings: &UserSettings) -> String {
    if remote::is_configured(settings) {
        remote::speech_model_storage_label(settings, None)
    } else {
        settings.local_model.clone()
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn transcribe<T, Fut>(
    app: &AppHandle<AppRuntime>,
    client: &Client,
    settings: &UserSettings,
    model_id: &str,
    wav_path: &Path,
    local_fallback_model: &str,
    wants_timestamps: bool,
    is_cancelled: impl Fn() -> bool,
    map_remote: impl FnOnce(TranscriptionSuccess) -> T,
    local: impl FnOnce() -> Fut,
) -> Result<T>
where
    Fut: std::future::Future<Output = Result<T>>,
{
    if !(remote::is_remote_model(model_id) && remote::is_configured(settings)) {
        return local().await;
    }

    match remote::attempt_remote(
        app,
        client,
        settings,
        wav_path,
        local_fallback_model,
        remote::TranscribeOptions {
            timestamps: wants_timestamps,
            diarization: false,
        },
        is_cancelled,
    )
    .await
    {
        remote::RemoteAttempt::Success(success) => Ok(map_remote(success.transcription)),
        remote::RemoteAttempt::Fallback => local().await,
        remote::RemoteAttempt::Cancelled => Err(anyhow!("Transcription cancelled")),
        remote::RemoteAttempt::Unavailable(message) => Err(anyhow!(message)),
    }
}

/// The speaker diarization model, once fully downloaded.
/// Nemotron-3 once downloaded, otherwise the Sortformer v2.1 diarizer a
/// previous version installed, which still works until the upgrade lands.
pub(crate) fn installed_diarizer_path(app: &AppHandle<AppRuntime>) -> Option<PathBuf> {
    let models_dir = install::model_cache_dir(app).ok()?;
    current_diarizer_path(&models_dir).or_else(|| {
        let retired = models_dir
            .join(catalog::RETIRED_DIARIZER_MODEL)
            .join(catalog::RETIRED_DIARIZER_FILE);
        retired.is_file().then_some(retired)
    })
}

/// The Nemotron-3 diarizer, once fully downloaded. Sortformer v2.1 has no live mode.
pub(crate) fn live_diarizer_path(app: &AppHandle<AppRuntime>) -> Option<PathBuf> {
    current_diarizer_path(&install::model_cache_dir(app).ok()?)
}

fn current_diarizer_path(models_dir: &std::path::Path) -> Option<PathBuf> {
    let manager = glimpse_speech::models::ModelInstallManager::new(models_dir);
    let spec = catalog::install_spec(catalog::DIARIZER_MODEL, false)?;
    manager.resolve(&spec).ok().map(|resolved| resolved.path)
}

/// People who installed the Sortformer v2.1 diarizer chose speaker detection,
/// so download its replacement in the background, then remove the old one.
pub(crate) fn upgrade_retired_diarizer(app: &AppHandle<AppRuntime>) {
    let Ok(models_dir) = install::model_cache_dir(app) else {
        return;
    };
    let retired = models_dir.join(catalog::RETIRED_DIARIZER_MODEL);
    if !retired.exists() {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let upgraded = current_diarizer_path(&models_dir).is_none();
        if upgraded {
            if let Err(err) = install::download_model_now(
                app.clone(),
                catalog::DIARIZER_MODEL.into(),
                Some(false),
            )
            .await
            {
                tracing::warn!("[speech] speaker model upgrade failed, keeping Sortformer: {err}");
                return;
            }
            // A cancelled download also returns Ok, so only a verified install replaces Sortformer.
            if current_diarizer_path(&models_dir).is_none() {
                tracing::warn!("[speech] speaker model upgrade did not finish, keeping Sortformer");
                return;
            }
        }
        if let Err(err) = crate::platform::remove_dir_all_compat(&retired) {
            tracing::warn!("[speech] could not remove {}: {err}", retired.display());
        }
        if upgraded {
            crate::toast::show(
                &app,
                "success",
                None,
                &crate::toast::native(&app, "native.toast.speaker_model_upgraded"),
            );
        }
    });
}

/// Whisper now runs on transcribe.cpp, which rejects the whisper.cpp Core ML
/// encoders earlier versions downloaded, so free their disk space along with
/// `.bin` downloads that will never resume.
pub(crate) fn remove_whisper_cpp_files(app: &AppHandle<AppRuntime>) {
    let Ok(models_dir) = install::model_cache_dir(app) else {
        return;
    };
    std::thread::spawn(move || {
        for manifest in catalog::local_manifests() {
            let model_dir = models_dir.join(manifest.id);
            if let Some(partial) = catalog::whisper_bin_partial(manifest) {
                let _ = std::fs::remove_file(model_dir.join(partial));
            }
            if !catalog::ANE_SUPPORTED {
                continue;
            }
            let Some(dir_name) = catalog::whisper_cpp_encoder_dir(manifest) else {
                continue;
            };
            let _ = std::fs::remove_file(model_dir.join(format!("{dir_name}.zip")));
            let encoder = model_dir.join(&dir_name);
            let Ok(metadata) = encoder.symlink_metadata() else {
                continue;
            };
            // Unlink a symlinked encoder instead of emptying its target.
            let removed = if metadata.file_type().is_symlink() {
                std::fs::remove_file(&encoder)
            } else {
                crate::platform::remove_dir_all_compat(&encoder)
            };
            match removed {
                Ok(()) => {
                    let _ =
                        std::fs::remove_file(model_dir.join(format!(".{dir_name}.manifest.json")));
                    tracing::info!("[speech] removed whisper.cpp encoder {}", encoder.display());
                }
                Err(err) => {
                    tracing::warn!("[speech] could not remove {}: {err}", encoder.display())
                }
            }
        }
    });
}

pub fn warm(app: &AppHandle<AppRuntime>, settings: &UserSettings) {
    if remote::is_configured(settings) {
        return;
    }

    warm_model(app, settings.local_model.clone());
}

/// Loads a model off-thread. The idle monitor unloads it again if it goes unused.
pub fn warm_model(app: &AppHandle<AppRuntime>, model_key: String) {
    let app_handle = app.clone();
    std::thread::spawn(move || {
        let ready = match install::ensure_model_ready(&app_handle, &model_key) {
            Ok(model) => model,
            Err(err) => {
                tracing::error!("[speech] skipping warm: {err}");
                return;
            }
        };
        let transcriber = app_handle.state::<AppState>().local_transcriber();
        if let Err(err) = transcriber.preload_and_warm_if_needed(&ready) {
            tracing::error!("[speech] warm failed: {err}");
        }
    });
}
