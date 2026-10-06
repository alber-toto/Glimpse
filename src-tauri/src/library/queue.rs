use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use chrono::Utc;
use tauri::{AppHandle, Emitter, Manager, async_runtime};
use tokio_util::sync::CancellationToken;
use webrtc_vad::VadMode;

use crate::transcribe::count_words;
use crate::{
    AppRuntime, AppState, LibraryJob, LibraryJobKind, dictionary, model_manager,
    recorder::speech_percentage_i16_with_mode, remote_speech, settings::UserSettings,
    storage::StorageManager, toast, transcribe, transcription_api,
};

use super::processing::{
    WavInfo, compute_total_chunks, convert_library_item, convert_segments_to_ms, diarize_segments,
    read_wav_info, stream_wav_chunks,
};
use super::types::{
    CHUNK_OVERLAP_SECONDS, DIRECT_TRANSCRIBE_MINUTES, EVENT_LIBRARY_COMPLETE, EVENT_LIBRARY_ERROR,
    EVENT_LIBRARY_METADATA_PROCESSING, EVENT_LIBRARY_PROGRESS, EVENT_LIBRARY_UPDATED,
    LibraryCompletePayload, LibraryErrorPayload, LibraryItem, LibraryItemPatch, LibraryItemStatus,
    LibraryMetadataProcessingPayload, LibraryProgressPayload, LibraryProgressUpdate,
    LibraryTranscriptionResult, LibraryUpdatedPayload, MAX_CHUNK_MINUTES, Speaker,
    TranscriptSegment, cancelled_error, is_cancelled_error, is_ffmpeg_error_message,
    is_meeting_item_kind,
};
use crate::speech::{
    VAD_MIN_SPEECH_PERCENT_CHUNK, VAD_MIN_SPEECH_PERCENT_FILE, WHISPER_CHUNK_OVERLAP_SECONDS,
    WHISPER_CHUNK_SECONDS,
};

fn start_library_job_internal(app: &AppHandle<AppRuntime>, job: LibraryJob) {
    let app_handle = app.clone();
    async_runtime::spawn(async move {
        let state_handle = app_handle.state::<AppState>();
        let job_id = job.id.clone();
        let token = state_handle.register_library_transcription(job_id.clone());

        match job.kind {
            LibraryJobKind::Import {
                source_path,
                store_original,
            } => {
                let app_for_task = app_handle.clone();
                let token_for_task = token.clone();
                let job_id_for_task = job_id.clone();
                let result = async_runtime::spawn_blocking(move || {
                    let state_for_task = app_for_task.state::<AppState>();
                    convert_library_item(
                        &app_for_task,
                        &state_for_task,
                        &job_id_for_task,
                        &source_path,
                        store_original,
                        &token_for_task,
                    )
                })
                .await;

                match result {
                    Ok(Ok(())) => {
                        if token.is_cancelled() {
                            handle_library_job_error(
                                &app_handle,
                                &state_handle,
                                &job_id,
                                cancelled_error(),
                            );
                            return;
                        }
                        start_library_transcription_internal(&app_handle, &state_handle, job_id);
                    }
                    Ok(Err(err)) => {
                        handle_library_job_error(&app_handle, &state_handle, &job_id, err);
                    }
                    Err(err) => {
                        handle_library_job_error(
                            &app_handle,
                            &state_handle,
                            &job_id,
                            anyhow!("Library import task failed: {err}"),
                        );
                    }
                }
            }
            LibraryJobKind::TranscribeExisting => {
                if token.is_cancelled() {
                    handle_library_job_error(
                        &app_handle,
                        &state_handle,
                        &job_id,
                        cancelled_error(),
                    );
                    return;
                }
                start_library_transcription_internal(&app_handle, &state_handle, job_id);
            }
        }
    });
}

fn start_library_transcription_internal(
    app: &AppHandle<AppRuntime>,
    state: &tauri::State<'_, AppState>,
    id: String,
) {
    let storage = state.storage();
    let item = match storage.get_library_item(&id) {
        Ok(Some(item)) => item,
        Ok(None) => {
            tracing::error!("Library item not found for transcription: {id}");
            let _ = app.emit(
                EVENT_LIBRARY_ERROR,
                LibraryErrorPayload {
                    id: id.clone(),
                    message: "Library item not found".to_string(),
                    cancelled: false,
                },
            );
            release_library_slot(app, state, &id);
            return;
        }
        Err(err) => {
            tracing::error!("Failed to load library item {id}: {err}");
            let _ = app.emit(
                EVENT_LIBRARY_ERROR,
                LibraryErrorPayload {
                    id: id.clone(),
                    message: format!("Failed to load library item: {err}"),
                    cancelled: false,
                },
            );
            release_library_slot(app, state, &id);
            return;
        }
    };

    if matches!(
        item.status,
        LibraryItemStatus::Cancelling | LibraryItemStatus::Cancelled
    ) {
        release_library_slot(app, state, &id);
        return;
    }

    if matches!(item.status, LibraryItemStatus::Transcribing { .. }) {
        release_library_slot(app, state, &id);
        return;
    }

    let _ = storage.update_library_item(
        &id,
        LibraryItemPatch {
            status: Some(LibraryItemStatus::Transcribing { progress: 0.0 }),
            transcript: Some(String::new()),
            segments: Some(Vec::new()),
            ..Default::default()
        },
    );
    let _ = app.emit(
        EVENT_LIBRARY_PROGRESS,
        LibraryProgressPayload {
            id: id.clone(),
            progress: 0.0,
            current_chunk: 0,
            total_chunks: 0,
            chunk_text: None,
            chunk_segments: None,
        },
    );

    let token = state.register_library_transcription(id.clone());
    let app_handle = app.clone();
    let organize_after_completion = should_automatically_organize(item.transcribed_at.as_deref());
    let item_for_task = item.clone();
    let transcription_started_at = Instant::now();
    async_runtime::spawn(async move {
        let id_for_release = id.clone();
        let token_handle = token.clone();
        let app_for_task = app_handle.clone();
        let result = async_runtime::spawn_blocking(move || {
            let state_handle = app_for_task.state::<AppState>();
            transcribe_library_item_for_kind(
                &app_for_task,
                &state_handle,
                &item_for_task,
                &token_handle,
            )
        })
        .await;

        let state_handle = app_handle.state::<AppState>();

        match result {
            Ok(Ok(mut result)) => {
                let mut final_transcript = result.transcript.clone();
                let settings = state_handle.current_settings();
                final_transcript =
                    dictionary::apply_replacements(&final_transcript, &settings.replacements);
                if !settings.replacements.is_empty() {
                    for entries in [result.segments.as_mut(), result.words.as_mut()]
                        .into_iter()
                        .flatten()
                    {
                        for entry in entries.iter_mut() {
                            entry.text =
                                dictionary::apply_replacements(&entry.text, &settings.replacements);
                        }
                    }
                }

                if count_words(&final_transcript) == 0 {
                    let speech_model = result
                        .speech_model
                        .as_deref()
                        .filter(|model| !model.trim().is_empty())
                        .unwrap_or(&item.speech_model);
                    crate::analytics::track_transcription_failed(
                        &app_handle,
                        "transcription",
                        library_transcription_mode(speech_model),
                        speech_model,
                        "no_speech",
                        Some(item.duration_seconds),
                        "uploaded_file",
                    );
                    let _ = storage.update_library_item(
                        &id,
                        LibraryItemPatch {
                            status: Some(LibraryItemStatus::Error {
                                message: "No speech detected".to_string(),
                            }),
                            ..Default::default()
                        },
                    );
                    let _ = app_handle.emit(
                        EVENT_LIBRARY_ERROR,
                        LibraryErrorPayload {
                            id: id.clone(),
                            message: "No speech detected".to_string(),
                            cancelled: false,
                        },
                    );
                } else {
                    let transcript_for_title = final_transcript.clone();
                    let speech_model = result
                        .speech_model
                        .as_deref()
                        .filter(|model| !model.trim().is_empty())
                        .unwrap_or(&item.speech_model);
                    let model_label = crate::model_manager::model_label(speech_model);
                    crate::analytics::track_transcription_completed(
                        &app_handle,
                        library_transcription_mode(speech_model),
                        Some(&model_label),
                        false,
                        item.duration_seconds,
                        transcription_started_at.elapsed().as_secs_f32(),
                        count_words(&final_transcript),
                        "uploaded_file",
                    );
                    let _ = storage.update_library_item(
                        &id,
                        LibraryItemPatch {
                            status: Some(LibraryItemStatus::Complete),
                            transcript: Some(final_transcript),
                            segments: result.segments.take(),
                            words: result.words.take(),
                            speech_model: result.speech_model.take(),
                            speakers: Some(result.speakers.take()),
                            transcribed_at: Some(Utc::now().to_rfc3339()),
                            ..Default::default()
                        },
                    );

                    let _ = app_handle.emit(
                        EVENT_LIBRARY_COMPLETE,
                        LibraryCompletePayload { id: id.clone() },
                    );
                    if organize_after_completion {
                        schedule_automatic_library_organization(
                            app_handle.clone(),
                            id.clone(),
                            item.name.clone(),
                            transcript_for_title,
                        );
                    }
                }
            }
            Ok(Err(err)) => {
                let cancelled = is_cancelled_error(&err);
                let message = err.to_string();
                if !cancelled {
                    crate::analytics::track_transcription_failed(
                        &app_handle,
                        "transcription",
                        library_transcription_mode(&item.speech_model),
                        &item.speech_model,
                        crate::analytics::classify_failure_reason(&message),
                        Some(item.duration_seconds),
                        "uploaded_file",
                    );
                }
                let status = if cancelled {
                    LibraryItemStatus::Cancelled
                } else {
                    LibraryItemStatus::Error {
                        message: message.clone(),
                    }
                };
                let _ = storage.update_library_item(
                    &id,
                    LibraryItemPatch {
                        status: Some(status),
                        ..Default::default()
                    },
                );
                let _ = app_handle.emit(
                    EVENT_LIBRARY_ERROR,
                    LibraryErrorPayload {
                        id: id.clone(),
                        cancelled,
                        message,
                    },
                );
            }
            Err(err) => {
                let message = format!("Library transcription task failed: {err}");
                crate::analytics::track_transcription_failed(
                    &app_handle,
                    "transcription",
                    library_transcription_mode(&item.speech_model),
                    &item.speech_model,
                    "task_failed",
                    Some(item.duration_seconds),
                    "uploaded_file",
                );
                let _ = storage.update_library_item(
                    &id,
                    LibraryItemPatch {
                        status: Some(LibraryItemStatus::Error {
                            message: message.clone(),
                        }),
                        ..Default::default()
                    },
                );
                let _ = app_handle.emit(
                    EVENT_LIBRARY_ERROR,
                    LibraryErrorPayload {
                        id: id.clone(),
                        cancelled: false,
                        message,
                    },
                );
            }
        }

        release_library_slot(&app_handle, &state_handle, &id_for_release);
    });
}

fn should_automatically_organize(transcribed_at: Option<&str>) -> bool {
    transcribed_at.is_none()
}

fn schedule_automatic_library_organization(
    app: AppHandle<AppRuntime>,
    id: String,
    original_name: String,
    transcript: String,
) {
    async_runtime::spawn(async move {
        let state = app.state::<AppState>();
        let settings = state.current_settings_unmasked();
        let Some(title_settings) = crate::llm_cleanup::title_generation_settings(&settings, true)
        else {
            return;
        };
        let _ = app.emit(
            EVENT_LIBRARY_METADATA_PROCESSING,
            LibraryMetadataProcessingPayload {
                id: id.clone(),
                active: true,
            },
        );
        let available_tags = match state.storage().get_library_tags() {
            Ok(tags) => tags,
            Err(err) => {
                tracing::warn!(item_id = %id, "Automatic library organization skipped: failed to load tags: {err}");
                Vec::new()
            }
        };
        let app_locale = crate::native_i18n::resolved_app_locale(&settings);

        let metadata = match crate::llm_cleanup::generate_library_metadata(
            &state.http(),
            &transcript,
            &available_tags,
            app_locale,
            &title_settings,
        )
        .await
        {
            Ok(metadata) => metadata,
            Err(err) => {
                tracing::warn!(
                    item_id = %id,
                    "Automatic library organization skipped: {}",
                    crate::llm_cleanup::llm_issue_message(&err)
                );
                let _ = app.emit(
                    EVENT_LIBRARY_METADATA_PROCESSING,
                    LibraryMetadataProcessingPayload { id, active: false },
                );
                return;
            }
        };

        match state.storage().apply_generated_library_metadata(
            &id,
            Some(&original_name),
            &metadata.title,
            &metadata.tags,
        ) {
            Ok(Some(_)) => {
                let _ = app.emit(
                    EVENT_LIBRARY_UPDATED,
                    LibraryUpdatedPayload { id: id.clone() },
                );
            }
            Ok(None) => {
                tracing::debug!(item_id = %id, "Automatic library organization kept the user's edits");
            }
            Err(err) => {
                tracing::warn!(item_id = %id, "Failed to save automatic library title and tags: {err}");
            }
        }
        let _ = app.emit(
            EVENT_LIBRARY_METADATA_PROCESSING,
            LibraryMetadataProcessingPayload { id, active: false },
        );
    });
}

fn handle_library_job_error(
    app: &AppHandle<AppRuntime>,
    state: &tauri::State<'_, AppState>,
    id: &str,
    err: anyhow::Error,
) {
    let cancelled = is_cancelled_error(&err);
    let message = err.to_string();
    let status = if cancelled {
        LibraryItemStatus::Cancelled
    } else {
        LibraryItemStatus::Error {
            message: message.clone(),
        }
    };
    if is_ffmpeg_error_message(&message) && state.should_show_ffmpeg_toast() {
        toast::show_with_action(
            app,
            "error",
            Some("FFmpeg Required"),
            "FFmpeg is required to import this file.",
            "open_ffmpeg_install",
            "FFmpeg Help",
        );
    }
    let _ = state.storage().update_library_item(
        id,
        LibraryItemPatch {
            status: Some(status),
            ..Default::default()
        },
    );
    let _ = app.emit(
        EVENT_LIBRARY_ERROR,
        LibraryErrorPayload {
            id: id.to_string(),
            cancelled,
            message,
        },
    );
    release_library_slot(app, state, id);
}

fn library_transcription_mode(model: &str) -> &'static str {
    if remote_speech::is_remote_model(model) {
        "remote"
    } else {
        "local"
    }
}

pub(crate) fn schedule_library_job(
    app: &AppHandle<AppRuntime>,
    state: &tauri::State<'_, AppState>,
    job: LibraryJob,
) {
    if !state.enqueue_library_job(job) {
        return;
    }
    start_next_library_job(app, state);
}

fn start_next_library_job(app: &AppHandle<AppRuntime>, state: &tauri::State<'_, AppState>) {
    let Some(job) = state.claim_next_library_job() else {
        return;
    };
    start_library_job_internal(app, job);
}

pub(crate) fn release_library_slot(
    app: &AppHandle<AppRuntime>,
    state: &tauri::State<'_, AppState>,
    id: &str,
) {
    state.clear_active_library_job(id);
    state.clear_library_transcription(id);
    start_next_library_job(app, state);
}

struct LocalRun<'a> {
    app: &'a AppHandle<AppRuntime>,
    state: &'a AppState,
    item: &'a LibraryItem,
    token: &'a CancellationToken,
    model: &'a model_manager::ReadyModel,
    dictionary: &'a [String],
    language: &'a str,
    sample_rate: u32,
    progress_scope: ProgressScope,
}

#[derive(Clone, Copy)]
struct ProgressScope {
    start: f32,
    end: f32,
    publish_content: bool,
}

impl ProgressScope {
    const FULL: Self = Self {
        start: 0.0,
        end: 1.0,
        publish_content: true,
    };

    fn meeting_track(index: usize, total: usize) -> Self {
        let total = total.max(1) as f32;
        Self {
            start: index as f32 / total,
            end: (index + 1) as f32 / total,
            publish_content: false,
        }
    }

    fn map(self, mut update: LibraryProgressUpdate) -> LibraryProgressUpdate {
        update.progress = self.start + update.progress.clamp(0.0, 1.0) * (self.end - self.start);
        if !self.publish_content {
            update.transcript = None;
            update.segments = None;
            update.chunk_text = None;
            update.chunk_segments = None;
        }
        update
    }
}

struct ChunkPlan {
    chunk_size: usize,
    overlap: usize,
    step: usize,
}

impl ChunkPlan {
    fn new(chunk_seconds: usize, overlap_seconds: usize, sample_rate: u32) -> Self {
        let chunk_size = (chunk_seconds * sample_rate as usize).max(1);
        let overlap = (overlap_seconds * sample_rate as usize).min(chunk_size.saturating_sub(1));
        let step = chunk_size.saturating_sub(overlap).max(1);
        Self {
            chunk_size,
            overlap,
            step,
        }
    }
}

fn offset_ms(start_idx: usize, sample_rate: u32) -> u64 {
    (start_idx as f64 / sample_rate as f64 * 1000.0) as u64
}

fn chunk_below_speech_gate(chunk: &[i16], sample_rate: u32) -> bool {
    speech_percentage_i16_with_mode(chunk, sample_rate, VadMode::VeryAggressive)
        < VAD_MIN_SPEECH_PERCENT_CHUNK
}

#[derive(Clone, Copy)]
enum MeetingTrackSource {
    Microphone,
    System,
}

impl MeetingTrackSource {
    fn speaker_id(self) -> &'static str {
        match self {
            Self::Microphone => "meeting_you",
            Self::System => "meeting_remote",
        }
    }

    fn speaker_name(self, app: &AppHandle<AppRuntime>) -> String {
        match self {
            Self::Microphone => crate::toast::native(app, "native.meeting.speaker_you"),
            Self::System => crate::toast::native(app, "native.meeting.speaker_remote"),
        }
    }
}

fn transcribe_library_item_for_kind(
    app: &AppHandle<AppRuntime>,
    state: &AppState,
    item: &LibraryItem,
    token: &CancellationToken,
) -> Result<LibraryTranscriptionResult> {
    if !is_meeting_item_kind(&item.kind) {
        return transcribe_library_item(app, state, item, token, ProgressScope::FULL);
    }

    let Some(item_dir) = Path::new(&item.audio_path).parent() else {
        return transcribe_library_item(app, state, item, token, ProgressScope::FULL);
    };
    let candidates = [
        (
            MeetingTrackSource::Microphone,
            item_dir.join("microphone.wav"),
        ),
        (MeetingTrackSource::System, item_dir.join("system.wav")),
    ];
    let tracks: Vec<_> = candidates
        .into_iter()
        .filter(|(_, path)| {
            read_wav_info(path)
                .map(|info| info.total_samples > 0)
                .unwrap_or(false)
        })
        .collect();
    if tracks.is_empty() {
        // Compatibility with meetings recorded before source tracks were
        // retained, and with any manually repaired Library item.
        return transcribe_library_item(app, state, item, token, ProgressScope::FULL);
    }

    transcribe_meeting_tracks(app, state, item, token, &tracks)
}

fn transcribe_meeting_tracks(
    app: &AppHandle<AppRuntime>,
    state: &AppState,
    item: &LibraryItem,
    token: &CancellationToken,
    tracks: &[(MeetingTrackSource, PathBuf)],
) -> Result<LibraryTranscriptionResult> {
    let mut results = Vec::with_capacity(tracks.len());
    for (index, (source, path)) in tracks.iter().enumerate() {
        if token.is_cancelled() {
            return Err(cancelled_error());
        }
        let scope = ProgressScope::meeting_track(index, tracks.len());
        let mut track_item = item.clone();
        track_item.audio_path = path.display().to_string();
        track_item.detect_speakers =
            meeting_track_person_detection_enabled(item.detect_speakers, *source);
        let duration_ms = read_wav_info(path)
            .map(|info| (info.duration_seconds * 1000.0).round() as u64)
            .unwrap_or_else(|_| (item.duration_seconds * 1000.0).round() as u64);
        let result = transcribe_library_item(app, state, &track_item, token, scope)?;
        results.push(label_meeting_track(
            result,
            *source,
            duration_ms,
            source.speaker_name(app),
        ));
        report_progress(
            app,
            state.storage(),
            &item.id,
            scope.map(LibraryProgressUpdate::with_chunk_counts(1.0, 1, 1)),
        );
    }

    Ok(merge_meeting_results(results))
}

fn meeting_track_person_detection_enabled(
    meeting_detection_enabled: bool,
    source: MeetingTrackSource,
) -> bool {
    meeting_detection_enabled && matches!(source, MeetingTrackSource::System)
}

fn label_meeting_track(
    mut result: LibraryTranscriptionResult,
    source: MeetingTrackSource,
    duration_ms: u64,
    speaker_name: String,
) -> LibraryTranscriptionResult {
    if matches!(source, MeetingTrackSource::System)
        && result
            .speakers
            .as_ref()
            .is_some_and(|speakers| !speakers.is_empty())
    {
        return result;
    }
    let speaker_id = source.speaker_id().to_string();
    let speaker = Speaker {
        id: speaker_id.clone(),
        name: speaker_name,
        color: None,
    };

    let segments = result.segments.get_or_insert_with(Vec::new);
    if segments.is_empty() && !result.transcript.trim().is_empty() {
        segments.push(TranscriptSegment {
            start_ms: 0,
            end_ms: duration_ms.max(1),
            text: result.transcript.trim().to_string(),
            speaker_id: Some(speaker_id.clone()),
        });
    } else {
        for segment in segments.iter_mut() {
            segment.speaker_id = Some(speaker_id.clone());
        }
    }
    if let Some(words) = result.words.as_mut() {
        for word in words {
            word.speaker_id = Some(speaker_id.clone());
        }
    }
    result.speakers = Some(vec![speaker]);
    result
}

fn merge_meeting_results(results: Vec<LibraryTranscriptionResult>) -> LibraryTranscriptionResult {
    let mut segments = Vec::new();
    let mut words = Vec::new();
    let mut speakers: Vec<Speaker> = Vec::new();
    let mut speech_model = None;

    for mut result in results {
        segments.extend(result.segments.take().unwrap_or_default());
        words.extend(result.words.take().unwrap_or_default());
        for speaker in result.speakers.take().unwrap_or_default() {
            if !speakers.iter().any(|existing| existing.id == speaker.id) {
                speakers.push(speaker);
            }
        }
        if result.speech_model.is_some() {
            speech_model = result.speech_model;
        }
    }

    segments.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then(left.end_ms.cmp(&right.end_ms))
            .then(left.speaker_id.cmp(&right.speaker_id))
    });
    words.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then(left.end_ms.cmp(&right.end_ms))
            .then(left.speaker_id.cmp(&right.speaker_id))
    });

    let transcript = segments
        .iter()
        .filter_map(|segment| {
            let text = segment.text.trim();
            if text.is_empty() {
                return None;
            }
            let name = segment
                .speaker_id
                .as_deref()
                .and_then(|id| speakers.iter().find(|speaker| speaker.id == id))
                .map(|speaker| speaker.name.as_str())
                .unwrap_or("Speaker");
            Some(format!("{name}: {text}"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    LibraryTranscriptionResult {
        transcript,
        segments: (!segments.is_empty()).then_some(segments),
        words: (!words.is_empty()).then_some(words),
        speech_model,
        speakers: (!speakers.is_empty()).then_some(speakers),
    }
}

fn transcribe_library_item(
    app: &AppHandle<AppRuntime>,
    state: &AppState,
    item: &LibraryItem,
    token: &CancellationToken,
    progress_scope: ProgressScope,
) -> Result<LibraryTranscriptionResult> {
    if token.is_cancelled() {
        return Err(cancelled_error());
    }

    let audio_path = PathBuf::from(&item.audio_path);
    if !audio_path.exists() {
        return Err(anyhow!("Audio file not found"));
    }

    let wav_info = read_wav_info(&audio_path)?;
    if wav_info.total_samples == 0 {
        return Err(anyhow!("No audio data decoded from WAV file"));
    }

    let settings = state.current_settings();

    let wants_remote = remote_speech::is_remote_model(&item.speech_model)
        && remote_speech::is_configured(&settings);
    let mut remote_fallback = false;
    if wants_remote {
        match transcribe_remote(
            app,
            state,
            &settings,
            item,
            &audio_path,
            token,
            progress_scope,
        )? {
            Some(result) => {
                return apply_local_diarization_if_needed(app, item, token, &audio_path, result);
            }
            None => remote_fallback = true,
        }
    }

    let ready_model = if remote_fallback || remote_speech::is_remote_model(&item.speech_model) {
        model_manager::ensure_local_fallback_model(app, &settings.local_model)?
    } else {
        model_manager::ensure_model_ready(app, &item.speech_model)?
    };
    let dictionary = dictionary::dictionary_entries_for_model(&ready_model, &settings);
    let language = settings.language.clone();

    let run = LocalRun {
        app,
        state,
        item,
        token,
        model: &ready_model,
        dictionary: &dictionary,
        language: &language,
        sample_rate: wav_info.sample_rate,
        progress_scope,
    };

    let result = if matches!(ready_model.engine, model_manager::LocalModelEngine::Whisper) {
        transcribe_whisper_chunked(&run, &audio_path, &wav_info)
    } else if wav_info.duration_seconds <= (DIRECT_TRANSCRIBE_MINUTES as f32 * 60.0) {
        transcribe_direct(&run, &audio_path)
    } else {
        transcribe_parakeet_chunked(&run, &audio_path, &wav_info)
    }?;
    apply_local_diarization_if_needed(app, item, token, &audio_path, result)
}

fn apply_local_diarization_if_needed(
    app: &AppHandle<AppRuntime>,
    item: &LibraryItem,
    token: &CancellationToken,
    audio_path: &Path,
    mut result: LibraryTranscriptionResult,
) -> Result<LibraryTranscriptionResult> {
    if !item.detect_speakers
        || result
            .speakers
            .as_ref()
            .is_some_and(|speakers| !speakers.is_empty())
        || !crate::diarization::is_installed(app)
    {
        return Ok(result);
    }
    if token.is_cancelled() {
        return Err(cancelled_error());
    }
    let segments = crate::diarization::run(app, audio_path)?;
    if token.is_cancelled() {
        return Err(cancelled_error());
    }
    let settings = app.state::<AppState>().current_settings();
    let strings = crate::native_i18n::MenuStrings::resolve(&settings);
    crate::diarization::apply_to_transcription(&mut result, &segments, |next_index| {
        strings.format(
            "native.library.person_default",
            &[("nextIndex", &next_index.to_string())],
        )
    });
    Ok(result)
}

// Ok(Some) = done, Ok(None) = fall back to local, Err = cancel/unavailable.
fn transcribe_remote(
    app: &AppHandle<AppRuntime>,
    state: &AppState,
    settings: &UserSettings,
    item: &LibraryItem,
    audio_path: &Path,
    token: &CancellationToken,
    progress_scope: ProgressScope,
) -> Result<Option<LibraryTranscriptionResult>> {
    let http = state.http();
    let remote_diarization = item.detect_speakers
        && glimpse_speech::remote::supports_diarization(&remote_speech::resolved_endpoint(
            settings,
        ));
    let attempt = async_runtime::block_on(remote_speech::attempt_remote(
        app,
        &http,
        settings,
        audio_path,
        &settings.local_model,
        remote_speech::TranscribeOptions {
            timestamps: true,
            diarization: remote_diarization,
        },
        || token.is_cancelled(),
    ));
    match attempt {
        remote_speech::RemoteAttempt::Success(success) => {
            let result = success.transcription;
            report_progress(
                app,
                state.storage(),
                &item.id,
                progress_scope.map(LibraryProgressUpdate::with_chunk_counts(1.0, 1, 1)),
            );
            let (segments, speakers) = match success.diarized_segments.as_deref() {
                Some(segs) => {
                    let strings = crate::native_i18n::MenuStrings::resolve(settings);
                    let (converted, speakers) = diarize_segments(segs, |next_index| {
                        strings.format(
                            "native.library.person_default",
                            &[("nextIndex", &next_index.to_string())],
                        )
                    });
                    (Some(converted), speakers)
                }
                None => (result.segments.as_deref().map(convert_segments_to_ms), None),
            };
            let words = result.words.as_deref().map(convert_segments_to_ms);
            Ok(Some(LibraryTranscriptionResult {
                transcript: result.transcript,
                segments,
                words,
                speech_model: result.speech_model,
                speakers,
            }))
        }
        remote_speech::RemoteAttempt::Cancelled => Err(cancelled_error()),
        remote_speech::RemoteAttempt::Unavailable(message) => Err(anyhow!(message)),
        remote_speech::RemoteAttempt::Fallback => Ok(None),
    }
}

fn transcribe_direct(run: &LocalRun, audio_path: &Path) -> Result<LibraryTranscriptionResult> {
    let (samples, sample_rate) = transcribe::load_audio_for_transcription(audio_path)?;
    let speech_percent =
        speech_percentage_i16_with_mode(&samples, sample_rate, VadMode::VeryAggressive);
    if speech_percent < VAD_MIN_SPEECH_PERCENT_FILE {
        return Ok(LibraryTranscriptionResult {
            transcript: String::new(),
            segments: None,
            words: None,
            speech_model: None,
            speakers: None,
        });
    }

    let result = run.state.local_transcriber().transcribe_with_segments(
        run.model,
        &samples,
        sample_rate,
        run.dictionary,
        Some(run.language),
    )?;
    if run.token.is_cancelled() {
        return Err(cancelled_error());
    }

    Ok(LibraryTranscriptionResult {
        transcript: result.transcript,
        segments: result.segments.as_deref().map(convert_segments_to_ms),
        words: result.words.as_deref().map(convert_segments_to_ms),
        speech_model: None,
        speakers: None,
    })
}

fn transcribe_whisper_chunked(
    run: &LocalRun,
    audio_path: &Path,
    wav_info: &WavInfo,
) -> Result<LibraryTranscriptionResult> {
    let sample_rate = run.sample_rate;
    let transcriber = run.state.local_transcriber();
    let plan = ChunkPlan::new(
        WHISPER_CHUNK_SECONDS as usize,
        WHISPER_CHUNK_OVERLAP_SECONDS as usize,
        sample_rate,
    );

    let mut total_chunks =
        compute_total_chunks(wav_info.total_samples, plan.chunk_size, plan.step).max(1);
    let mut full_text = String::new();
    let mut merged_segments: Vec<TranscriptSegment> = Vec::new();
    let mut merged_words: Vec<TranscriptSegment> = Vec::new();
    let mut last_end_ms: u64 = 0;
    let mut chunk_index: u32 = 0;

    stream_wav_chunks(
        audio_path,
        plan.chunk_size,
        plan.overlap,
        |start_idx, chunk| {
            if run.token.is_cancelled() {
                return Err(cancelled_error());
            }

            chunk_index = chunk_index.saturating_add(1);
            let remaining = wav_info
                .total_samples
                .saturating_sub(start_idx + chunk.len());
            total_chunks = total_chunks.max(chunk_index + u32::from(remaining > 0));
            if chunk_below_speech_gate(chunk, sample_rate) {
                let progress =
                    ((start_idx + chunk.len()) as f32 / wav_info.total_samples as f32).min(1.0);
                report_progress(
                    run.app,
                    run.state.storage(),
                    &run.item.id,
                    run.progress_scope
                        .map(LibraryProgressUpdate::with_chunk_counts(
                            progress,
                            chunk_index,
                            total_chunks,
                        )),
                );
                return Ok(());
            }
            let result = transcriber.transcribe_with_segments(
                run.model,
                chunk,
                sample_rate,
                run.dictionary,
                Some(run.language),
            )?;
            if run.token.is_cancelled() {
                return Err(cancelled_error());
            }

            let regions = glimpse_speech::vad::speech_regions(chunk, sample_rate);
            let in_speech = |start_ms: u64, end_ms: u64| match regions.as_deref() {
                Some(regions) => transcription_api::overlaps_speech(
                    start_ms as f32 / 1000.0,
                    end_ms as f32 / 1000.0,
                    regions,
                ),
                None => true,
            };

            let chunk_text = transcription_api::keep_spoken_segments(
                &result.transcript,
                result.segments.as_deref(),
                regions.as_deref(),
            );
            let mut appended_text = None;
            if !chunk_text.trim().is_empty() {
                let deduped = transcribe::dedupe_overlap_text(&full_text, &chunk_text);
                if !deduped.trim().is_empty() {
                    let appended = append_library_chunk(&mut full_text, &deduped);
                    appended_text = Some(appended);
                }
            }

            let mut new_segments: Vec<TranscriptSegment> = Vec::new();
            if let Some(segments) = result.segments {
                let offset = offset_ms(start_idx, sample_rate);
                for seg in convert_segments_to_ms(&segments) {
                    let start_ms = seg.start_ms + offset;
                    let end_ms = seg.end_ms + offset;
                    if end_ms <= last_end_ms || !in_speech(seg.start_ms, seg.end_ms) {
                        continue;
                    }
                    let new_segment = TranscriptSegment {
                        start_ms,
                        end_ms,
                        text: seg.text,
                        speaker_id: None,
                    };
                    merged_segments.push(new_segment.clone());
                    new_segments.push(new_segment);
                    last_end_ms = end_ms;
                }
            }

            if let Some(words) = result.words {
                let offset = offset_ms(start_idx, sample_rate);
                let spoken: Vec<_> = convert_segments_to_ms(&words)
                    .into_iter()
                    .filter(|w| in_speech(w.start_ms, w.end_ms))
                    .collect();

                let skip = if start_idx == 0 {
                    0
                } else {
                    let appended_words = appended_text
                        .as_deref()
                        .map_or(0, |t| t.split_whitespace().count());
                    spoken.len().saturating_sub(appended_words)
                };
                for word in spoken.into_iter().skip(skip) {
                    merged_words.push(TranscriptSegment {
                        start_ms: word.start_ms + offset,
                        end_ms: word.end_ms + offset,
                        text: word.text,
                        speaker_id: None,
                    });
                }
            }

            let progress =
                ((start_idx + chunk.len()) as f32 / wav_info.total_samples as f32).min(1.0);
            let transcript_patch = appended_text.as_ref().map(|_| full_text.clone());
            let segments_patch = if new_segments.is_empty() {
                None
            } else {
                Some(merged_segments.clone())
            };
            let chunk_segments = if new_segments.is_empty() {
                None
            } else {
                Some(new_segments)
            };

            report_progress(
                run.app,
                run.state.storage(),
                &run.item.id,
                run.progress_scope.map(LibraryProgressUpdate {
                    progress,
                    current_chunk: chunk_index,
                    total_chunks,
                    transcript: transcript_patch,
                    segments: segments_patch,
                    chunk_text: appended_text,
                    chunk_segments,
                }),
            );
            Ok(())
        },
    )?;

    Ok(LibraryTranscriptionResult {
        transcript: full_text.trim().to_string(),
        segments: if merged_segments.is_empty() {
            None
        } else {
            Some(merged_segments)
        },
        words: (!merged_words.is_empty()).then_some(merged_words),
        speech_model: None,
        speakers: None,
    })
}

fn transcribe_parakeet_chunked(
    run: &LocalRun,
    audio_path: &Path,
    wav_info: &WavInfo,
) -> Result<LibraryTranscriptionResult> {
    let sample_rate = run.sample_rate;
    let transcriber = run.state.local_transcriber();
    let plan = ChunkPlan::new(
        MAX_CHUNK_MINUTES as usize * 60,
        CHUNK_OVERLAP_SECONDS as usize,
        sample_rate,
    );

    let mut total_chunks =
        compute_total_chunks(wav_info.total_samples, plan.chunk_size, plan.step).max(1);
    let mut full_text = String::new();
    let mut merged_segments: Vec<TranscriptSegment> = Vec::new();
    let mut merged_words: Vec<TranscriptSegment> = Vec::new();
    let mut last_end_ms: u64 = 0;
    let mut last_word_end_ms: u64 = 0;
    let mut chunk_index: u32 = 0;

    stream_wav_chunks(
        audio_path,
        plan.chunk_size,
        plan.overlap,
        |start_idx, chunk| {
            if run.token.is_cancelled() {
                return Err(cancelled_error());
            }

            chunk_index = chunk_index.saturating_add(1);
            let remaining = wav_info
                .total_samples
                .saturating_sub(start_idx + chunk.len());
            total_chunks = total_chunks.max(chunk_index + u32::from(remaining > 0));
            if chunk_below_speech_gate(chunk, sample_rate) {
                let progress =
                    ((start_idx + chunk.len()) as f32 / wav_info.total_samples as f32).min(1.0);
                report_progress(
                    run.app,
                    run.state.storage(),
                    &run.item.id,
                    run.progress_scope
                        .map(LibraryProgressUpdate::with_chunk_counts(
                            progress,
                            chunk_index,
                            total_chunks,
                        )),
                );
                return Ok(());
            }
            let result = transcriber.transcribe_with_segments(
                run.model,
                chunk,
                sample_rate,
                run.dictionary,
                Some(run.language),
            )?;
            if run.token.is_cancelled() {
                return Err(cancelled_error());
            }

            let chunk_text = result.transcript;
            let mut kept_words = 0usize;
            let mut appended_text = None;
            if !chunk_text.trim().is_empty() {
                let deduped = transcribe::dedupe_overlap_text(&full_text, &chunk_text);
                if !deduped.trim().is_empty() {
                    kept_words = deduped.split_whitespace().count();
                    appended_text = Some(append_library_chunk(&mut full_text, &deduped));
                }
            }

            let mut new_segments: Vec<TranscriptSegment> = Vec::new();
            if let Some(segments) = result.segments {
                let offset = offset_ms(start_idx, sample_rate);
                for seg in convert_segments_to_ms(&segments) {
                    let start_ms = seg.start_ms + offset;
                    let end_ms = seg.end_ms + offset;
                    if end_ms <= last_end_ms {
                        continue;
                    }
                    let new_segment = TranscriptSegment {
                        start_ms,
                        end_ms,
                        text: seg.text,
                        speaker_id: None,
                    };
                    merged_segments.push(new_segment.clone());
                    new_segments.push(new_segment);
                    last_end_ms = end_ms;
                }
            }

            if let Some(words) = result.words {
                let offset = offset_ms(start_idx, sample_rate);
                let converted = convert_segments_to_ms(&words);
                let exact_skip = (chunk_text.split_whitespace().count() == converted.len())
                    .then(|| converted.len().saturating_sub(kept_words));
                let chunk_word_floor = last_word_end_ms;
                for (index, word) in converted.into_iter().enumerate() {
                    let start_ms = word.start_ms + offset;
                    let end_ms = word.end_ms + offset;
                    if matches!(exact_skip, Some(skip) if index < skip) {
                        continue;
                    }
                    if end_ms <= chunk_word_floor {
                        continue;
                    }
                    last_word_end_ms = last_word_end_ms.max(end_ms);
                    merged_words.push(TranscriptSegment {
                        start_ms,
                        end_ms,
                        text: word.text,
                        speaker_id: None,
                    });
                }
            }

            let progress =
                ((start_idx + chunk.len()) as f32 / wav_info.total_samples as f32).min(1.0);
            let transcript_patch = appended_text.as_ref().map(|_| full_text.clone());
            let segments_patch = if new_segments.is_empty() {
                None
            } else {
                Some(merged_segments.clone())
            };
            let chunk_segments = if new_segments.is_empty() {
                None
            } else {
                Some(new_segments)
            };
            report_progress(
                run.app,
                run.state.storage(),
                &run.item.id,
                run.progress_scope.map(LibraryProgressUpdate {
                    progress,
                    current_chunk: chunk_index,
                    total_chunks,
                    transcript: transcript_patch,
                    segments: segments_patch,
                    chunk_text: appended_text,
                    chunk_segments,
                }),
            );
            Ok(())
        },
    )?;

    Ok(LibraryTranscriptionResult {
        transcript: full_text.trim().to_string(),
        segments: if merged_segments.is_empty() {
            None
        } else {
            Some(merged_segments)
        },
        words: (!merged_words.is_empty()).then_some(merged_words),
        speech_model: None,
        speakers: None,
    })
}

fn report_progress(
    app: &AppHandle<AppRuntime>,
    storage: Arc<StorageManager>,
    id: &str,
    update: LibraryProgressUpdate,
) {
    let LibraryProgressUpdate {
        progress,
        current_chunk,
        total_chunks,
        transcript,
        segments,
        chunk_text,
        chunk_segments,
    } = update;

    let _ = storage.update_library_item(
        id,
        LibraryItemPatch {
            status: Some(LibraryItemStatus::Transcribing { progress }),
            transcript,
            segments,
            ..Default::default()
        },
    );
    let _ = app.emit(
        EVENT_LIBRARY_PROGRESS,
        LibraryProgressPayload {
            id: id.to_string(),
            progress,
            current_chunk,
            total_chunks,
            chunk_text,
            chunk_segments,
        },
    );
}

fn append_library_chunk(existing: &mut String, next: &str) -> String {
    let trimmed = next.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut normalized = trimmed.to_string();
    let ends_sentence = existing
        .chars()
        .rev()
        .find(|ch| !ch.is_whitespace())
        .map(|ch| matches!(ch, '.' | '!' | '?' | ':' | ';'))
        .unwrap_or(true);

    if !ends_sentence {
        lowercase_first_alpha(&mut normalized);
    }

    transcribe::append_deduped_chunk(existing, &normalized);
    normalized
}

fn lowercase_first_alpha(text: &mut String) {
    if let Some((idx, ch)) = text.char_indices().find(|(_, ch)| ch.is_alphabetic())
        && ch.is_uppercase()
    {
        let mut lowered = String::with_capacity(text.len());
        lowered.push_str(&text[..idx]);
        lowered.extend(ch.to_lowercase());
        lowered.push_str(&text[idx + ch.len_utf8()..]);
        *text = lowered;
    }
}

#[cfg(test)]
mod meeting_tests {
    use super::*;

    #[test]
    fn newly_transcribed_items_are_organized_automatically() {
        assert!(should_automatically_organize(None));
    }

    #[test]
    fn retranscription_keeps_existing_metadata() {
        assert!(!should_automatically_organize(Some("2026-08-23T16:30:00Z")));
    }

    fn result_with_segment(text: &str, start_ms: u64, end_ms: u64) -> LibraryTranscriptionResult {
        LibraryTranscriptionResult {
            transcript: text.to_string(),
            segments: Some(vec![TranscriptSegment {
                start_ms,
                end_ms,
                text: text.to_string(),
                speaker_id: None,
            }]),
            words: None,
            speech_model: None,
            speakers: None,
        }
    }

    #[test]
    fn meeting_merge_keeps_overlapping_sources_as_separate_segments() {
        let microphone = label_meeting_track(
            result_with_segment("I am speaking", 100, 1_500),
            MeetingTrackSource::Microphone,
            2_000,
            "You".to_string(),
        );
        let system = label_meeting_track(
            result_with_segment("So am I", 100, 1_400),
            MeetingTrackSource::System,
            2_000,
            "Meeting".to_string(),
        );

        let merged = merge_meeting_results(vec![microphone, system]);
        let segments = merged.segments.expect("meeting segments");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_ms, 100);
        assert_eq!(segments[1].start_ms, 100);
        assert_ne!(segments[0].speaker_id, segments[1].speaker_id);
        assert!(merged.transcript.contains("You: I am speaking"));
        assert!(merged.transcript.contains("Meeting: So am I"));
        assert_eq!(merged.speakers.expect("meeting speakers").len(), 2);
    }

    #[test]
    fn meeting_track_without_timestamps_gets_a_source_segment() {
        let result = LibraryTranscriptionResult {
            transcript: "Fallback transcript".to_string(),
            segments: None,
            words: None,
            speech_model: None,
            speakers: None,
        };
        let labeled = label_meeting_track(
            result,
            MeetingTrackSource::Microphone,
            3_000,
            "You".to_string(),
        );
        let segment = &labeled.segments.expect("fallback segment")[0];
        assert_eq!(segment.start_ms, 0);
        assert_eq!(segment.end_ms, 3_000);
        assert_eq!(segment.speaker_id.as_deref(), Some("meeting_you"));
    }

    #[test]
    fn person_detection_only_runs_on_the_meeting_audio_track() {
        assert!(!meeting_track_person_detection_enabled(
            true,
            MeetingTrackSource::Microphone,
        ));
        assert!(meeting_track_person_detection_enabled(
            true,
            MeetingTrackSource::System,
        ));
        assert!(!meeting_track_person_detection_enabled(
            false,
            MeetingTrackSource::System,
        ));
    }
}
