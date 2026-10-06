import { useLingui } from "@lingui/react/macro";
import { useCallback, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  FolderOpen,
  CircleNotch as Loader2,
  Plus,
  Record as RecordIcon,
  Stop,
  MagnifyingGlass as Search,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import DotMatrix from "../../../shared/ui/DotMatrix";
import ScreenHeader from "../../../shared/ui/ScreenHeader";
import { useDebouncedValue } from "../../../shared/hooks/useDebouncedValue";
import { useShiftHeld } from "../../../shared/hooks/useShiftHeld";
import { useModelDownloadEvents } from "../../../shared/hooks/useModelDownloadEvents";
import { useSettings } from "../../settings/queries";
import {
  modelKeys as settingsModelKeys,
  useSpeechModels,
} from "../../settings/models-queries";
import LibraryImportModal from "./LibraryImportModal";
import LibraryCard from "./LibraryCard";
import LibraryDetail from "./LibraryDetail";
import ActiveMeetingCard from "./ActiveMeetingCard";
import {
  useLibraryItems as useLibraryItemsQuery,
  useLibraryMetadataProcessing,
  useCreateLibraryItem,
  useUpdateLibraryItem,
  useGenerateLibraryItemTitle,
  useDeleteLibraryItem,
  useCancelLibraryTranscription,
  useRetryLibraryTranscription,
  useExportLibraryItem,
  useLibraryTags,
  libraryKeys,
  useMeetingState,
  useStartMeetingRecording,
  useStopMeetingRecording,
} from "../queries";
import {
  formatDeleteErrorMessage,
  formatImportErrorMessage,
  getFileExtension,
  SUPPORTED_EXTENSIONS,
  uniquePaths,
} from "./library-utils";
import { Dropdown } from "../../../shared/ui/Dropdown";
import type {
  LibraryFilter,
  LibraryItem,
  LibraryItemPatch,
} from "../../../types";

type LibraryViewProps = {
  pendingImportPaths: string[] | null;
  onSetImportPaths: (paths: string[] | null) => void;
  isActive: boolean;
  scope: "files" | "meetings";
};

type LibraryStatusFilter = "all" | "active" | "complete" | "error";

const LibraryView = ({
  pendingImportPaths,
  onSetImportPaths,
  isActive,
  scope,
}: LibraryViewProps) => {
  const { t } = useLingui();
  const queryClient = useQueryClient();

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<LibraryStatusFilter>("all");
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [editingNameDraft, setEditingNameDraft] = useState("");
  const [editingTagId, setEditingTagId] = useState<string | null>(null);
  const [tagDraft, setTagDraft] = useState("");
  const [followTimestamps, setFollowTimestamps] = useState(true);
  const shiftHeld = useShiftHeld(isActive);
  const { data: meetingState } = useMeetingState(
    isActive && scope === "meetings",
  );
  const startMeetingMutation = useStartMeetingRecording();
  const stopMeetingMutation = useStopMeetingRecording();
  const debouncedSearchQuery = useDebouncedValue(searchQuery, 300);
  const filter = useMemo<LibraryFilter>(() => {
    return {
      search: debouncedSearchQuery || null,
      status: statusFilter === "all" ? null : statusFilter,
      kind: scope,
      tag: null,
      since_days: null,
    };
  }, [debouncedSearchQuery, scope, statusFilter]);

  const {
    data,
    isLoading,
    isFetchingNextPage,
    hasNextPage,
    fetchNextPage,
    error: queryError,
  } = useLibraryItemsQuery(filter, isActive);

  const { data: availableTags = [] } = useLibraryTags(isActive);
  const automaticallyOrganizingIds = useLibraryMetadataProcessing(isActive);
  const { data: speechModels = [] } = useSpeechModels(isActive);
  const { data: defaultModelKey = "" } = useSettings(
    (settings) => settings.local_model,
    isActive,
  );

  const items = useMemo(
    () => data?.pages.flatMap((page) => page.items) ?? [],
    [data],
  );
  const selectedItem = useMemo(
    () => items.find((item) => item.id === selectedItemId) ?? null,
    [items, selectedItemId],
  );
  const error = queryError
    ? queryError instanceof Error
      ? queryError.message
      : String(queryError)
    : null;

  const createItemMutation = useCreateLibraryItem();
  const updateItemMutation = useUpdateLibraryItem();
  const generateTitleMutation = useGenerateLibraryItemTitle();
  const deleteItemMutation = useDeleteLibraryItem();
  const cancelMutation = useCancelLibraryTranscription();
  const retryMutation = useRetryLibraryTranscription();
  const exportMutation = useExportLibraryItem();

  const invalidateTags = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: libraryKeys.tags() });
  }, [queryClient]);

  const updateItemWithTags = useCallback(
    async (id: string, patch: LibraryItemPatch) => {
      const updated = await updateItemMutation.mutateAsync({ id, patch });
      if (patch.tags != null) invalidateTags();
      return updated;
    },
    [updateItemMutation, invalidateTags],
  );

  const deleteItemAndRefreshTags = useCallback(
    async (id: string) => {
      try {
        await deleteItemMutation.mutateAsync(id);
        invalidateTags();
      } catch (err) {
        console.error("Failed to delete library item:", err);
        const message = err instanceof Error ? err.message : String(err);
        invoke("debug_show_toast", {
          toastType: "error",
          message: formatDeleteErrorMessage(message),
        }).catch(() => {});
        throw err;
      }
    },
    [deleteItemMutation, invalidateTags],
  );

  const generateItemTitle = useCallback(
    async (id: string) => {
      try {
        await generateTitleMutation.mutateAsync(id);
      } catch (err) {
        console.error("Failed to generate library item title and tags:", err);
        const code = err instanceof Error ? err.message : String(err);
        const message = code.includes("title_model_not_configured")
          ? t({
              id: "library.card.title.error.not_configured",
              message: "Configure a writing model before generating a title.",
            })
          : code.includes("title_rate_limited")
            ? t({
                id: "library.card.title.error.rate_limited",
                message:
                  "The writing provider has reached its rate or usage limit. Try again later.",
              })
            : code.includes("title_unauthorized")
              ? t({
                  id: "library.card.title.error.unauthorized",
                  message:
                    "The writing provider rejected its API key. Check it in Settings.",
                })
              : code.includes("title_model_not_found")
                ? t({
                    id: "library.card.title.error.not_found",
                    message:
                      "The configured writing model could not be found. Check it in Settings.",
                  })
                : code.includes("title_request_rejected")
                  ? t({
                      id: "library.card.title.error.rejected",
                      message:
                        "The writing provider rejected the title request.",
                    })
                  : code.includes("title_unreachable")
                    ? t({
                        id: "library.card.title.error.unreachable",
                        message:
                          "Could not reach the writing provider. Check your connection and try again.",
                      })
                    : code.includes("title_invalid_response")
                      ? t({
                          id: "library.card.title.error.invalid_response",
                          message:
                            "The writing model responded, but could not organize this item. Try again.",
                        })
                      : t({
                          id: "library.card.title.error",
                          message:
                            "Could not generate a title and tags. Try again.",
                        });
        invoke("debug_show_toast", {
          toastType: "error",
          message,
        }).catch(() => {});
        throw err;
      }
    },
    [generateTitleMutation, t],
  );

  const installedModels = useMemo(
    () => speechModels.filter((model) => model.installed),
    [speechModels],
  );

  const refreshSpeechModels = useCallback(() => {
    void queryClient.invalidateQueries({
      queryKey: settingsModelKeys.speech(),
    });
  }, [queryClient]);

  useModelDownloadEvents({
    enabled: isActive,
    onComplete: refreshSpeechModels,
    onError: refreshSpeechModels,
  });

  const handleImportClick = async () => {
    try {
      const selection = await open({
        multiple: true,
        filters: [
          {
            name: t({
              id: "library.view.file_filter",
              message: "Audio & Video",
            }),
            extensions: SUPPORTED_EXTENSIONS,
          },
        ],
      });

      if (!selection) return;

      const paths = Array.isArray(selection) ? selection : [selection];
      if (paths.length > 0) {
        onSetImportPaths(uniquePaths(paths));
      }
    } catch (err) {
      console.error("Failed to open import dialog:", err);
      invoke("debug_show_toast", {
        toastType: "error",
        message: t({
          id: "library.view.import_dialog_error",
          message: "Could not open the import dialog.",
        }),
      }).catch(() => {});
    }
  };

  const startTagEdit = (item: LibraryItem) => {
    setEditingTagId(item.id);
    setTagDraft("");
  };

  const startNameEdit = (item: LibraryItem) => {
    setEditingNameId(item.id);
    setEditingNameDraft(item.name);
  };

  const cancelNameEdit = () => {
    setEditingNameId(null);
    setEditingNameDraft("");
  };

  const commitNameEdit = async (itemId: string) => {
    const nextName = editingNameDraft.trim();
    const original = items.find((entry) => entry.id === itemId)?.name ?? "";
    setEditingNameId(null);
    setEditingNameDraft("");
    if (!nextName || nextName === original) return;
    await updateItemWithTags(itemId, { name: nextName });
  };

  const cancelTagEdit = () => {
    setEditingTagId(null);
    setTagDraft("");
  };

  const commitTagAdd = async (itemId: string, overrideTag?: string) => {
    const nextTag = (overrideTag ?? tagDraft).trim();
    if (!nextTag) {
      setEditingTagId(null);
      setTagDraft("");
      return;
    }
    const item = items.find((entry) => entry.id === itemId);
    if (!item) return;
    if (item.tags.some((tag) => tag.toLowerCase() === nextTag.toLowerCase())) {
      setTagDraft("");
      setEditingTagId(null);
      return;
    }
    await updateItemWithTags(itemId, { tags: [...item.tags, nextTag] });
    setTagDraft("");
    setEditingTagId(null);
  };

  const defaultSpeechModelKey =
    installedModels.find((model) => model.remote)?.id ??
    installedModels.find((model) => model.key === defaultModelKey)?.id ??
    installedModels[0]?.id;
  const meetingBusy =
    startMeetingMutation.isPending || stopMeetingMutation.isPending;
  const handleMeetingClick = async () => {
    try {
      if (meetingState?.recording) {
        await stopMeetingMutation.mutateAsync();
        return;
      }
      if (!defaultSpeechModelKey) {
        invoke("debug_show_toast", {
          toastType: "error",
          message: t({
            id: "library.meeting.model_required",
            message: "Install or configure a speech model first.",
          }),
        }).catch(() => {});
        return;
      }
      await startMeetingMutation.mutateAsync({
        store_original: false,
        model_key: defaultSpeechModelKey,
        llm_cleanup_enabled: false,
        show_timestamps: true,
        detect_speakers: true,
      });
    } catch (err) {
      console.error("Meeting recording failed:", err);
      invoke("debug_show_toast", {
        toastType: "error",
        message: t({
          id: "library.meeting.error",
          message: "Meeting recording failed. Check permissions and try again.",
        }),
      }).catch(() => {});
    }
  };
  const statusFilterOptions = useMemo(
    () => [
      {
        value: "all" as const,
        label: t({ id: "library.filter.all", message: "All" }),
      },
      {
        value: "active" as const,
        label: t({ id: "library.filter.active", message: "Active" }),
      },
      {
        value: "complete" as const,
        label: t({ id: "library.filter.done", message: "Done" }),
      },
      {
        value: "error" as const,
        label: t({ id: "library.filter.failed", message: "Failed" }),
      },
    ],
    [t],
  );
  const headerAction =
    scope === "files" ? (
      <button
        type="button"
        onClick={handleImportClick}
        className="inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg bg-content-primary px-3.5 py-1.5 text-sm leading-5 font-semibold text-surface-secondary transition-all hover:bg-content-secondary shadow-[0_3px_0_-1px_rgba(255,255,255,0.25),inset_0_1px_0_0_rgba(255,255,255,0.1)] active:translate-y-[1px] active:shadow-none"
      >
        <Plus size={14} aria-hidden="true" />
        {t({ id: "library.view.import_button", message: "Import" })}
      </button>
    ) : (
      <button
        type="button"
        onClick={handleMeetingClick}
        disabled={
          meetingBusy || (!meetingState?.recording && !defaultSpeechModelKey)
        }
        className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg px-3.5 py-1.5 text-sm leading-5 font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-50 active:translate-y-[1px] active:shadow-none ${
          meetingState?.recording
            ? "bg-[var(--color-error)] text-white hover:opacity-90 shadow-[0_3px_0_-1px_rgba(255,255,255,0.2)]"
            : "bg-content-primary text-surface-secondary hover:bg-content-secondary shadow-[0_3px_0_-1px_rgba(255,255,255,0.25),inset_0_1px_0_0_rgba(255,255,255,0.1)]"
        }`}
      >
        {meetingBusy ? (
          <Loader2 size={14} className="animate-spin" aria-hidden="true" />
        ) : meetingState?.recording ? (
          <Stop size={14} weight="fill" aria-hidden="true" />
        ) : (
          <RecordIcon size={14} weight="fill" aria-hidden="true" />
        )}
        {meetingState?.recording
          ? t({ id: "library.meeting.stop", message: "Stop recording" })
          : t({ id: "library.meeting.start", message: "Record meeting" })}
      </button>
    );
  return (
    <div className="relative flex h-full min-h-0 min-w-0 flex-1 flex-col">
      {selectedItem ? (
        <motion.div
          key={selectedItem.id || "selected-library-item"}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18, ease: "easeOut" }}
          className="flex h-full min-h-0 flex-col"
        >
          <LibraryDetail
            item={selectedItem}
            models={installedModels}
            followTimestamps={followTimestamps}
            onFollowTimestampsChange={setFollowTimestamps}
            onClose={() => setSelectedItemId(null)}
            onDelete={async () => {
              await deleteItemAndRefreshTags(selectedItem.id);
              setSelectedItemId(null);
            }}
            onRetry={() => retryMutation.mutateAsync(selectedItem.id)}
            onCancel={() => cancelMutation.mutateAsync(selectedItem.id)}
            onUpdate={(patch) => updateItemWithTags(selectedItem.id, patch)}
            onExport={(format, outputPath) =>
              exportMutation.mutateAsync({
                id: selectedItem.id,
                format,
                outputPath,
              })
            }
            availableTags={availableTags}
            onGenerateTitle={() => generateItemTitle(selectedItem.id)}
            isGeneratingTitle={
              automaticallyOrganizingIds.has(selectedItem.id) ||
              (generateTitleMutation.isPending &&
                generateTitleMutation.variables === selectedItem.id)
            }
            backLabel={
              scope === "meetings"
                ? t({
                    id: "meeting.detail.back",
                    message: "Back to meetings",
                  })
                : t({
                    id: "library.detail.back",
                    message: "Back to library",
                  })
            }
          />
        </motion.div>
      ) : (
        <>
          <div className="mx-auto flex w-full max-w-7xl min-w-0 flex-col gap-4 pt-8 pb-4 px-0 text-left">
            <div className="flex flex-col gap-4 mb-4 mt-2 md:-mt-6">
              <ScreenHeader
                icon={
                  <DotMatrix
                    rows={2}
                    cols={3}
                    activeDots={[0, 1, 2, 4]}
                    dotSize={3}
                    gap={3}
                    color="var(--color-section-marker-alt)"
                  />
                }
                title={
                  scope === "meetings"
                    ? t({ id: "meeting.view.title", message: "Meetings" })
                    : t({ id: "library.view.title", message: "Library" })
                }
                description={
                  scope === "meetings"
                    ? t({
                        id: "meeting.view.description",
                        message: "Record and transcribe online meetings.",
                      })
                    : t({
                        id: "library.view.description",
                        message:
                          "Import audio and video files for transcription.",
                      })
                }
                trailing={headerAction}
              />

              <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(14rem,1fr)_auto] md:items-center">
                <div className="relative min-w-0 w-full group">
                  <Search
                    size={14}
                    className="absolute left-3 top-1/2 -translate-y-1/2 ui-color-muted transition-colors"
                  />
                  <input
                    type="text"
                    placeholder={
                      scope === "meetings"
                        ? t({
                            id: "meeting.view.search_placeholder",
                            message: "Search meetings...",
                          })
                        : t({
                            id: "library.view.search_placeholder",
                            message: "Search library...",
                          })
                    }
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-lg focus:border-[var(--color-border-hover)] pl-9 pr-4 py-1.5 ui-text-input ui-color-primary placeholder-[var(--color-text-muted)] outline-none transition-all duration-100 ease-out"
                  />
                </div>

                <Dropdown
                  value={statusFilter}
                  options={statusFilterOptions}
                  onChange={setStatusFilter}
                  fitButtonToWidestOption
                  className="w-full md:w-auto"
                  buttonClassName="px-3 py-1.5 ui-text-body-sm"
                  menuClassName="min-w-36 md:left-auto md:right-0"
                />
              </div>
            </div>

            {error && (
              <div className="rounded-lg border border-[var(--color-error)]/30 bg-[var(--color-error)]/10 px-4 py-3 ui-text-body-sm ui-color-error-tint mx-4 mb-2">
                {error}
              </div>
            )}
          </div>
          <div className="flex-1 min-h-0 overflow-y-scroll overflow-x-hidden custom-scrollbar scrollbar-gutter pb-6 pr-3 pt-1">
            <div key="library-list" className="flex flex-col gap-6 w-full">
              <div className="mx-auto flex w-full max-w-6xl min-w-0 flex-col gap-6">
                {scope === "meetings" && meetingState?.recording && (
                  <ActiveMeetingCard
                    meeting={meetingState}
                    stopping={stopMeetingMutation.isPending}
                    onStop={() => void handleMeetingClick()}
                  />
                )}
                <div className="grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,216px),1fr))] gap-4">
                  {isLoading && items.length === 0 && (
                    <div className="col-span-full py-12 flex items-center justify-center">
                      <DotMatrix
                        rows={2}
                        cols={8}
                        activeDots={[0, 1, 2, 3, 4, 5, 6, 7]}
                        dotSize={3}
                        gap={3}
                        color="var(--color-text-muted)"
                        animated
                        className="opacity-50"
                      />
                    </div>
                  )}

                  {!isLoading &&
                    items.length === 0 &&
                    !(scope === "meetings" && meetingState?.recording) && (
                      <button
                        type="button"
                        onClick={
                          scope === "meetings"
                            ? handleMeetingClick
                            : handleImportClick
                        }
                        className="col-span-full rounded-xl border border-dashed border-border-secondary bg-surface-secondary p-8 flex flex-col items-center justify-center text-center hover:text-content-secondary hover:border-border-hover transition-colors"
                      >
                        {scope === "meetings" ? (
                          <RecordIcon
                            size={20}
                            className="text-content-disabled"
                            weight="fill"
                          />
                        ) : (
                          <FolderOpen
                            size={20}
                            className="text-content-disabled"
                          />
                        )}
                        <p className="mt-3 ui-text-body ui-color-muted">
                          {scope === "meetings"
                            ? t({
                                id: "meeting.view.empty_state",
                                message: "Record your first meeting.",
                              })
                            : t({
                                id: "library.view.empty_state",
                                message:
                                  "Drag files here to build your Library.",
                              })}
                        </p>
                      </button>
                    )}

                  {items.map((item, index) => (
                    <LibraryCard
                      key={item.id || `library-item-${index}`}
                      item={item}
                      onOpen={() => setSelectedItemId(item.id)}
                      onRemoveTag={async (tag) => {
                        const nextTags = item.tags.filter(
                          (entry) => entry !== tag,
                        );
                        await updateItemWithTags(item.id, { tags: nextTags });
                      }}
                      onClickTag={(tag) => setSearchQuery(`#${tag}`)}
                      editingNameId={editingNameId}
                      editingNameDraft={editingNameDraft}
                      onStartNameEdit={() => startNameEdit(item)}
                      onChangeNameDraft={setEditingNameDraft}
                      onCommitNameEdit={() => commitNameEdit(item.id)}
                      onCancelNameEdit={cancelNameEdit}
                      onRetry={() => retryMutation.mutateAsync(item.id)}
                      onCancel={() => cancelMutation.mutateAsync(item.id)}
                      onDelete={async () => {
                        await deleteItemAndRefreshTags(item.id);
                      }}
                      onGenerateTitle={() => generateItemTitle(item.id)}
                      isGeneratingTitle={
                        automaticallyOrganizingIds.has(item.id) ||
                        (generateTitleMutation.isPending &&
                          generateTitleMutation.variables === item.id)
                      }
                      editingTagId={editingTagId}
                      tagDraft={tagDraft}
                      onStartTagEdit={() => startTagEdit(item)}
                      onChangeTagDraft={setTagDraft}
                      onCommitTagAdd={(value) => commitTagAdd(item.id, value)}
                      onCancelTagEdit={cancelTagEdit}
                      shiftHeld={shiftHeld}
                      availableTags={availableTags}
                    />
                  ))}

                  {scope === "files" && items.length > 0 && (
                    <button
                      onClick={handleImportClick}
                      className="rounded-xl border border-dashed border-border-secondary bg-surface-secondary p-4 flex flex-col items-center justify-center text-center ui-color-muted hover:text-content-secondary hover:border-border-hover transition-colors"
                    >
                      <FolderOpen size={18} />
                      <span className="mt-2 ui-text-body-sm">
                        {t({
                          id: "library.view.dropzone",
                          message: "Drop files to import",
                        })}
                      </span>
                    </button>
                  )}

                  {items.length > 0 && hasNextPage && (
                    <div className="col-span-full flex items-center justify-center pt-2">
                      <button
                        onClick={() => fetchNextPage()}
                        disabled={isFetchingNextPage}
                        className="flex items-center gap-2 rounded-lg border border-border-primary bg-surface-surface px-4 py-2 ui-text-body-sm ui-color-secondary hover:text-content-primary hover:border-border-secondary hover:bg-surface-overlay transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        {isFetchingNextPage ? (
                          <>
                            <Loader2 size={14} className="animate-spin" />
                            <span>
                              {t({
                                id: "library.view.loading_more",
                                message: "Loading...",
                              })}
                            </span>
                          </>
                        ) : (
                          <span>
                            {t({
                              id: "library.view.load_more",
                              message: "Load more",
                            })}
                          </span>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <AnimatePresence>
        {scope === "files" && pendingImportPaths !== null && (
          <LibraryImportModal
            paths={pendingImportPaths}
            models={installedModels}
            defaultModelKey={defaultSpeechModelKey}
            onCancel={() => onSetImportPaths(null)}
            onConfirm={async (paths, options) => {
              const supported = paths.filter((path) =>
                SUPPORTED_EXTENSIONS.includes(getFileExtension(path)),
              );
              const unsupported = paths.filter(
                (path) =>
                  !SUPPORTED_EXTENSIONS.includes(getFileExtension(path)),
              );

              if (unsupported.length > 0) {
                invoke("debug_show_toast", {
                  toastType: "warning",
                  message: t({
                    id: "library.view.unsupported_files",
                    message: `${unsupported.length} file(s) skipped due to unsupported format.`,
                  }),
                }).catch(() => {});
              }

              for (const path of supported) {
                try {
                  await createItemMutation.mutateAsync({ path, options });
                } catch (err) {
                  console.error("Failed to import file:", err);
                  const message =
                    err instanceof Error ? err.message : String(err);
                  const toastMessage = formatImportErrorMessage(message);
                  invoke("debug_show_toast", {
                    toastType: "error",
                    message: toastMessage,
                  }).catch(() => {});
                }
              }

              onSetImportPaths(null);
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default LibraryView;
