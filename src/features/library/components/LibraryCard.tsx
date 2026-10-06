import { useLingui } from "@lingui/react/macro";
import { useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  WarningCircle as AlertCircle,
  CaretDown as ChevronDown,
  DotsThree as MoreHorizontal,
  PencilSimple as Pencil,
  Sparkle,
  CircleNotch as Loader2,
  ArrowClockwise as RotateCw,
  Trash as Trash2,
  X,
} from "@phosphor-icons/react";
import {
  clampProgress,
  formatDuration,
  getLibraryErrorDetails,
  shouldShowImportProgress,
  formatLibraryName,
  formatLibraryCardDate,
} from "./library-utils";
import { formatBytes } from "../../../shared/lib/format";
import { useClickOutside } from "../../../shared/hooks/useClickOutside";
import { IntelligencePixel } from "../../../shared/ui/IntelligencePixel";
import FloatingPortal from "../../../shared/ui/FloatingPortal";
import type { LibraryItem } from "../../../types";

const LibraryCard = ({
  item,
  onOpen,
  onRemoveTag,
  onClickTag,
  editingNameId,
  editingNameDraft,
  onStartNameEdit,
  onChangeNameDraft,
  onCommitNameEdit,
  onCancelNameEdit,
  onRetry,
  onCancel,
  onDelete,
  onGenerateTitle,
  isGeneratingTitle,
  editingTagId,
  tagDraft,
  onStartTagEdit,
  onChangeTagDraft,
  onCommitTagAdd,
  onCancelTagEdit,
  shiftHeld,
  availableTags,
}: {
  item: LibraryItem;
  onOpen: () => void;
  onRemoveTag: (tag: string) => Promise<void>;
  onClickTag?: (tag: string) => void;
  editingNameId: string | null;
  editingNameDraft: string;
  onStartNameEdit: () => void;
  onChangeNameDraft: (value: string) => void;
  onCommitNameEdit: () => void;
  onCancelNameEdit: () => void;
  onRetry: () => Promise<void>;
  onCancel: () => Promise<void>;
  onDelete: () => Promise<void>;
  onGenerateTitle: () => Promise<void>;
  isGeneratingTitle: boolean;
  editingTagId: string | null;
  tagDraft: string;
  onStartTagEdit: () => void;
  onChangeTagDraft: (value: string) => void;
  onCommitTagAdd: (value?: string) => void;
  onCancelTagEdit: () => void;
  shiftHeld: boolean;
  availableTags: string[];
}) => {
  const { t } = useLingui();
  const status = item.status;

  const showImportProgress =
    status.type === "importing" && shouldShowImportProgress(status.progress);
  const isTranscribing = status.type === "transcribing" || showImportProgress;
  const isError = status.type === "error";

  const isProcessing = isTranscribing || isGeneratingTitle;
  const progress = isTranscribing ? clampProgress(status.progress) : 0;

  const isEditingName = editingNameId === item.id;
  const isAddingTag = editingTagId === item.id;
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuPopupRef = useRef<HTMLDivElement>(null);
  const [tagMenuOpen, setTagMenuOpen] = useState(false);
  const tagMenuRef = useRef<HTMLDivElement>(null);
  const tagPopupRef = useRef<HTMLDivElement>(null);
  const errorTooltipRef = useRef<HTMLDivElement>(null);
  const [errorTooltipOpen, setErrorTooltipOpen] = useState(false);
  const errorDetails =
    status.type === "error" ? getLibraryErrorDetails(status.message) : null;
  const displayName = formatLibraryName(item.name);
  const createdAtLabel = formatLibraryCardDate(item.created_at);
  const recoveredMeeting =
    item.kind === "recovered_meeting" ||
    (item.kind === "meeting" &&
      item.tags.some((tag) => tag.toLowerCase() === "recovered"));
  const visibleTags = recoveredMeeting
    ? item.tags.filter((tag) => tag.toLowerCase() !== "recovered")
    : item.tags;
  const statusLabel = isGeneratingTitle
    ? t({
        id: "library.card.title.generating",
        message: "Organizing...",
      })
    : status.type === "complete"
      ? null
      : status.type === "transcribing"
        ? t({
            id: "library.card.status.thinking",
            message: `Thinking ${(progress * 100).toFixed(0)}%`,
          })
        : status.type === "importing" && showImportProgress
          ? t({
              id: "library.card.status.converting",
              message: `Converting ${(progress * 100).toFixed(0)}%`,
            })
          : status.type === "error"
            ? t({ id: "library.card.status.failed", message: "Failed" })
            : status.type === "cancelling"
              ? t({
                  id: "library.card.status.cancelling",
                  message: "Cancelling",
                })
              : status.type === "cancelled"
                ? t({
                    id: "library.card.status.cancelled",
                    message: "Cancelled",
                  })
                : t({ id: "library.card.status.queued", message: "Queued" });

  const normalizedDraft = tagDraft.trim().toLowerCase();
  const filteredTagOptions = availableTags.filter((tag) => {
    const tagLower = tag.toLowerCase();
    if (item.tags.some((existing) => existing.toLowerCase() === tagLower)) {
      return false;
    }
    if (!normalizedDraft) return true;
    return tagLower.includes(normalizedDraft);
  });

  useClickOutside(menuRef, () => setMenuOpen(false), menuOpen, [menuPopupRef]);
  useClickOutside(tagMenuRef, () => setTagMenuOpen(false), tagMenuOpen, [
    tagPopupRef,
  ]);

  const handleDelete = async () => {
    setMenuOpen(false);
    try {
      await onDelete();
    } catch (err) {
      console.error("Failed to delete library item:", err);
    }
  };

  const handleRetry = async () => {
    setMenuOpen(false);
    try {
      await onRetry();
    } catch (err) {
      console.error("Failed to retry library transcription:", err);
    }
  };

  const handleCancel = async () => {
    setMenuOpen(false);
    try {
      await onCancel();
    } catch (err) {
      console.error("Failed to cancel library transcription:", err);
    }
  };

  const handleGenerateTitle = async () => {
    setMenuOpen(false);
    try {
      await onGenerateTitle();
    } catch {
      // The owner displays the localized error toast.
    }
  };

  return (
    <div
      onClick={() => {
        if (!isEditingName && !isAddingTag) {
          onOpen();
        }
      }}
      onContextMenu={(event) => {
        event.preventDefault();
        if (shiftHeld) {
          void handleDelete();
        } else {
          setMenuOpen(true);
        }
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          if (!isEditingName && !isAddingTag) {
            onOpen();
          }
        }
      }}
      role="button"
      tabIndex={0}
      className={`ui-card-liftable group relative z-0 flex h-[236px] min-w-0 flex-col outline-none hover:z-10 ${
        shiftHeld
          ? "!border-[var(--color-error)]/30 hover:!border-[var(--color-error)]/60 !bg-[var(--color-error)]/5"
          : ""
      }`}
    >
      <div className="relative flex h-full w-full min-w-0 flex-col px-4 py-2.5">
        <div className="mb-2.5 flex h-6 shrink-0 items-center justify-between gap-3">
          <div className="flex h-6 min-w-0 items-center gap-2">
            <IntelligencePixel
              active={isProcessing}
              statusType={item.status.type}
            />
            {createdAtLabel && (
              <span className="min-w-0 truncate ui-text-label ui-color-muted">
                {createdAtLabel}
              </span>
            )}
          </div>

          <div className="-mr-1 flex h-6 items-center overflow-visible">
            <div
              ref={menuRef}
              data-no-press
              className="flex relative items-center justify-center"
            >
              <button
                data-no-press
                onPointerDown={(e) => {
                  e.stopPropagation();
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  if (shiftHeld) {
                    handleDelete();
                  } else {
                    setMenuOpen((prev) => !prev);
                  }
                }}
                onKeyDown={(e) => {
                  e.stopPropagation();
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    if (e.repeat) return;
                    if (shiftHeld) {
                      handleDelete();
                    } else {
                      setMenuOpen((prev) => !prev);
                    }
                  }
                }}
                onKeyUp={(e) => e.stopPropagation()}
                className={`ml-1 flex h-6 w-6 items-center justify-center rounded p-1 outline-none transition-colors duration-200 focus-visible:ring-1 focus-visible:ring-[var(--color-border-hover)] ${
                  shiftHeld
                    ? "ui-color-error hover:bg-[var(--color-error)]/10"
                    : menuOpen
                      ? "ui-color-primary bg-[var(--color-bg-elevated)]"
                      : "ui-color-muted hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-elevated)]"
                }`}
                aria-label={t({
                  id: "library.card.more_options",
                  message: "More options",
                })}
              >
                {shiftHeld ? (
                  <Trash2 size={14} className="shrink-0 transform-gpu" />
                ) : (
                  <MoreHorizontal
                    size={14}
                    className="shrink-0 transform-gpu"
                  />
                )}
              </button>
              {menuOpen && (
                <FloatingPortal
                  anchorRef={menuRef}
                  ref={menuPopupRef}
                  placement="bottom-end"
                  data-no-press
                  className="min-w-[160px] rounded-lg border border-[var(--color-border-secondary)] bg-[var(--color-bg-overlay)] shadow-xl shadow-[var(--color-shadow-soft-50)] overflow-hidden"
                  onClick={(event) => event.stopPropagation()}
                >
                  <button
                    onClick={() => {
                      setMenuOpen(false);
                      onStartNameEdit();
                    }}
                    className="flex w-full items-center gap-2.5 px-3 py-2 ui-text-menu-item ui-color-secondary hover:bg-[var(--color-bg-elevated)] transition-colors"
                  >
                    <Pencil size={12} className="ui-color-muted" />
                    <span>
                      {t({ id: "library.card.rename", message: "Rename" })}
                    </span>
                  </button>

                  {status.type === "complete" && item.transcript?.trim() && (
                    <button
                      onClick={handleGenerateTitle}
                      disabled={isGeneratingTitle}
                      className="flex w-full items-center gap-2.5 px-3 py-2 ui-text-menu-item ui-color-secondary transition-colors hover:bg-[var(--color-bg-elevated)] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {isGeneratingTitle ? (
                        <Loader2
                          size={12}
                          className="animate-spin ui-color-accent"
                        />
                      ) : (
                        <Sparkle size={12} className="ui-color-accent" />
                      )}
                      <span>
                        {isGeneratingTitle
                          ? t({
                              id: "library.card.title.generating",
                              message: "Organizing...",
                            })
                          : t({
                              id: "library.card.title.generate",
                              message: "Generate title and tags",
                            })}
                      </span>
                    </button>
                  )}

                  {status.type === "transcribing" ||
                  status.type === "cancelling" ||
                  status.type === "pending" ||
                  status.type === "importing" ? (
                    <button
                      onClick={handleCancel}
                      className="flex w-full items-center gap-2.5 px-3 py-2 ui-text-menu-item ui-color-secondary hover:bg-[var(--color-bg-elevated)] transition-colors"
                    >
                      <X size={12} className="ui-color-warning" />
                      <span>
                        {t({ id: "library.card.cancel", message: "Cancel" })}
                      </span>
                    </button>
                  ) : (
                    <button
                      onClick={handleRetry}
                      className="flex w-full items-center gap-2.5 px-3 py-2 ui-text-menu-item ui-color-secondary hover:bg-[var(--color-bg-elevated)] transition-colors"
                    >
                      <RotateCw size={12} className="ui-color-cloud" />
                      <span>
                        {status.type === "error"
                          ? t({
                              id: "library.card.retry",
                              message: "Retry",
                            })
                          : t({
                              id: "library.card.retranscribe",
                              message: "Retranscribe",
                            })}
                      </span>
                    </button>
                  )}

                  <div className="h-px bg-[var(--color-border-secondary)] mx-2 my-1" />

                  <button
                    onClick={handleDelete}
                    className="flex w-full items-center gap-2.5 px-3 py-2 ui-text-menu-item ui-color-error-strong hover:bg-[var(--color-error)]/10 transition-colors"
                  >
                    <Trash2 size={12} />
                    <span>
                      {t({ id: "library.card.delete", message: "Delete" })}
                    </span>
                  </button>
                </FloatingPortal>
              )}
            </div>
          </div>
        </div>

        <div className="relative flex min-h-0 w-full flex-1 flex-col justify-start">
          {isEditingName ? (
            <input
              value={editingNameDraft}
              onChange={(event) => onChangeNameDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  onCommitNameEdit();
                }
                if (event.key === "Escape") {
                  event.preventDefault();
                  onCancelNameEdit();
                }
              }}
              onBlur={onCommitNameEdit}
              onClick={(event) => event.stopPropagation()}
              className="w-full min-w-0 border-0 border-b border-[var(--color-border-primary)] bg-transparent p-0 ui-text-title-lg font-medium leading-snug ui-color-primary outline-hidden focus:border-[var(--color-border-hover)]"
              autoFocus
            />
          ) : (
            <h3
              className="line-clamp-4 break-words [overflow-wrap:anywhere] ui-text-title-lg font-medium leading-snug ui-color-primary"
              title={displayName}
            >
              {displayName}
            </h3>
          )}

          {recoveredMeeting && (
            <div className="mt-1 flex min-w-0 items-center gap-1.5 ui-text-micro font-medium ui-color-warning-strong">
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--color-warning)]"
                aria-hidden="true"
              />
              <span className="truncate">
                {t({
                  id: "library.card.recovered",
                  message: "Recovered",
                })}
              </span>
            </div>
          )}

          {statusLabel && (
            <div className="mt-1.5 flex flex-col items-start gap-1">
              <div className="flex max-w-full min-w-0 items-center gap-1.5">
                <span
                  className={`min-w-0 truncate ui-text-label-strong ${
                    isError
                      ? "ui-color-error-strong font-semibold"
                      : isProcessing
                        ? "ui-color-accent font-semibold"
                        : "ui-color-muted"
                  }`}
                >
                  {statusLabel}
                </span>
                {isError && errorDetails && (
                  <div
                    ref={errorTooltipRef}
                    className="relative flex items-center cursor-default min-w-0"
                    onClick={(event) => event.stopPropagation()}
                    onMouseEnter={() => setErrorTooltipOpen(true)}
                    onMouseLeave={() => setErrorTooltipOpen(false)}
                  >
                    <AlertCircle size={12} className="ui-color-error-strong" />
                    {errorTooltipOpen && (
                      <FloatingPortal
                        anchorRef={errorTooltipRef}
                        placement="bottom-end"
                        offset={8}
                        className="pointer-events-none w-56 rounded-lg border border-[var(--color-border-hover)] bg-[var(--color-bg-overlay)] p-3 shadow-xl"
                        role="tooltip"
                      >
                        <p className="ui-text-body-sm ui-color-primary normal-case tracking-normal">
                          {errorDetails.message}
                        </p>
                      </FloatingPortal>
                    )}
                  </div>
                )}
              </div>
              {isProcessing && (
                <div className="w-16 h-[2px] bg-[var(--color-border-hover)] rounded-full overflow-hidden flex">
                  {isGeneratingTitle ? (
                    <motion.div
                      className="h-full w-5 shrink-0 bg-[var(--color-accent)]"
                      initial={{ x: -20 }}
                      animate={{ x: 64 }}
                      transition={{
                        ease: "linear",
                        duration: 0.9,
                        repeat: Infinity,
                      }}
                    />
                  ) : (
                    <motion.div
                      className="h-full bg-[var(--color-accent)]"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress * 100}%` }}
                      transition={{ ease: "linear", duration: 0.5 }}
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="mt-auto flex shrink-0 flex-col gap-2">
          <div className="relative min-h-7 w-full overflow-visible">
            {isAddingTag ? (
              <div
                className="flex h-7 items-center gap-1.5"
                onClick={(event) => event.stopPropagation()}
              >
                <div ref={tagMenuRef} className="relative flex items-center">
                  <button
                    type="button"
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => setTagMenuOpen((prev) => !prev)}
                    className="flex items-center justify-center w-[16px] h-[16px] shrink-0 ui-color-primary hover:text-[var(--color-text-secondary)] transition-colors"
                    aria-label={t({
                      id: "library.card.select_existing_tag",
                      message: "Select existing tag",
                    })}
                    title={t({
                      id: "library.card.select_existing_tag",
                      message: "Select existing tag",
                    })}
                  >
                    <ChevronDown
                      size={12}
                      className={`translate-y-[1px] transition-transform duration-150 ${tagMenuOpen ? "rotate-180" : ""}`}
                    />
                  </button>
                  {tagMenuOpen && (
                    <FloatingPortal
                      anchorRef={tagMenuRef}
                      ref={tagPopupRef}
                      placement="bottom-start"
                      className="w-36 rounded-md border border-[var(--color-border-secondary)] bg-[var(--color-bg-overlay)] shadow-lg shadow-[var(--color-shadow-soft-40)] overflow-hidden"
                    >
                      <div className="max-h-36 overflow-y-auto custom-scrollbar">
                        {filteredTagOptions.length > 0 ? (
                          filteredTagOptions.map((tag, index) => (
                            <button
                              key={`tag-option-${index}-${tag || "empty"}`}
                              type="button"
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={() => {
                                onCommitTagAdd(tag);
                                setTagMenuOpen(false);
                              }}
                              className="w-full truncate px-2.5 py-1.5 text-left ui-text-button-sm ui-color-secondary transition-colors hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)]"
                              title={tag}
                            >
                              {tag}
                            </button>
                          ))
                        ) : (
                          <div className="px-2.5 py-2 ui-text-micro ui-color-muted">
                            {availableTags.length === 0
                              ? t({
                                  id: "library.card.no_tags_yet",
                                  message: "No tags yet",
                                })
                              : t({
                                  id: "library.card.no_other_tags",
                                  message: "No other tags",
                                })}
                          </div>
                        )}
                      </div>
                    </FloatingPortal>
                  )}
                </div>
                <input
                  value={tagDraft}
                  onChange={(event) => onChangeTagDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      onCommitTagAdd();
                    }
                    if (event.key === "Escape") {
                      event.preventDefault();
                      onCancelTagEdit();
                    }
                  }}
                  onBlur={onCancelTagEdit}
                  placeholder={t({
                    id: "library.card.new_tag",
                    message: "New tag...",
                  })}
                  className="tag-input-intro box-border h-7 min-w-0 flex-1 border-b border-[var(--color-border-primary)] bg-transparent px-0.5 py-0 ui-text-meta leading-none ui-color-secondary outline-hidden placeholder:text-[var(--color-text-disabled)] focus:border-[var(--color-border-hover)]"
                  autoFocus
                />
              </div>
            ) : (
              <div className="flex max-h-14 min-w-0 flex-wrap content-start items-center gap-1.5 overflow-hidden">
                <button
                  type="button"
                  data-no-press
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    setTagMenuOpen(true);
                    onStartTagEdit();
                  }}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-[var(--color-border-secondary)] bg-[var(--color-bg-surface)] text-[14px] leading-none ui-color-secondary transition-colors hover:border-[var(--color-border-hover)] hover:text-[var(--color-text-primary)]"
                  aria-label={t({
                    id: "library.card.new_tag",
                    message: "New tag...",
                  })}
                >
                  +
                </button>
                {visibleTags.map((tag, index) => (
                  <div
                    data-no-press
                    key={`tag-${index}-${tag || "empty"}`}
                    onPointerDown={(event) => event.stopPropagation()}
                    className={`group/tag inline-flex h-6 max-w-full min-w-0 items-center justify-center rounded-md border border-[var(--color-border-secondary)] bg-[var(--color-bg-surface)] px-2 text-center ui-text-meta ui-color-secondary transition-colors duration-100 ease-out hover:border-[var(--color-border-hover)] hover:text-[var(--color-text-primary)] ${
                      shiftHeld
                        ? "hover:!border-[var(--color-error)]/50 hover:!text-[var(--color-error)] hover:line-through"
                        : ""
                    }`}
                  >
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        void onRemoveTag(tag);
                      }}
                      aria-label={t({
                        id: "library.card.remove_tag",
                        message: `Remove ${tag}`,
                      })}
                      className="relative mr-0.5 flex h-3 w-3 shrink-0 items-center justify-center rounded-full"
                    >
                      <span className="opacity-40 transition-opacity group-hover/tag:opacity-0">
                        #
                      </span>
                      <span className="absolute inset-0 flex items-center justify-center rounded-full border border-current opacity-0 transition-opacity group-hover/tag:opacity-100">
                        <X size={8} weight="bold" aria-hidden="true" />
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        if (shiftHeld) void onRemoveTag(tag);
                        else onClickTag?.(tag);
                      }}
                      className="min-w-0 truncate"
                      title={tag}
                    >
                      {tag}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div
            className="h-px w-full shrink-0 bg-[var(--color-border-primary)]"
            aria-hidden="true"
          />

          <div className="flex min-w-0 items-center justify-between gap-3 overflow-hidden ui-text-label ui-color-muted">
            <span className="shrink-0">
              {formatDuration(item.duration_seconds)}
            </span>
            <span className="min-w-0 truncate text-right">
              {formatBytes(item.file_size_bytes)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LibraryCard;
