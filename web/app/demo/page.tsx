"use client";

import { useEffect, useState } from "react";
import styles from "./demo.module.css";
import { useInferenceWorker } from "../../lib/useInferenceWorker";
import { checkBrowserSupport, type BrowserSupportResult } from "../../lib/browserSupport";
import { MODEL_NAME, PACKED_FILE_SIZE_MB, TOTAL_PARAMS, formatMB, formatParams } from "../../lib/modelInfo";
import categoriesData from "../../lib/aiml/generated/categories.json";
import type { CompiledCategory } from "../../lib/aiml/types";
import { useChatStore } from "../../lib/store/chatStore";
import { useChatSession } from "../../components/chat/useChatSession";
import AppShell from "../../components/chat/AppShell";
import Sidebar from "../../components/chat/Sidebar";
import MetricsPanel from "../../components/chat/MetricsPanel";
import ChatWindow from "../../components/chat/ChatWindow";

// The JSON import loses the literal ("literal" | "*" | "_" | ...) types
// TypeScript infers from a plain .json file down to `string` -- this is a
// static, compile-time-known dataset produced by the Python AIML compiler
// (aiml/tools/aiml_compiler.py), so the cast just restores the precise
// shape lib/aiml/types.ts already declares for it.
const categories = categoriesData.categories as unknown as CompiledCategory[];

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 MB";
  return formatMB(bytes);
}

/**
 * Thin container for the /demo app shell: owns the model-loading gate
 * (browser support check + worker download/init status, unchanged from the
 * original single-conversation page), which conversation is currently
 * active, and wires the persisted chat store + inference worker into the
 * AppShell/Sidebar/ChatWindow/MetricsPanel components. All the actual
 * routing/generation logic lives in components/chat/useChatSession.ts.
 */
export default function DemoPage() {
  const [browserSupport, setBrowserSupport] = useState<BrowserSupportResult | null>(null);

  useEffect(() => {
    // Deliberately deferred to an effect (not computed during render): this
    // reads navigator.userAgent, which must stay in sync with the actual
    // client after hydration rather than whatever a server render guessed.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setBrowserSupport(checkBrowserSupport());
  }, []);

  const supported = browserSupport?.supported ?? true; // avoid flashing "unsupported" before check runs
  const checked = browserSupport !== null;

  const { status, downloadProgress, modelStats, error, generate, cancel } = useInferenceWorker(checked && supported);

  const {
    hydrated,
    projects,
    conversations,
    createConversation,
    updateConversation,
    autoTitleFromFirstMessage,
    renameConversation,
    deleteConversation,
    moveConversationToProject,
    createProject,
    renameProject,
    updateProjectInstructions,
    deleteProject,
  } = useChatStore();

  // Which conversation is open -- component state (not persisted), defaults
  // to the most-recently-updated conversation once the store has hydrated,
  // or creates a fresh one if none exist yet. The same effect also re-runs
  // this selection if the active conversation is ever deleted (activeConversationId
  // reset to null below), so there's always something sensible open.
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  useEffect(() => {
    if (!hydrated || activeConversationId !== null) return;
    if (conversations.length > 0) {
      const mostRecent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0];
      // Deliberate: picking the default active conversation is inherently a
      // post-hydration side effect (localStorage isn't available during
      // render), same pattern as the browserSupport effect above.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setActiveConversationId(mostRecent.id);
    } else {
      const conv = createConversation(null);
      setActiveConversationId(conv.id);
    }
  }, [hydrated, activeConversationId, conversations, createConversation]);

  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;
  const activeProject = activeConversation ? projects.find((p) => p.id === activeConversation.projectId) ?? null : null;

  const session = useChatSession({
    activeConversation,
    projectInstructions: activeProject?.instructions,
    categories,
    maxContextTokens: modelStats?.contextLen,
    workerReady: status === "ready",
    generate,
    cancel,
    updateConversation,
    autoTitleFromFirstMessage,
  });

  function handleNewChat(projectId: string | null) {
    const conv = createConversation(projectId);
    setActiveConversationId(conv.id);
  }

  const progressPct =
    !downloadProgress || downloadProgress.totalBytes === 0
      ? downloadProgress?.fromCache
        ? 100
        : 0
      : Math.min(100, (downloadProgress.loadedBytes / downloadProgress.totalBytes) * 100);

  const modelBadgeText = modelStats
    ? `${formatParams(modelStats.paramCount)} params · ternary · ${formatBytes(modelStats.fileSizeBytes)} packed`
    : `~${formatParams(TOTAL_PARAMS)} params · ternary · ${PACKED_FILE_SIZE_MB.toFixed(2)} MB packed`;

  const ready = checked && supported && status === "ready" && hydrated && activeConversationId !== null;

  if (!ready) {
    return (
      <div className={styles.wrap}>
        <div className={styles.topBar}>
          <h1 className={styles.title}>{MODEL_NAME} demo</h1>
          <div className={styles.statsRow}>
            <span className="badge">{modelBadgeText}</span>
          </div>
        </div>

        {checked && !supported && (
          <div className={styles.unsupportedBox}>
            <h3>Your browser/device isn&apos;t supported</h3>
            <ul>
              {browserSupport?.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
            <p>Try a recent desktop Chrome, Firefox, Edge, or Safari 16.4+ / iOS 16.4+.</p>
          </div>
        )}

        {checked && supported && status === "error" && (
          <div className={styles.errorBox}>Failed to initialize: {error}</div>
        )}

        {checked && supported && (status === "downloading" || status === "initializing" || status === "idle") && (
          <div className={styles.progressWrap}>
            <div className={styles.progressLabel}>
              <span>
                {downloadProgress?.fromCache
                  ? "Loading model from browser cache..."
                  : `Downloading model (one-time, ~${PACKED_FILE_SIZE_MB.toFixed(0)}MB; cached after this)...`}
              </span>
              <span className="mono">
                {downloadProgress
                  ? `${formatBytes(downloadProgress.loadedBytes)} / ${formatBytes(downloadProgress.totalBytes || downloadProgress.loadedBytes)}`
                  : "starting..."}
              </span>
            </div>
            <div className={styles.progressTrack}>
              <div className={styles.progressFill} style={{ width: `${progressPct}%` }} />
            </div>
          </div>
        )}

        {checked && supported && status === "ready" && !ready && (
          <div className={styles.progressWrap}>
            <div className={styles.progressLabel}>
              <span>Loading your conversations...</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <AppShell
      sidebar={
        <Sidebar
          projects={projects}
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={setActiveConversationId}
          onNewChat={handleNewChat}
          onRenameConversation={renameConversation}
          onDeleteConversation={(id) => {
            deleteConversation(id);
            if (id === activeConversationId) setActiveConversationId(null);
          }}
          onMoveConversation={moveConversationToProject}
          onCreateProject={createProject}
          onRenameProject={renameProject}
          onDeleteProject={deleteProject}
          onUpdateProjectInstructions={updateProjectInstructions}
        />
      }
      metrics={
        <MetricsPanel
          modelStats={modelStats}
          activeConversation={activeConversation}
          tokensPerSecond={session.tokensPerSecond}
          isGenerating={session.isGenerating}
          lastElapsedMs={session.lastElapsedMs}
        />
      }
    >
      <ChatWindow
        messages={session.messages}
        streamingText={session.streamingText}
        isGenerating={session.isGenerating}
        tokensPerSecond={session.tokensPerSecond}
        onSend={session.sendMessage}
        onCancel={session.cancelGeneration}
      />
    </AppShell>
  );
}
