"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./demo.module.css";
import { useInferenceWorker } from "../../lib/useInferenceWorker";
import { checkBrowserSupport, type BrowserSupportResult } from "../../lib/browserSupport";
import { MODEL_NAME, PACKED_FILE_SIZE_MB, TOTAL_PARAMS, formatMB, formatParams } from "../../lib/modelInfo";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const MAX_NEW_TOKENS = 96;

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 MB";
  return formatMB(bytes);
}

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

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [tokensPerSecond, setTokensPerSecond] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const currentRequestId = useRef<string | null>(null);
  const chatWindowRef = useRef<HTMLDivElement | null>(null);
  const messageIdCounter = useRef(0);
  const nextMessageId = (prefix: string) => `${prefix}-${(messageIdCounter.current += 1)}`;

  useEffect(() => {
    chatWindowRef.current?.scrollTo({ top: chatWindowRef.current.scrollHeight });
  }, [messages, streamingText]);

  const progressPct = useMemo(() => {
    if (!downloadProgress || downloadProgress.totalBytes === 0) return downloadProgress?.fromCache ? 100 : 0;
    return Math.min(100, (downloadProgress.loadedBytes / downloadProgress.totalBytes) * 100);
  }, [downloadProgress]);

  function handleSend() {
    const prompt = input.trim();
    if (!prompt || isGenerating || status !== "ready") return;

    const userMsg: ChatMessage = { id: nextMessageId("u"), role: "user", content: prompt };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setStreamingText("");
    setTokensPerSecond(0);
    setIsGenerating(true);

    const requestId = generate(prompt, MAX_NEW_TOKENS, {
      onToken: (textSoFar, tps) => {
        setStreamingText(textSoFar);
        setTokensPerSecond(tps);
      },
      onDone: () => {
        setIsGenerating(false);
        currentRequestId.current = null;
        finalizeAssistantMessage();
      },
      onError: (message) => {
        setIsGenerating(false);
        currentRequestId.current = null;
        setMessages((prev) => [...prev, { id: nextMessageId("e"), role: "assistant", content: `[error: ${message}]` }]);
      },
    });
    currentRequestId.current = requestId;
  }

  // Ref so onDone/handleCancel can read the latest streamed text without a stale closure.
  const streamingTextRef = useRef("");
  useEffect(() => {
    streamingTextRef.current = streamingText;
  }, [streamingText]);

  function finalizeAssistantMessage() {
    setMessages((prev) => [
      ...prev,
      { id: nextMessageId("a"), role: "assistant", content: streamingTextRef.current },
    ]);
    setStreamingText("");
  }

  function handleCancel() {
    if (currentRequestId.current) {
      cancel(currentRequestId.current);
      setIsGenerating(false);
      finalizeAssistantMessage();
      currentRequestId.current = null;
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const modelBadgeText = modelStats
    ? `${formatParams(modelStats.paramCount)} params · ternary · ${formatBytes(modelStats.fileSizeBytes)} packed`
    : `~${formatParams(TOTAL_PARAMS)} params · ternary · ${PACKED_FILE_SIZE_MB.toFixed(2)} MB packed`;

  return (
    <div className={styles.wrap}>
      <div className={styles.topBar}>
        <h1 className={styles.title}>{MODEL_NAME} demo</h1>
        <div className={styles.statsRow}>
          <span className="badge">{modelBadgeText}</span>
          {status === "ready" && <span className="badge">100% client-side</span>}
        </div>
      </div>

      <div className={`callout ${styles.limitsCallout}`}>
        This is a {formatParams(TOTAL_PARAMS)}-parameter model running entirely in your browser &mdash;
        it has a short context window and will not reason like a frontier assistant. The weights
        loaded right now are an <strong>untrained placeholder</strong>, so responses are expected to be
        gibberish until the real training run lands. This page exists to prove the plumbing (download,
        WASM load, worker, streaming, tokens/sec) works end-to-end.
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
                : "Downloading model (one-time, ~48MB; cached after this)..."}
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

      {checked && supported && status === "ready" && (
        <>
          <div className={styles.chatWindow} ref={chatWindowRef}>
            {messages.length === 0 && !streamingText && (
              <div className={styles.emptyState}>
                Model loaded. Type a message below &mdash; generation runs in a Web Worker, off the main
                thread.
              </div>
            )}
            {messages.map((m) => (
              <div
                key={m.id}
                className={`${styles.messageRow} ${m.role === "user" ? styles.messageRowUser : styles.messageRowAssistant}`}
              >
                <span className={styles.messageRole}>{m.role}</span>
                <div className={`${styles.bubble} ${m.role === "user" ? styles.bubbleUser : styles.bubbleAssistant}`}>
                  {m.content}
                </div>
              </div>
            ))}
            {isGenerating && (
              <div className={`${styles.messageRow} ${styles.messageRowAssistant}`}>
                <span className={styles.messageRole}>assistant</span>
                <div className={`${styles.bubble} ${styles.bubbleAssistant}`}>
                  {streamingText}
                  <span className={styles.cursor} />
                </div>
              </div>
            )}
          </div>

          <div className={styles.liveStats}>
            <span>{isGenerating ? `${tokensPerSecond.toFixed(1)} tok/s` : tokensPerSecond > 0 ? `last run: ${tokensPerSecond.toFixed(1)} tok/s` : ""}</span>
          </div>

          <div className={styles.inputRow}>
            <textarea
              className={styles.textarea}
              placeholder="Say something to Benny..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isGenerating}
            />
            {isGenerating ? (
              <button className="button buttonSecondary" onClick={handleCancel}>
                Stop
              </button>
            ) : (
              <button className="button" onClick={handleSend} disabled={!input.trim()}>
                Send
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
