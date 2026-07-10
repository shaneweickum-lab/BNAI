"use client";

/**
 * Message list + streaming-response rendering + input row for whichever
 * conversation is active. Lifted from the old single-conversation
 * app/demo/page.tsx (message bubbles, resolvedBy badges, streaming cursor)
 * -- this part isn't redesigned, just moved into its own component that
 * receives the active conversation's data as props.
 */

import { useEffect, useRef, useState } from "react";
import styles from "./ChatWindow.module.css";
import FileAttachment, { type AttachedFile } from "./FileAttachment";
import { buildAttachedMessage } from "../../lib/chat/textFile";
import type { ChatMessage } from "../../lib/chat/messages";

interface ChatWindowProps {
  messages: ChatMessage[];
  streamingText: string;
  isGenerating: boolean;
  tokensPerSecond: number;
  onSend: (text: string) => void;
  onCancel: () => void;
}

export default function ChatWindow({ messages, streamingText, isGenerating, tokensPerSecond, onSend, onCancel }: ChatWindowProps) {
  const [input, setInput] = useState("");
  const [attached, setAttached] = useState<AttachedFile | null>(null);
  const chatWindowRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatWindowRef.current?.scrollTo({ top: chatWindowRef.current.scrollHeight });
  }, [messages, streamingText]);

  function handleSend() {
    const typed = input.trim();
    if (!typed && !attached) return;
    if (isGenerating) return;

    const finalText = attached ? buildAttachedMessage(attached.name, attached.content, typed) : typed;
    if (!finalText.trim()) return;

    onSend(finalText);
    setInput("");
    setAttached(null);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const canSend = (input.trim().length > 0 || attached !== null) && !isGenerating;

  return (
    <div className={styles.wrap}>
      <div className={`callout ${styles.limitsCallout}`}>
        This is a small parameter-count model running entirely in your browser &mdash; it has a short context
        window and will not reason like a frontier assistant. The weights loaded right now are an{" "}
        <strong>untrained placeholder</strong>, so responses are expected to be gibberish until the real training
        run lands.
      </div>

      <div className={styles.chatWindow} ref={chatWindowRef}>
        {messages.length === 0 && !streamingText && (
          <div className={styles.emptyState}>
            Type a message below &mdash; generation runs in a Web Worker, off the main thread.
          </div>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            className={`${styles.messageRow} ${m.role === "user" ? styles.messageRowUser : styles.messageRowAssistant}`}
          >
            <span className={styles.messageRole}>
              {m.role}
              {m.resolvedBy && (
                <span
                  className={`${styles.resolvedBadge} ${
                    m.resolvedBy === "deterministic" ? styles.resolvedBadgeDeterministic : styles.resolvedBadgeGenerated
                  }`}
                >
                  {m.resolvedBy === "deterministic" ? "deterministic" : "generated"}
                </span>
              )}
            </span>
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
        <span>
          {isGenerating
            ? `${tokensPerSecond.toFixed(1)} tok/s`
            : tokensPerSecond > 0
              ? `last run: ${tokensPerSecond.toFixed(1)} tok/s`
              : ""}
        </span>
      </div>

      <div className={styles.inputArea}>
        <FileAttachment attached={attached} onAttach={setAttached} onRemove={() => setAttached(null)} disabled={isGenerating} />
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
            <button className="button buttonSecondary" onClick={onCancel}>
              Stop
            </button>
          ) : (
            <button className="button" onClick={handleSend} disabled={!canSend}>
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
