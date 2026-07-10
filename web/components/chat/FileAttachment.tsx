"use client";

/**
 * File-picker control + attached-file chip row, rendered above the chat
 * input. Reads the file via FileReader.readAsText() and rejects anything
 * that doesn't decode as plausible plain text (binary/images/PDFs), with a
 * clear inline message -- never a silent failure.
 */

import { useRef, useState } from "react";
import styles from "./FileAttachment.module.css";
import { FILE_INPUT_ACCEPT, UNSUPPORTED_FILE_MESSAGE, looksLikeDecodedText } from "../../lib/chat/textFile";

export interface AttachedFile {
  name: string;
  content: string;
}

interface FileAttachmentProps {
  attached: AttachedFile | null;
  onAttach: (file: AttachedFile) => void;
  onRemove: () => void;
  disabled?: boolean;
}

export default function FileAttachment({ attached, onAttach, onRemove, disabled }: FileAttachmentProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    // Reset the input value so selecting the same file again still fires onChange.
    e.target.value = "";
    if (!file) return;

    setError(null);
    const reader = new FileReader();
    reader.onload = () => {
      const content = typeof reader.result === "string" ? reader.result : "";
      if (!looksLikeDecodedText(content)) {
        setError(UNSUPPORTED_FILE_MESSAGE);
        return;
      }
      onAttach({ name: file.name, content });
    };
    reader.onerror = () => {
      setError("Couldn't read that file.");
    };
    reader.readAsText(file);
  }

  return (
    <div className={styles.wrap}>
      {attached && (
        <div className={styles.chipRow}>
          <span className={styles.chip}>
            <span className={styles.chipName}>{attached.name}</span>
            <button
              type="button"
              className={styles.chipRemove}
              onClick={onRemove}
              aria-label={`Remove attached file ${attached.name}`}
            >
              ×
            </button>
          </span>
        </div>
      )}
      {error && <div className={styles.error}>{error}</div>}
      <button
        type="button"
        className={`button buttonSecondary ${styles.attachButton}`}
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
      >
        📎 Attach file
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={FILE_INPUT_ACCEPT}
        className={styles.hiddenInput}
        onChange={handleFileChange}
        aria-label="Attach a text file"
      />
    </div>
  );
}
