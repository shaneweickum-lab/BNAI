"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { MainToWorkerMessage, WorkerToMainMessage } from "./workerProtocol";

export interface ModelStats {
  paramCount: number;
  fileSizeBytes: number;
  contextLen: number;
}

export interface DownloadProgress {
  loadedBytes: number;
  totalBytes: number;
  fromCache: boolean;
}

export type EngineStatus = "idle" | "downloading" | "initializing" | "ready" | "error";

export interface GenerateCallbacks {
  /** `textSoFar` is the full decoded reply text generated so far, not just the newest token. */
  onToken?: (textSoFar: string, tokensPerSecond: number) => void;
  onDone?: (totalTokens: number, elapsedMs: number) => void;
  onError?: (message: string) => void;
}

let requestCounter = 0;
function nextRequestId(): string {
  requestCounter += 1;
  return `req-${requestCounter}-${Date.now()}`;
}

/**
 * Owns the lifecycle of the inference Web Worker and exposes a small
 * imperative API to the chat UI. The worker itself (workers/inference.worker.ts)
 * owns the WASM load + generation loop; this hook just relays postMessage
 * traffic into React state and callbacks.
 */
export function useInferenceWorker(enabled: boolean = true) {
  const workerRef = useRef<Worker | null>(null);
  const callbacksRef = useRef<Map<string, GenerateCallbacks>>(new Map());

  const [status, setStatus] = useState<EngineStatus>("idle");
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;

    // Loaded from a pre-bundled plain-JS asset (see scripts/build-worker.mjs)
    // rather than `new Worker(new URL("../workers/inference.worker.ts", ...))`
    // -- Next's current Turbopack build does not reliably transpile that
    // pattern (see comment in scripts/build-worker.mjs for details).
    const worker = new Worker("/workers/inference.worker.js");
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
      const msg = event.data;
      switch (msg.type) {
        case "download-progress":
          setStatus("downloading");
          setDownloadProgress({ loadedBytes: msg.loadedBytes, totalBytes: msg.totalBytes, fromCache: msg.fromCache });
          break;
        case "ready":
          setStatus("ready");
          setModelStats({ paramCount: msg.paramCount, fileSizeBytes: msg.fileSizeBytes, contextLen: msg.contextLen });
          break;
        case "token": {
          const cb = callbacksRef.current.get(msg.requestId);
          cb?.onToken?.(msg.textSoFar, msg.tokensPerSecond);
          break;
        }
        case "done": {
          const cb = callbacksRef.current.get(msg.requestId);
          cb?.onDone?.(msg.totalTokens, msg.elapsedMs);
          callbacksRef.current.delete(msg.requestId);
          break;
        }
        case "error": {
          if (msg.requestId) {
            const cb = callbacksRef.current.get(msg.requestId);
            cb?.onError?.(msg.message);
            callbacksRef.current.delete(msg.requestId);
          } else {
            setStatus("error");
            setError(msg.message);
          }
          break;
        }
      }
    };

    worker.onerror = (event: ErrorEvent) => {
      setStatus("error");
      setError(event.message || "Unknown worker error");
    };

    // No setStatus("downloading") here: status starts at "idle", and the
    // demo UI already renders the same loading affordance for "idle" as for
    // "downloading" -- the real transition happens once the worker's first
    // "download-progress" message arrives, below.
    const initMsg: MainToWorkerMessage = { type: "init" };
    worker.postMessage(initMsg);

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [enabled]);

  const generate = useCallback((prompt: string, maxNewTokens: number, callbacks: GenerateCallbacks = {}): string => {
    const requestId = nextRequestId();
    callbacksRef.current.set(requestId, callbacks);
    const msg: MainToWorkerMessage = { type: "generate", requestId, prompt, maxNewTokens };
    workerRef.current?.postMessage(msg);
    return requestId;
  }, []);

  const cancel = useCallback((requestId: string) => {
    const msg: MainToWorkerMessage = { type: "cancel", requestId };
    workerRef.current?.postMessage(msg);
  }, []);

  const reset = useCallback(() => {
    const msg: MainToWorkerMessage = { type: "reset" };
    workerRef.current?.postMessage(msg);
  }, []);

  return { status, downloadProgress, modelStats, error, generate, cancel, reset };
}
