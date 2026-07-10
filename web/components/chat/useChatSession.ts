"use client";

/**
 * Owns the routing/generation logic for whichever conversation is currently
 * active -- lifted out of the old single-conversation app/demo/page.tsx
 * (its handleSend/finalizeAssistantMessage/dialogue-manager wiring) so it
 * operates on ANY conversation's persisted DialogueState, not one
 * page-level useState.
 *
 * The ref-based "avoid stale closures in worker callbacks" pattern from the
 * old page.tsx (dialogueStateRef) is kept, it just now tracks whichever
 * conversation is active and gets reloaded whenever `activeConversation`
 * changes.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { GenerateCallbacks } from "../../lib/useInferenceWorker";
import { route } from "../../lib/dialogue/dialogueManager";
import type { DialogueState } from "../../lib/dialogue/types";
import type { CompiledCategory } from "../../lib/aiml/types";
import type { RoutingStats, StoredConversation } from "../../lib/store/types";
import { EMPTY_ROUTING_STATS } from "../../lib/store/types";
import { deriveMessages, type ChatMessage, type ResolvedBy, type ResolvedByIndex } from "../../lib/chat/messages";
import { injectProjectInstructions } from "../../lib/chat/promptBuilder";

const MAX_NEW_TOKENS = 96;
const EMPTY_DIALOGUE_STATE: DialogueState = { topic: null, history: [] };
const EMPTY_RESOLVED_BY_INDEX: ResolvedByIndex = {};

interface UseChatSessionArgs {
  activeConversation: StoredConversation | null;
  /** The active conversation's project's instructions, if any (null/undefined = none). */
  projectInstructions: string | null | undefined;
  categories: CompiledCategory[];
  maxContextTokens: number | undefined;
  workerReady: boolean;
  generate: (prompt: string, maxNewTokens: number, callbacks: GenerateCallbacks) => string;
  cancel: (requestId: string) => void;
  updateConversation: (
    id: string,
    patch: Partial<Pick<StoredConversation, "dialogueState" | "routingStats" | "title" | "projectId">>,
  ) => void;
  autoTitleFromFirstMessage: (id: string, firstMessage: string) => void;
}

export interface ChatSession {
  messages: ChatMessage[];
  streamingText: string;
  isGenerating: boolean;
  /** Tokens/sec: live while generating, "last run" value once done. */
  tokensPerSecond: number;
  /** Wall-clock time of the most recent completed generation for the
   * active conversation, or null if none has happened yet this session. */
  lastElapsedMs: number | null;
  sendMessage: (userInput: string) => void;
  cancelGeneration: () => void;
}

export function useChatSession(args: UseChatSessionArgs): ChatSession {
  const {
    activeConversation,
    projectInstructions,
    categories,
    maxContextTokens,
    workerReady,
    generate,
    cancel,
    updateConversation,
    autoTitleFromFirstMessage,
  } = args;

  const activeId = activeConversation?.id ?? null;

  const dialogueStateRef = useRef<DialogueState>(activeConversation?.dialogueState ?? EMPTY_DIALOGUE_STATE);
  const routingStatsRef = useRef<RoutingStats>(activeConversation?.routingStats ?? EMPTY_ROUTING_STATS);
  const currentRequestIdRef = useRef<string | null>(null);
  // Request ids that were already finalized manually (via cancelGeneration
  // or a conversation switch) -- guards against the worker's late/delayed
  // "done" (or "error") message for that same request firing a *second*
  // finalize once it eventually arrives (the worker can still emit "done"
  // with partial output after a mid-decode cancel; see workers/inference.worker.ts).
  const finalizedRequestIdsRef = useRef<Set<string>>(new Set());
  const prevActiveIdRef = useRef<string | null>(null);
  const streamingTextRef = useRef("");

  const [streamingText, setStreamingText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [tokensPerSecond, setTokensPerSecond] = useState(0);
  const [lastElapsedMs, setLastElapsedMs] = useState<number | null>(null);
  // Per-conversation "how was this turn resolved" markers (see
  // lib/chat/messages.ts) -- kept as real state (not a ref) because it's
  // read during render (to build the message list) and only ever written
  // from effects/event handlers, never during render itself.
  const [resolvedByMaps, setResolvedByMaps] = useState<Record<string, ResolvedByIndex>>({});

  const markResolvedBy = useCallback((conversationId: string, index: number, resolvedBy: ResolvedBy) => {
    setResolvedByMaps((prev) => ({
      ...prev,
      [conversationId]: { ...prev[conversationId], [index]: resolvedBy },
    }));
  }, []);

  // Appends a (possibly partial) assistant turn to the given conversation
  // and persists it -- shared by the normal onDone path, cancelGeneration,
  // and the conversation-switch cleanup below.
  const appendAssistantTurn = useCallback(
    (conversationId: string, baseState: DialogueState, baseStats: RoutingStats, content: string) => {
      const newState: DialogueState = {
        ...baseState,
        history: [...baseState.history, { role: "assistant", content }],
      };
      markResolvedBy(conversationId, newState.history.length - 1, "generated");
      updateConversation(conversationId, { dialogueState: newState, routingStats: baseStats });
      return newState;
    },
    [markResolvedBy, updateConversation],
  );

  // Conversation switch: cancel any in-flight generation for the
  // conversation being left (finalizing whatever partial text had streamed
  // so far, same as the explicit "Stop" affordance), then load the newly
  // active conversation's persisted state.
  useEffect(() => {
    const prevId = prevActiveIdRef.current;
    if (prevId !== null && prevId !== activeId) {
      const inFlightRequestId = currentRequestIdRef.current;
      if (inFlightRequestId) {
        finalizedRequestIdsRef.current.add(inFlightRequestId);
        cancel(inFlightRequestId);
        currentRequestIdRef.current = null;

        const partial = streamingTextRef.current;
        // Only record a partial turn if something had actually streamed --
        // an instant cancel (nothing generated yet) shouldn't leave behind
        // an empty assistant bubble.
        if (partial.length > 0) {
          appendAssistantTurn(prevId, dialogueStateRef.current, routingStatsRef.current, partial);
        }
      }
    }

    prevActiveIdRef.current = activeId;
    dialogueStateRef.current = activeConversation?.dialogueState ?? EMPTY_DIALOGUE_STATE;
    routingStatsRef.current = activeConversation?.routingStats ?? EMPTY_ROUTING_STATS;
    streamingTextRef.current = "";
    // Deliberate: resetting ephemeral per-conversation UI state (streaming
    // text/generating flag/last stats) synchronously when the active
    // conversation actually changes, same pattern as the browserSupport
    // hydration effect in app/demo/page.tsx.
    setStreamingText("");
    setIsGenerating(false);
    setTokensPerSecond(0);
    setLastElapsedMs(null);
    // Only re-run this on an actual conversation switch (activeId change) --
    // activeConversation's *content* changes on every one of our own writes
    // too, and re-running this effect then would wrongly clobber in-flight
    // ephemeral UI state (streamingText etc).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  const sendMessage = useCallback(
    (userInput: string) => {
      const conv = activeConversation;
      const prompt = userInput.trim();
      if (!conv || !prompt || isGenerating || !workerReady) return;

      const conversationId = conv.id;
      const routed = route(dialogueStateRef.current, prompt, categories, maxContextTokens);

      if (routed.path === "aiml") {
        dialogueStateRef.current = routed.newState;
        const newStats: RoutingStats = {
          ...routingStatsRef.current,
          deterministic: routingStatsRef.current.deterministic + 1,
        };
        routingStatsRef.current = newStats;
        markResolvedBy(conversationId, routed.newState.history.length - 1, "deterministic");
        updateConversation(conversationId, { dialogueState: routed.newState, routingStats: newStats });
        autoTitleFromFirstMessage(conversationId, prompt);
        return;
      }

      // "gpt-no-match" / "gpt-ambiguous": dialogueManager already appended
      // the user turn; the assistant turn is completed once GPT responds.
      dialogueStateRef.current = routed.newState;
      const newStats: RoutingStats =
        routed.path === "gpt-no-match"
          ? { ...routingStatsRef.current, noMatch: routingStatsRef.current.noMatch + 1 }
          : { ...routingStatsRef.current, ambiguous: routingStatsRef.current.ambiguous + 1 };
      routingStatsRef.current = newStats;
      updateConversation(conversationId, { dialogueState: routed.newState, routingStats: newStats });
      autoTitleFromFirstMessage(conversationId, prompt);

      streamingTextRef.current = "";
      setStreamingText("");
      setTokensPerSecond(0);
      setIsGenerating(true);

      const finalPrompt = injectProjectInstructions(routed.gptPrompt, projectInstructions);

      const requestId = generate(finalPrompt, MAX_NEW_TOKENS, {
        onToken: (textSoFar, tps) => {
          streamingTextRef.current = textSoFar;
          setStreamingText(textSoFar);
          setTokensPerSecond(tps);
        },
        onDone: (_totalTokens, elapsedMs) => {
          if (finalizedRequestIdsRef.current.delete(requestId)) return; // already finalized (cancelled)
          setIsGenerating(false);
          currentRequestIdRef.current = null;
          setLastElapsedMs(elapsedMs);
          const responseText = streamingTextRef.current;
          dialogueStateRef.current = appendAssistantTurn(
            conversationId,
            dialogueStateRef.current,
            routingStatsRef.current,
            responseText,
          );
          streamingTextRef.current = "";
          setStreamingText("");
        },
        onError: (message) => {
          if (finalizedRequestIdsRef.current.delete(requestId)) return;
          setIsGenerating(false);
          currentRequestIdRef.current = null;
          dialogueStateRef.current = appendAssistantTurn(
            conversationId,
            dialogueStateRef.current,
            routingStatsRef.current,
            `[error: ${message}]`,
          );
          streamingTextRef.current = "";
          setStreamingText("");
        },
      });
      currentRequestIdRef.current = requestId;
    },
    [
      activeConversation,
      isGenerating,
      workerReady,
      categories,
      maxContextTokens,
      projectInstructions,
      generate,
      markResolvedBy,
      updateConversation,
      autoTitleFromFirstMessage,
      appendAssistantTurn,
    ],
  );

  const cancelGeneration = useCallback(() => {
    const conv = activeConversation;
    const requestId = currentRequestIdRef.current;
    if (!conv || !requestId) return;

    finalizedRequestIdsRef.current.add(requestId);
    cancel(requestId);
    currentRequestIdRef.current = null;
    setIsGenerating(false);

    const partial = streamingTextRef.current;
    if (partial.length > 0) {
      dialogueStateRef.current = appendAssistantTurn(conv.id, dialogueStateRef.current, routingStatsRef.current, partial);
    }
    streamingTextRef.current = "";
    setStreamingText("");
  }, [activeConversation, cancel, appendAssistantTurn]);

  const messages = activeConversation
    ? deriveMessages(
        activeConversation.dialogueState.history,
        resolvedByMaps[activeConversation.id] ?? EMPTY_RESOLVED_BY_INDEX,
        activeConversation.id,
      )
    : [];

  return { messages, streamingText, isGenerating, tokensPerSecond, lastElapsedMs, sendMessage, cancelGeneration };
}
