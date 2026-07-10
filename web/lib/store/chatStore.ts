"use client";

/**
 * Client-only persistence for chat history + projects, via localStorage.
 * Nothing here ever leaves the browser -- this is still the "100%
 * client-side, no server calls" story, just no longer wiped on refresh.
 *
 * This is a thin React wrapper; the actual state-transition logic lives in
 * `chatStoreOps.ts` as plain, directly-testable functions.
 */

import { useCallback, useEffect, useState } from "react";
import {
  STORAGE_KEY,
  addConversation,
  addProject,
  autoTitleFromFirstMessage as autoTitleFromFirstMessageOp,
  newConversation,
  newProject,
  parseStore,
  patchConversation,
  removeConversation,
  removeProject,
  renameProjectOp,
  updateProjectInstructionsOp,
} from "./chatStoreOps";
import { ChatStoreData, EMPTY_STORE, StoredConversation, StoredProject } from "./types";

function persistStore(data: ChatStoreData) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (err) {
    // Quota exceeded, private-browsing storage restrictions, etc. -- the
    // in-memory session still works, it just won't survive a reload.
    console.warn("[chatStore] failed to persist data (storage full or unavailable)", err);
  }
}

function loadStore(): ChatStoreData {
  if (typeof window === "undefined") return EMPTY_STORE;
  return parseStore(window.localStorage.getItem(STORAGE_KEY));
}

export function useChatStore() {
  const [data, setData] = useState<ChatStoreData>(EMPTY_STORE);
  const [hydrated, setHydrated] = useState(false);

  // Loaded in an effect (not during render) to avoid an SSR/client
  // hydration mismatch -- localStorage doesn't exist on the server.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(loadStore());
    setHydrated(true);
  }, []);

  const update = useCallback((updater: (prev: ChatStoreData) => ChatStoreData) => {
    setData((prev) => {
      const next = updater(prev);
      persistStore(next);
      return next;
    });
  }, []);

  const createConversation = useCallback(
    (projectId: string | null = null): StoredConversation => {
      const conversation = newConversation(crypto.randomUUID(), Date.now(), projectId);
      update((prev) => addConversation(prev, conversation));
      return conversation;
    },
    [update],
  );

  const updateConversation = useCallback(
    (id: string, patch: Partial<Pick<StoredConversation, "dialogueState" | "routingStats" | "title" | "projectId">>) => {
      update((prev) => patchConversation(prev, id, patch, Date.now()));
    },
    [update],
  );

  const autoTitleFromFirstMessage = useCallback(
    (id: string, firstMessage: string) => {
      update((prev) => autoTitleFromFirstMessageOp(prev, id, firstMessage));
    },
    [update],
  );

  const renameConversation = useCallback(
    (id: string, title: string) => {
      update((prev) => patchConversation(prev, id, { title }, Date.now()));
    },
    [update],
  );

  const deleteConversation = useCallback(
    (id: string) => {
      update((prev) => removeConversation(prev, id));
    },
    [update],
  );

  const moveConversationToProject = useCallback(
    (id: string, projectId: string | null) => {
      update((prev) => patchConversation(prev, id, { projectId }, Date.now()));
    },
    [update],
  );

  const createProject = useCallback(
    (name: string): StoredProject => {
      const project = newProject(crypto.randomUUID(), Date.now(), name);
      update((prev) => addProject(prev, project));
      return project;
    },
    [update],
  );

  const renameProject = useCallback(
    (id: string, name: string) => {
      update((prev) => renameProjectOp(prev, id, name));
    },
    [update],
  );

  const updateProjectInstructions = useCallback(
    (id: string, instructions: string) => {
      update((prev) => updateProjectInstructionsOp(prev, id, instructions));
    },
    [update],
  );

  const deleteProject = useCallback(
    (id: string) => {
      update((prev) => removeProject(prev, id));
    },
    [update],
  );

  return {
    hydrated,
    projects: data.projects,
    conversations: data.conversations,
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
  };
}
