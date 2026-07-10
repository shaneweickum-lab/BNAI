/**
 * Pure state-transition functions for the chat store, kept separate from
 * `chatStore.ts`'s React hook wrapper so they're directly unit-testable
 * without rendering anything (same split as `lib/dialogue/dialogueManager.ts`
 * being pure logic the UI layer calls into).
 */

import { ChatStoreData, EMPTY_ROUTING_STATS, EMPTY_STORE, StoredConversation, StoredProject } from "./types";

export const STORAGE_KEY = "benny.chatStore.v1";
const TITLE_MAX_LENGTH = 60;

export function isValidStore(value: unknown): value is ChatStoreData {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return v.version === 1 && Array.isArray(v.projects) && Array.isArray(v.conversations);
}

/** Parses a raw (possibly missing/corrupt/unversioned) localStorage string
 * into a valid ChatStoreData, defaulting to an empty store rather than
 * throwing -- callers decide how/whether to log the fallback. */
export function parseStore(raw: string | null): ChatStoreData {
  if (!raw) return EMPTY_STORE;
  try {
    const parsed = JSON.parse(raw);
    return isValidStore(parsed) ? parsed : EMPTY_STORE;
  } catch {
    return EMPTY_STORE;
  }
}

export function deriveTitle(firstMessage: string): string {
  const trimmed = firstMessage.trim().replace(/\s+/g, " ");
  if (trimmed.length <= TITLE_MAX_LENGTH) return trimmed || "New chat";
  return trimmed.slice(0, TITLE_MAX_LENGTH - 1) + "…";
}

export function newConversation(id: string, now: number, projectId: string | null = null): StoredConversation {
  return {
    id,
    projectId,
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    dialogueState: { topic: null, history: [] },
    routingStats: { ...EMPTY_ROUTING_STATS },
  };
}

export function newProject(id: string, now: number, name: string): StoredProject {
  return { id, name, createdAt: now };
}

export function addConversation(data: ChatStoreData, conversation: StoredConversation): ChatStoreData {
  return { ...data, conversations: [conversation, ...data.conversations] };
}

export function patchConversation(
  data: ChatStoreData,
  id: string,
  patch: Partial<Pick<StoredConversation, "dialogueState" | "routingStats" | "title" | "projectId">>,
  now: number,
): ChatStoreData {
  return {
    ...data,
    conversations: data.conversations.map((c) => (c.id === id ? { ...c, ...patch, updatedAt: now } : c)),
  };
}

/** Auto-titles a conversation from its first user message -- only while
 * the title is still the "New chat" default, so an explicit rename is
 * never silently overwritten. */
export function autoTitleFromFirstMessage(data: ChatStoreData, id: string, firstMessage: string): ChatStoreData {
  return {
    ...data,
    conversations: data.conversations.map((c) =>
      c.id === id && c.title === "New chat" ? { ...c, title: deriveTitle(firstMessage) } : c,
    ),
  };
}

export function removeConversation(data: ChatStoreData, id: string): ChatStoreData {
  return { ...data, conversations: data.conversations.filter((c) => c.id !== id) };
}

export function addProject(data: ChatStoreData, project: StoredProject): ChatStoreData {
  return { ...data, projects: [project, ...data.projects] };
}

export function renameProjectOp(data: ChatStoreData, id: string, name: string): ChatStoreData {
  return { ...data, projects: data.projects.map((p) => (p.id === id ? { ...p, name } : p)) };
}

export function updateProjectInstructionsOp(data: ChatStoreData, id: string, instructions: string): ChatStoreData {
  return { ...data, projects: data.projects.map((p) => (p.id === id ? { ...p, instructions } : p)) };
}

/** Un-groups a project's conversations rather than deleting them --
 * destroying chat history as a side effect of deleting a folder would be
 * a bad surprise. */
export function removeProject(data: ChatStoreData, id: string): ChatStoreData {
  return {
    ...data,
    projects: data.projects.filter((p) => p.id !== id),
    conversations: data.conversations.map((c) => (c.projectId === id ? { ...c, projectId: null } : c)),
  };
}
