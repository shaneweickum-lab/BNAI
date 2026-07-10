import type { DialogueState } from "../dialogue/types";

export interface StoredProject {
  id: string;
  name: string;
  /** Short, optional custom-instructions text (Claude-Projects-style),
   * prepended to the system turn for conversations in this project.
   * Deliberately NOT a persistent "project knowledge files" feature --
   * Benny's 2048-token context makes always-on large file context a poor
   * fit; per-conversation file attachments cover the "add files" ask directly. */
  instructions?: string;
  createdAt: number;
}

export interface RoutingStats {
  deterministic: number;
  noMatch: number;
  ambiguous: number;
}

export interface StoredConversation {
  id: string;
  /** null = ungrouped ("Chats" section), otherwise a StoredProject.id */
  projectId: string | null;
  title: string;
  createdAt: number;
  updatedAt: number;
  dialogueState: DialogueState;
  routingStats: RoutingStats;
}

export const EMPTY_ROUTING_STATS: RoutingStats = { deterministic: 0, noMatch: 0, ambiguous: 0 };

export interface ChatStoreData {
  version: 1;
  projects: StoredProject[];
  conversations: StoredConversation[];
}

export const EMPTY_STORE: ChatStoreData = { version: 1, projects: [], conversations: [] };
