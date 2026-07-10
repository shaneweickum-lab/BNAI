import { describe, expect, it } from "vitest";
import {
  addConversation,
  addProject,
  autoTitleFromFirstMessage,
  deriveTitle,
  isValidStore,
  newConversation,
  newProject,
  parseStore,
  patchConversation,
  removeConversation,
  removeProject,
  renameProjectOp,
  updateProjectInstructionsOp,
} from "./chatStoreOps";
import { EMPTY_STORE, type ChatStoreData } from "./types";

describe("deriveTitle", () => {
  it("uses the trimmed, whitespace-collapsed message as the title when short", () => {
    expect(deriveTitle("  hello   there  ")).toBe("hello there");
  });

  it("falls back to 'New chat' for an empty/whitespace-only message", () => {
    expect(deriveTitle("   ")).toBe("New chat");
  });

  it("truncates long messages with an ellipsis", () => {
    const long = "a".repeat(100);
    const title = deriveTitle(long);
    expect(title.length).toBe(60);
    expect(title.endsWith("…")).toBe(true);
  });
});

describe("isValidStore / parseStore", () => {
  it("accepts a well-formed store", () => {
    const data: ChatStoreData = { version: 1, projects: [], conversations: [] };
    expect(isValidStore(data)).toBe(true);
  });

  it("rejects missing/wrong version", () => {
    expect(isValidStore({ version: 2, projects: [], conversations: [] })).toBe(false);
    expect(isValidStore({ projects: [], conversations: [] })).toBe(false);
  });

  it("rejects non-array projects/conversations", () => {
    expect(isValidStore({ version: 1, projects: "nope", conversations: [] })).toBe(false);
  });

  it("parseStore falls back to EMPTY_STORE for null input", () => {
    expect(parseStore(null)).toEqual(EMPTY_STORE);
  });

  it("parseStore falls back to EMPTY_STORE for unparseable JSON", () => {
    expect(parseStore("{not valid json")).toEqual(EMPTY_STORE);
  });

  it("parseStore falls back to EMPTY_STORE for valid JSON with the wrong shape", () => {
    expect(parseStore(JSON.stringify({ foo: "bar" }))).toEqual(EMPTY_STORE);
  });

  it("parseStore accepts a genuinely valid stored blob", () => {
    const data: ChatStoreData = { version: 1, projects: [], conversations: [] };
    expect(parseStore(JSON.stringify(data))).toEqual(data);
  });
});

describe("conversation ops", () => {
  it("addConversation prepends the new conversation", () => {
    const conv = newConversation("c1", 1000);
    const next = addConversation(EMPTY_STORE, conv);
    expect(next.conversations).toEqual([conv]);
    expect(conv.title).toBe("New chat");
    expect(conv.dialogueState).toEqual({ topic: null, history: [] });
  });

  it("patchConversation updates only the targeted conversation and bumps updatedAt", () => {
    const conv = newConversation("c1", 1000);
    const store = addConversation(EMPTY_STORE, conv);
    const next = patchConversation(store, "c1", { title: "Renamed" }, 2000);
    expect(next.conversations[0].title).toBe("Renamed");
    expect(next.conversations[0].updatedAt).toBe(2000);
    expect(next.conversations[0].createdAt).toBe(1000); // unchanged
  });

  it("autoTitleFromFirstMessage only applies while title is still the default", () => {
    const conv = newConversation("c1", 1000);
    let store = addConversation(EMPTY_STORE, conv);
    store = autoTitleFromFirstMessage(store, "c1", "hello there, how are you?");
    expect(store.conversations[0].title).toBe("hello there, how are you?");

    // A second call must not clobber an already-derived (or user-renamed) title.
    store = autoTitleFromFirstMessage(store, "c1", "a totally different message");
    expect(store.conversations[0].title).toBe("hello there, how are you?");
  });

  it("removeConversation removes only the targeted conversation", () => {
    let store = addConversation(EMPTY_STORE, newConversation("c1", 1000));
    store = addConversation(store, newConversation("c2", 1000));
    store = removeConversation(store, "c1");
    expect(store.conversations.map((c) => c.id)).toEqual(["c2"]);
  });
});

describe("project ops", () => {
  it("addProject prepends the new project", () => {
    const project = newProject("p1", 1000, "Research");
    const next = addProject(EMPTY_STORE, project);
    expect(next.projects).toEqual([project]);
  });

  it("renameProjectOp renames only the targeted project", () => {
    let store = addProject(EMPTY_STORE, newProject("p1", 1000, "Old name"));
    store = renameProjectOp(store, "p1", "New name");
    expect(store.projects[0].name).toBe("New name");
  });

  it("updateProjectInstructionsOp sets instructions on the targeted project", () => {
    let store = addProject(EMPTY_STORE, newProject("p1", 1000, "Research"));
    store = updateProjectInstructionsOp(store, "p1", "Be terse.");
    expect(store.projects[0].instructions).toBe("Be terse.");
  });

  it("removeProject un-groups its conversations instead of deleting them", () => {
    let store = addProject(EMPTY_STORE, newProject("p1", 1000, "Research"));
    store = addConversation(store, newConversation("c1", 1000, "p1"));
    store = removeProject(store, "p1");

    expect(store.projects).toEqual([]);
    expect(store.conversations).toHaveLength(1);
    expect(store.conversations[0].projectId).toBeNull();
  });
});
