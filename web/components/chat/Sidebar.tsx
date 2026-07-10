"use client";

/**
 * Left-drawer content: "+ New chat", a collapsible "Projects" section (each
 * project's conversations nested underneath, most-recently-updated first),
 * and an ungrouped "Chats" section below for conversations with
 * projectId: null, same recency ordering.
 */

import { useState } from "react";
import styles from "./Sidebar.module.css";
import type { StoredConversation, StoredProject } from "../../lib/store/types";

interface SidebarProps {
  projects: StoredProject[];
  conversations: StoredConversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: (projectId: string | null) => void;
  onRenameConversation: (id: string, title: string) => void;
  onDeleteConversation: (id: string) => void;
  onMoveConversation: (id: string, projectId: string | null) => void;
  onCreateProject: (name: string) => void;
  onRenameProject: (id: string, name: string) => void;
  onDeleteProject: (id: string) => void;
  onUpdateProjectInstructions: (id: string, instructions: string) => void;
}

function byRecency(a: StoredConversation, b: StoredConversation): number {
  return b.updatedAt - a.updatedAt;
}

export default function Sidebar({
  projects,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onRenameConversation,
  onDeleteConversation,
  onMoveConversation,
  onCreateProject,
  onRenameProject,
  onDeleteProject,
  onUpdateProjectInstructions,
}: SidebarProps) {
  const [collapsedProjects, setCollapsedProjects] = useState<Set<string>>(new Set());
  const [renamingConversationId, setRenamingConversationId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renamingProjectId, setRenamingProjectId] = useState<string | null>(null);
  const [projectRenameDraft, setProjectRenameDraft] = useState("");
  const [showNewProjectInput, setShowNewProjectInput] = useState(false);
  const [newProjectDraft, setNewProjectDraft] = useState("");
  const [editingInstructionsId, setEditingInstructionsId] = useState<string | null>(null);
  const [instructionsDraft, setInstructionsDraft] = useState("");

  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;
  const currentProjectContext = activeConversation?.projectId ?? null;

  const ungrouped = conversations.filter((c) => c.projectId === null).sort(byRecency);

  function toggleProjectCollapsed(id: string) {
    setCollapsedProjects((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function startRename(conv: StoredConversation) {
    setRenamingConversationId(conv.id);
    setRenameDraft(conv.title);
  }

  function commitRename() {
    if (renamingConversationId && renameDraft.trim()) {
      onRenameConversation(renamingConversationId, renameDraft.trim());
    }
    setRenamingConversationId(null);
  }

  function startProjectRename(project: StoredProject) {
    setRenamingProjectId(project.id);
    setProjectRenameDraft(project.name);
  }

  function commitProjectRename() {
    if (renamingProjectId && projectRenameDraft.trim()) {
      onRenameProject(renamingProjectId, projectRenameDraft.trim());
    }
    setRenamingProjectId(null);
  }

  function submitNewProject(e: React.FormEvent) {
    e.preventDefault();
    const name = newProjectDraft.trim();
    if (name) {
      onCreateProject(name);
    }
    setNewProjectDraft("");
    setShowNewProjectInput(false);
  }

  function toggleInstructionsEditor(project: StoredProject) {
    if (editingInstructionsId === project.id) {
      setEditingInstructionsId(null);
      return;
    }
    setInstructionsDraft(project.instructions ?? "");
    setEditingInstructionsId(project.id);
  }

  function commitInstructions(id: string) {
    onUpdateProjectInstructions(id, instructionsDraft.trim());
    setEditingInstructionsId(null);
  }

  function renderConversationRow(conv: StoredConversation) {
    const isActive = conv.id === activeConversationId;
    const isRenaming = renamingConversationId === conv.id;

    return (
      <div
        key={conv.id}
        className={`${styles.conversationRow} ${isActive ? styles.conversationRowActive : ""}`}
        onClick={() => !isRenaming && onSelectConversation(conv.id)}
      >
        {isRenaming ? (
          <input
            className={styles.renameInput}
            value={renameDraft}
            autoFocus
            onChange={(e) => setRenameDraft(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") setRenamingConversationId(null);
            }}
          />
        ) : (
          <span className={styles.conversationTitle} title={conv.title}>
            {conv.title}
          </span>
        )}
        {!isRenaming && (
          <span className={styles.conversationActions} onClick={(e) => e.stopPropagation()}>
            <select
              className={styles.moveSelect}
              value={conv.projectId ?? ""}
              aria-label={`Move "${conv.title}" to project`}
              onChange={(e) => onMoveConversation(conv.id, e.target.value || null)}
            >
              <option value="">No project</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              className={styles.iconButton}
              title="Rename"
              aria-label={`Rename "${conv.title}"`}
              onClick={() => startRename(conv)}
            >
              ✎
            </button>
            <button
              type="button"
              className={styles.iconButton}
              title="Delete"
              aria-label={`Delete "${conv.title}"`}
              onClick={() => {
                if (window.confirm(`Delete "${conv.title}"? This can't be undone.`)) {
                  onDeleteConversation(conv.id);
                }
              }}
            >
              ×
            </button>
          </span>
        )}
      </div>
    );
  }

  return (
    <nav className={styles.sidebar} aria-label="Conversations">
      <button type="button" className={`button ${styles.newChatButton}`} onClick={() => onNewChat(currentProjectContext)}>
        + New chat
      </button>

      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>Projects</span>
          <button
            type="button"
            className={styles.iconButton}
            title="New project"
            aria-label="New project"
            onClick={() => setShowNewProjectInput((v) => !v)}
          >
            +
          </button>
        </div>

        {showNewProjectInput && (
          <form className={styles.newProjectForm} onSubmit={submitNewProject}>
            <input
              className={styles.newProjectInput}
              placeholder="Project name"
              autoFocus
              value={newProjectDraft}
              onChange={(e) => setNewProjectDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") setShowNewProjectInput(false);
              }}
            />
            <button type="submit" className="button buttonSecondary">
              Add
            </button>
          </form>
        )}

        {projects.length === 0 && !showNewProjectInput && <div className={styles.emptyHint}>No projects yet.</div>}

        {projects.map((project) => {
          const projectConversations = conversations.filter((c) => c.projectId === project.id).sort(byRecency);
          const collapsed = collapsedProjects.has(project.id);
          const isRenamingProject = renamingProjectId === project.id;

          return (
            <div key={project.id} className={styles.projectGroup}>
              <div className={styles.projectHeader} onClick={() => toggleProjectCollapsed(project.id)}>
                <span className={styles.projectCaret}>{collapsed ? "▸" : "▾"}</span>
                {isRenamingProject ? (
                  <input
                    className={styles.renameInput}
                    value={projectRenameDraft}
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setProjectRenameDraft(e.target.value)}
                    onBlur={commitProjectRename}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitProjectRename();
                      if (e.key === "Escape") setRenamingProjectId(null);
                    }}
                  />
                ) : (
                  <span className={styles.projectName} title={project.name}>
                    {project.name}
                  </span>
                )}
                {!isRenamingProject && (
                  <span className={styles.projectActions} onClick={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      className={styles.iconButton}
                      title="New chat in project"
                      aria-label={`New chat in ${project.name}`}
                      onClick={() => onNewChat(project.id)}
                    >
                      +
                    </button>
                    <button
                      type="button"
                      className={`${styles.iconButton} ${project.instructions ? styles.iconButtonActive : ""}`}
                      title="Custom instructions"
                      aria-label={`Edit custom instructions for ${project.name}`}
                      onClick={() => toggleInstructionsEditor(project)}
                    >
                      🗒
                    </button>
                    <button
                      type="button"
                      className={styles.iconButton}
                      title="Rename project"
                      aria-label={`Rename project ${project.name}`}
                      onClick={() => startProjectRename(project)}
                    >
                      ✎
                    </button>
                    <button
                      type="button"
                      className={styles.iconButton}
                      title="Delete project"
                      aria-label={`Delete project ${project.name}`}
                      onClick={() => {
                        if (window.confirm(`Delete project "${project.name}"? Its chats will be ungrouped, not deleted.`)) {
                          onDeleteProject(project.id);
                        }
                      }}
                    >
                      ×
                    </button>
                  </span>
                )}
              </div>
              {editingInstructionsId === project.id && (
                <div className={styles.instructionsEditor} onClick={(e) => e.stopPropagation()}>
                  <textarea
                    className={styles.instructionsTextarea}
                    placeholder={`Custom instructions for ${project.name} (prepended for GPT-fallback replies in this project's chats)`}
                    value={instructionsDraft}
                    autoFocus
                    onChange={(e) => setInstructionsDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") setEditingInstructionsId(null);
                    }}
                  />
                  <div className={styles.instructionsActions}>
                    <button type="button" className="button buttonSecondary" onClick={() => setEditingInstructionsId(null)}>
                      Cancel
                    </button>
                    <button type="button" className="button" onClick={() => commitInstructions(project.id)}>
                      Save
                    </button>
                  </div>
                </div>
              )}
              {!collapsed && (
                <div className={styles.projectConversations}>
                  {projectConversations.length === 0 && <div className={styles.emptyHint}>No chats yet.</div>}
                  {projectConversations.map(renderConversationRow)}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionTitle}>Chats</span>
        </div>
        {ungrouped.length === 0 && <div className={styles.emptyHint}>No ungrouped chats yet.</div>}
        {ungrouped.map(renderConversationRow)}
      </div>
    </nav>
  );
}
