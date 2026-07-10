"use client";

/**
 * The 3-region multi-conversation app shell for /demo: a left drawer
 * (chats/projects), a center chat column, and a right drawer (engineering
 * showcase), each independently collapsible via a fixed edge tab -- the
 * "pull out tab" pattern -- plus a compact in-app top bar with explicit
 * toggle buttons for the same two drawers.
 */

import Link from "next/link";
import { useState } from "react";
import styles from "./AppShell.module.css";
import { MODEL_NAME, MODEL_SUBTITLE } from "../../lib/modelInfo";

interface AppShellProps {
  sidebar: React.ReactNode;
  metrics: React.ReactNode;
  children: React.ReactNode;
}

export default function AppShell({ sidebar, metrics, children }: AppShellProps) {
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(false);

  return (
    <div className={styles.shell}>
      <button
        type="button"
        className={`${styles.edgeTab} ${styles.edgeTabLeft} ${leftOpen ? styles.edgeTabOpen : ""}`}
        onClick={() => setLeftOpen((v) => !v)}
        aria-label={leftOpen ? "Collapse chats panel" : "Expand chats panel"}
        aria-expanded={leftOpen}
      >
        ☰ Chats
      </button>

      <aside className={`${styles.drawer} ${styles.leftDrawer} ${leftOpen ? styles.drawerOpen : ""}`}>
        <div className={styles.drawerInner}>{sidebar}</div>
      </aside>

      <div className={styles.main}>
        <div className={styles.topBar}>
          <Link href="/" className={styles.brand}>
            <span>{MODEL_NAME}</span>
            <span className={styles.brandSubtitle}>{MODEL_SUBTITLE}</span>
          </Link>
          <div className={styles.topBarToggles}>
            <button
              type="button"
              className={`${styles.toggleButton} ${leftOpen ? styles.toggleButtonActive : ""}`}
              onClick={() => setLeftOpen((v) => !v)}
              aria-expanded={leftOpen}
            >
              ☰ Chats
            </button>
            <button
              type="button"
              className={`${styles.toggleButton} ${rightOpen ? styles.toggleButtonActive : ""}`}
              onClick={() => setRightOpen((v) => !v)}
              aria-expanded={rightOpen}
            >
              📊 Engineering
            </button>
          </div>
        </div>
        <div className={styles.content}>{children}</div>
      </div>

      <aside className={`${styles.drawer} ${styles.rightDrawer} ${rightOpen ? styles.drawerOpen : ""}`}>
        <div className={styles.drawerInner}>{metrics}</div>
      </aside>

      <button
        type="button"
        className={`${styles.edgeTab} ${styles.edgeTabRight} ${rightOpen ? styles.edgeTabOpen : ""}`}
        onClick={() => setRightOpen((v) => !v)}
        aria-label={rightOpen ? "Collapse engineering panel" : "Expand engineering panel"}
        aria-expanded={rightOpen}
      >
        📊 Engineering
      </button>
    </div>
  );
}
