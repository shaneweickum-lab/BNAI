import styles from "./Footer.module.css";
import { MODEL_NAME, MODEL_SUBTITLE } from "../lib/modelInfo";

const REPO_URL = "https://github.com/shaneweickum-lab/bnai";

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <span>
          {MODEL_NAME} &middot; {MODEL_SUBTITLE} &middot; runs entirely in your browser
        </span>
        <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
          Source on GitHub
        </a>
      </div>
    </footer>
  );
}
