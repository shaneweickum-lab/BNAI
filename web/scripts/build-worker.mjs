// Bundles workers/inference.worker.ts (plus everything it imports from
// lib/ and worker/) into a single, plain-JS classic Web Worker script at
// public/workers/inference.worker.js.
//
// Why a separate esbuild step instead of Next's bundler: Next 16's default
// Turbopack build does not yet reliably compile the standard
// `new Worker(new URL("./worker.ts", import.meta.url))` pattern -- it was
// observed to copy the raw, untranspiled .ts source into .next/static/media
// verbatim, which is not valid JS a browser can execute as a worker script.
// Pre-bundling with esbuild into a real static asset under public/ sidesteps
// that entirely and works the same in dev and prod.
//
// Run via `npm run build:worker` (also wired into `predev`/`prebuild`).

import { build } from "esbuild";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");

await build({
  entryPoints: [path.join(projectRoot, "workers", "inference.worker.ts")],
  outfile: path.join(projectRoot, "public", "workers", "inference.worker.js"),
  bundle: true,
  format: "iife",
  target: "es2020",
  platform: "browser",
  sourcemap: true,
  logLevel: "info",
});
