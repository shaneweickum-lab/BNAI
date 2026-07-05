/**
 * Fetches a (potentially large) static asset with real download-progress
 * reporting, using the browser Cache API so repeat visits skip re-downloading
 * the 48MB model file entirely.
 *
 * This runs inside the Web Worker (workers/inference.worker.ts), which has
 * access to both `fetch` and `caches`.
 */

const CACHE_NAME = "benny-model-cache-v1";

export interface FetchProgress {
  loadedBytes: number;
  totalBytes: number;
  fromCache: boolean;
}

export async function fetchWithProgress(
  url: string,
  onProgress: (progress: FetchProgress) => void,
): Promise<ArrayBuffer> {
  const cache = await safeOpenCache();

  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      const buf = await cached.arrayBuffer();
      onProgress({ loadedBytes: buf.byteLength, totalBytes: buf.byteLength, fromCache: true });
      return buf;
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }

  const totalBytes = Number(response.headers.get("content-length")) || 0;

  // Cache a clone of the raw response before we consume the body, so the
  // stored entry is byte-for-byte what the server sent.
  if (cache) {
    try {
      await cache.put(url, response.clone());
    } catch {
      // Quota errors etc. are non-fatal -- we just won't cache this time.
    }
  }

  if (!response.body) {
    // Fallback for environments without a readable stream body.
    const buf = await response.arrayBuffer();
    onProgress({ loadedBytes: buf.byteLength, totalBytes: buf.byteLength || totalBytes, fromCache: false });
    return buf;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loadedBytes += value.byteLength;
      onProgress({ loadedBytes, totalBytes, fromCache: false });
    }
  }

  const merged = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return merged.buffer;
}

async function safeOpenCache(): Promise<Cache | null> {
  try {
    if (typeof caches === "undefined") return null;
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}
