/**
 * Pure helpers backing the file-attachment control (components/chat/FileAttachment.tsx).
 *
 * The `accept` attribute below is only a UI hint -- browsers still let users
 * pick "All Files", and FileReader.readAsText() will happily "decode" a
 * binary file (image, PDF, ...) into mojibake rather than throwing. So the
 * real gate is `looksLikeDecodedText`, run on the actual decoded content
 * *after* reading, not the file extension/MIME type beforehand.
 */

export const FILE_INPUT_ACCEPT = ".txt,.md,.json,.csv,.log,text/plain,application/json";

export const UNSUPPORTED_FILE_MESSAGE =
  "This file doesn't look like plain text — Benny can only read text-based files, no images or PDFs.";

/**
 * Heuristic check that decoded file content is plausible plain text, not a
 * mis-decoded binary blob. Deliberately generic (not tied to file
 * extension/MIME type) so a differently-extensioned-but-genuinely-textual
 * file still passes, and a mislabeled binary file still gets caught.
 */
export function looksLikeDecodedText(content: string): boolean {
  if (content.length === 0) return true; // an empty file is valid (empty) text

  // U+FFFD (the replacement character) shows up wherever bytes couldn't be
  // decoded as valid text -- a strong, simple binary signal.
  if (content.includes("�")) return false;

  // Cap the scan cost on huge files; a few thousand characters is plenty to
  // tell prose/code/data from binary noise.
  const sampleLength = Math.min(content.length, 8000);
  let controlCount = 0;
  for (let i = 0; i < sampleLength; i++) {
    const code = content.charCodeAt(i);
    if (code === 0) return false; // NUL byte: unambiguous binary signal
    // Control characters other than tab/newline/CR are rare in real text
    // and common in binary data.
    if (code < 32 && code !== 9 && code !== 10 && code !== 13) controlCount += 1;
  }
  return controlCount / sampleLength < 0.02;
}

/**
 * Prefixes an attached file's content into a turn's message text, clearly
 * delimited, before it's handed to the existing dialogueManager/GPT
 * pipeline (which truncates oldest-turns-first on context overflow --
 * unchanged, no special-casing needed for large file content).
 */
export function buildAttachedMessage(fileName: string, fileContent: string, userMessage: string): string {
  return `[Attached: ${fileName}]\n${fileContent}\n\n${userMessage}`;
}
