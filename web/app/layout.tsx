import type { Metadata } from "next";
import "./globals.css";
import { MODEL_NAME, MODEL_SUBTITLE, formatParams, TOTAL_PARAMS } from "../lib/modelInfo";

export const metadata: Metadata = {
  title: `${MODEL_NAME} — ${MODEL_SUBTITLE}`,
  description: `Benny (BNAI V1.0) is a ${formatParams(TOTAL_PARAMS)}-parameter ternary-weight language model that runs 100% client-side in your browser via WebAssembly.`,
};

// Deliberately minimal: no Header/Footer here. The marketing pages (/ and
// /about) get the site chrome from app/(marketing)/layout.tsx; /demo is a
// full-viewport app shell with its own compact in-app top bar and renders
// directly inside this bare <body>.
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
