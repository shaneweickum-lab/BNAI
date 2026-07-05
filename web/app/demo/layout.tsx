import type { Metadata } from "next";
import { MODEL_NAME } from "../../lib/modelInfo";

export const metadata: Metadata = {
  title: `Demo — ${MODEL_NAME}`,
  description: "Chat with Benny (BNAI V1.0) entirely client-side, in your browser, via WebAssembly.",
};

export default function DemoLayout({ children }: { children: React.ReactNode }) {
  return children;
}
