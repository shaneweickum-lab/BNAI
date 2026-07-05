import type { Metadata } from "next";
import "./globals.css";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { MODEL_NAME, MODEL_SUBTITLE } from "../lib/modelInfo";

export const metadata: Metadata = {
  title: `${MODEL_NAME} — ${MODEL_SUBTITLE}`,
  description:
    "Benny (BNAI V1.0) is a 75M-parameter ternary-weight language model that runs 100% client-side in your browser via WebAssembly.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Header />
        <main style={{ flex: 1 }}>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
