import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fantasy Football Toolkit — start/sit",
  description:
    "Fantasy football projections from a trained sequence model, explained by an LLM grounded in retrieved news.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
