import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ledger — Legal Research Workbench",
  description:
    "Ask questions across Acts, Judgments, Tax Documents, and legal opinions — every answer traced to its exact page.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-ink text-parchment">
        {children}
      </body>
    </html>
  );
}
