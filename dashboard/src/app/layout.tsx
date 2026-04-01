import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "NavBuddy-100 Viewer",
  description: "Browse NavBuddy-100 samples and benchmark results",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen">
        <div className="flex">
          {/* Sidebar */}
          <aside className="w-56 border-r border-[var(--color-border)] min-h-screen p-4 flex flex-col gap-6 shrink-0">
            <Link href="/" className="block">
              <h1 className="text-base font-semibold tracking-tight">NavBuddy-100</h1>
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">Sample Viewer</p>
            </Link>
            <nav className="flex flex-col gap-1">
              <Link
                href="/samples"
                className="px-3 py-1.5 rounded-md text-sm hover:bg-[var(--color-card)] transition-colors"
              >
                Samples
              </Link>
            </nav>
          </aside>
          {/* Main content */}
          <main className="flex-1 min-w-0 p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
