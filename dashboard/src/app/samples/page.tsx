"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Sample, Stats } from "@/lib/types";
import { Search } from "lucide-react";

const CITIES_ALL = "all";
const MANEUVERS_ALL = "all";
const PER_PAGE = 25;

export default function SamplesPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [cityFilter, setCityFilter] = useState(CITIES_ALL);
  const [maneuverFilter, setManeuverFilter] = useState(MANEUVERS_ALL);
  const [page, setPage] = useState(0);

  useEffect(() => {
    Promise.all([api.samples(), api.stats()]).then(([s, st]) => {
      setSamples(s);
      setStats(st);
      setLoading(false);
    });
  }, []);

  const cities = useMemo(
    () => [...new Set(samples.map((s) => s.city))].sort(),
    [samples]
  );
  const maneuvers = useMemo(
    () => [...new Set(samples.map((s) => s.maneuver))].sort(),
    [samples]
  );

  const filtered = useMemo(() => {
    let out = samples;
    if (cityFilter !== CITIES_ALL)
      out = out.filter((s) => s.city === cityFilter);
    if (maneuverFilter !== MANEUVERS_ALL)
      out = out.filter((s) => s.maneuver === maneuverFilter);
    if (search) {
      const q = search.toLowerCase();
      out = out.filter(
        (s) =>
          s.id.toLowerCase().includes(q) ||
          s.instruction.toLowerCase().includes(q)
      );
    }
    return out;
  }, [samples, cityFilter, maneuverFilter, search]);

  const totalPages = Math.ceil(filtered.length / PER_PAGE);
  const paged = filtered.slice(page * PER_PAGE, (page + 1) * PER_PAGE);

  // Reset page when filters change
  useEffect(() => setPage(0), [cityFilter, maneuverFilter, search]);

  if (loading) {
    return (
      <div className="space-y-3">
        <div className="h-8 w-48 bg-[var(--color-card)] rounded animate-pulse" />
        {Array.from({ length: 10 }).map((_, i) => (
          <div
            key={i}
            className="h-10 bg-[var(--color-card)] rounded animate-pulse"
          />
        ))}
      </div>
    );
  }

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold">Samples</h1>
        {stats && (
          <p className="text-sm text-[var(--color-text-muted)] mt-1">
            {stats.total_samples} samples &middot; {stats.total_models} models
            &middot; {stats.total_results.toLocaleString()} results
          </p>
        )}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--color-text-muted)]" />
          <input
            type="text"
            placeholder="Search samples..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-[var(--color-card)] border border-[var(--color-border)] rounded-md focus:outline-none focus:border-[var(--color-accent)] placeholder:text-[var(--color-text-muted)]"
          />
        </div>
        <select
          value={cityFilter}
          onChange={(e) => setCityFilter(e.target.value)}
          className="px-3 py-1.5 text-sm bg-[var(--color-card)] border border-[var(--color-border)] rounded-md focus:outline-none focus:border-[var(--color-accent)]"
        >
          <option value={CITIES_ALL}>All cities</option>
          {cities.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <select
          value={maneuverFilter}
          onChange={(e) => setManeuverFilter(e.target.value)}
          className="px-3 py-1.5 text-sm bg-[var(--color-card)] border border-[var(--color-border)] rounded-md focus:outline-none focus:border-[var(--color-accent)]"
        >
          <option value={MANEUVERS_ALL}>All maneuvers</option>
          {maneuvers.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <span className="text-xs text-[var(--color-text-muted)]">
          {filtered.length} of {samples.length}
        </span>
      </div>

      {/* Table */}
      <div className="border border-[var(--color-border)] rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border)] bg-[var(--color-card)]">
              <th className="text-left px-4 py-2.5 font-medium text-[var(--color-text-muted)]">
                Sample
              </th>
              <th className="text-left px-4 py-2.5 font-medium text-[var(--color-text-muted)]">
                City
              </th>
              <th className="text-left px-4 py-2.5 font-medium text-[var(--color-text-muted)]">
                Maneuver
              </th>
              <th className="text-left px-4 py-2.5 font-medium text-[var(--color-text-muted)]">
                Instruction
              </th>
              <th className="text-left px-4 py-2.5 font-medium text-[var(--color-text-muted)]">
                Split
              </th>
              <th className="text-right px-4 py-2.5 font-medium text-[var(--color-text-muted)]">
                Results
              </th>
            </tr>
          </thead>
          <tbody>
            {paged.map((s) => (
              <tr
                key={s.id}
                className="border-b border-[var(--color-border)] hover:bg-[var(--color-card-hover)] transition-colors"
              >
                <td className="px-4 py-2">
                  <Link
                    href={`/samples/${s.id}`}
                    className="font-mono text-xs text-[var(--color-accent)] hover:underline"
                  >
                    {s.id.replace(/^[a-z]+_/, "").slice(0, 24)}
                  </Link>
                </td>
                <td className="px-4 py-2 text-[var(--color-text-muted)] capitalize">
                  {s.city}
                </td>
                <td className="px-4 py-2">
                  <span className="inline-block px-1.5 py-0.5 text-xs bg-[var(--color-card)] border border-[var(--color-border)] rounded font-mono">
                    {s.maneuver}
                  </span>
                </td>
                <td className="px-4 py-2 max-w-sm truncate text-[var(--color-text-muted)]">
                  {s.instruction}
                </td>
                <td className="px-4 py-2">
                  {s.split && (
                    <span
                      className={`inline-block px-1.5 py-0.5 text-[10px] font-medium rounded ${
                        s.split === "test"
                          ? "bg-blue-500/15 text-blue-400"
                          : s.split === "val"
                            ? "bg-yellow-500/15 text-yellow-400"
                            : "bg-zinc-500/15 text-zinc-400"
                      }`}
                    >
                      {s.split}
                    </span>
                  )}
                </td>
                <td className="px-4 py-2 text-right font-mono text-xs text-[var(--color-text-muted)]">
                  {s.result_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-3 py-1.5 text-sm bg-[var(--color-card)] border border-[var(--color-border)] rounded-md hover:bg-[var(--color-card-hover)] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-xs text-[var(--color-text-muted)]">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="px-3 py-1.5 text-sm bg-[var(--color-card)] border border-[var(--color-border)] rounded-md hover:bg-[var(--color-card-hover)] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
