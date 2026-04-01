"use client";

import { useEffect, useState, useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, frameUrl, mapUrl } from "@/lib/api";
import { ModelBadge } from "@/components/model-badge";
import { providerColor } from "@/lib/models";
import type {
  SampleDetail,
  SampleResults,
  CanonicalGt,
  InferenceResult,
} from "@/lib/types";
import {
  ChevronLeft,
  ChevronRight,
  MapPin,
  ArrowLeft,
} from "lucide-react";

const MODALITIES = [
  "all",
  "video + prior",
  "image + prior",
  "prior",
  "augment + prior",
] as const;

function ActionBadge({
  action,
  gtAction,
  acceptable,
}: {
  action: string;
  gtAction?: string;
  acceptable?: string[];
}) {
  const isMatch =
    action === gtAction || (acceptable && acceptable.includes(action));
  const bg = isMatch
    ? "bg-green-500/15 text-green-400"
    : gtAction
      ? "bg-red-500/15 text-red-400"
      : "bg-zinc-500/15 text-zinc-400";

  return (
    <span
      className={`inline-block px-1.5 py-0.5 text-[11px] font-mono rounded ${bg}`}
    >
      {action}
    </span>
  );
}

function LcBadge({
  value,
  gtValue,
}: {
  value: string | boolean;
  gtValue?: boolean;
}) {
  const normalized =
    typeof value === "boolean"
      ? value
      : String(value).toLowerCase() === "yes";
  const isMatch = gtValue === undefined ? true : normalized === gtValue;
  const bg = isMatch
    ? "bg-zinc-500/15 text-zinc-300"
    : "bg-red-500/15 text-red-400";

  return (
    <span className={`inline-block px-1.5 py-0.5 text-[11px] font-mono rounded ${bg}`}>
      LC: {normalized ? "yes" : "no"}
    </span>
  );
}

export default function SampleDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [sample, setSample] = useState<SampleDetail | null>(null);
  const [results, setResults] = useState<SampleResults | null>(null);
  const [gt, setGt] = useState<CanonicalGt | null>(null);
  const [modality, setModality] = useState<string>("all");
  const [filterCompany, setFilterCompany] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.sample(id),
      api.sampleResults(id),
      api.canonicalGt(id).catch(() => null),
    ]).then(([s, r, g]) => {
      setSample(s);
      setResults(r);
      setGt(g);
      setLoading(false);
    });
  }, [id]);

  // Unique companies present in results
  const presentCompanies = useMemo(() => {
    if (!results) return [];
    const companies = new Set<string>();
    for (const r of results.results) {
      companies.add(r.model_id.split("/")[0].toLowerCase());
    }
    return [...companies].sort();
  }, [results]);

  // Group results by model, filter by modality and company
  const groupedResults = useMemo(() => {
    if (!results) return [];
    let filtered = results.results;
    if (modality !== "all") {
      filtered = filtered.filter((r) => r.modality === modality);
    }
    if (filterCompany) {
      filtered = filtered.filter(
        (r) => r.model_id.split("/")[0].toLowerCase() === filterCompany
      );
    }
    // Group by model_id
    const byModel = new Map<string, InferenceResult[]>();
    for (const r of filtered) {
      const key = r.model_id;
      if (!byModel.has(key)) byModel.set(key, []);
      byModel.get(key)!.push(r);
    }
    return [...byModel.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [results, modality, filterCompany]);

  // Count results per modality
  const modalityCounts = useMemo(() => {
    if (!results) return {};
    const counts: Record<string, number> = {};
    for (const r of results.results) {
      counts[r.modality] = (counts[r.modality] || 0) + 1;
    }
    return counts;
  }, [results]);

  if (loading) {
    return (
      <div className="space-y-4 max-w-5xl">
        <div className="h-6 w-64 bg-[var(--color-card)] rounded animate-pulse" />
        <div className="grid grid-cols-2 gap-4">
          <div className="aspect-video bg-[var(--color-card)] rounded-lg animate-pulse" />
          <div className="aspect-video bg-[var(--color-card)] rounded-lg animate-pulse" />
        </div>
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-24 bg-[var(--color-card)] rounded-lg animate-pulse" />
        ))}
      </div>
    );
  }

  if (!sample) {
    return (
      <div className="text-[var(--color-text-muted)]">Sample not found.</div>
    );
  }

  const lastFrame = sample.frames[sample.frames.length - 1];

  return (
    <div className="max-w-5xl">
      {/* Navigation */}
      <div className="flex items-center justify-between mb-4">
        <Link
          href="/samples"
          className="inline-flex items-center gap-1.5 text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Samples
        </Link>
        <div className="flex items-center gap-2">
          {sample.prev_id && (
            <Link
              href={`/samples/${sample.prev_id}`}
              className="p-1.5 rounded-md border border-[var(--color-border)] hover:bg-[var(--color-card)] transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </Link>
          )}
          {sample.next_id && (
            <Link
              href={`/samples/${sample.next_id}`}
              className="p-1.5 rounded-md border border-[var(--color-border)] hover:bg-[var(--color-card)] transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </Link>
          )}
        </div>
      </div>

      {/* Sample header */}
      <div className="mb-6">
        <h1 className="text-lg font-semibold font-mono">{sample.id}</h1>
        <div className="flex items-center gap-3 mt-1.5 text-sm text-[var(--color-text-muted)]">
          <span className="capitalize">{sample.city}</span>
          <span>&middot;</span>
          <span className="font-mono text-xs px-1.5 py-0.5 bg-[var(--color-card)] border border-[var(--color-border)] rounded">
            {sample.maneuver}
          </span>
          {sample.split && (
            <>
              <span>&middot;</span>
              <span className="text-xs">{sample.split}</span>
            </>
          )}
          {sample.distances?.step_distance_m && (
            <>
              <span>&middot;</span>
              <span>{sample.distances.step_distance_m}m</span>
            </>
          )}
        </div>
      </div>

      {/* Images */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {lastFrame && (
          <div className="rounded-lg overflow-hidden border border-[var(--color-border)]">
            <img
              src={frameUrl(lastFrame)}
              alt="Dashcam frame"
              className="w-full aspect-video object-cover"
            />
          </div>
        )}
        {sample.map && (
          <div className="rounded-lg overflow-hidden border border-[var(--color-border)]">
            <img
              src={mapUrl(sample.map)}
              alt="Overhead map"
              className="w-full aspect-video object-cover"
            />
          </div>
        )}
      </div>

      {/* Navigation instruction + GT */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
          <div className="text-[11px] font-medium text-[var(--color-text-muted)] uppercase tracking-wider mb-2">
            Navigation Instruction
          </div>
          <p className="text-sm leading-relaxed">{sample.instruction}</p>
        </div>
        {gt && (
          <div className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
            <div className="text-[11px] font-medium text-[var(--color-text-muted)] uppercase tracking-wider mb-2">
              Ground Truth
            </div>
            <p className="text-sm leading-relaxed mb-3">
              {gt.enhanced_instruction}
            </p>
            <div className="flex flex-wrap gap-1.5">
              <ActionBadge action={gt.next_action} />
              <LcBadge value={gt.lane_change_required} />
              <span className="inline-block px-1.5 py-0.5 text-[11px] font-mono rounded bg-zinc-500/15 text-zinc-300">
                {gt.lanes_count} lane{gt.lanes_count !== 1 ? "s" : ""}
              </span>
              {gt.relevant_landmarks.map((l, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[11px] rounded bg-blue-500/10 text-blue-400"
                >
                  <MapPin className="w-2.5 h-2.5" />
                  {l}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Model Results */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold">
            Model Results
            <span className="text-[var(--color-text-muted)] font-normal ml-2 text-sm">
              {results?.results.length || 0} total
            </span>
          </h2>
          <div className="flex items-center gap-3">
            {/* Modality tabs */}
            <div className="flex items-center gap-1">
              {MODALITIES.map((m) => {
                const count =
                  m === "all"
                    ? results?.results.length || 0
                    : modalityCounts[m] || 0;
                if (m !== "all" && count === 0) return null;
                return (
                  <button
                    key={m}
                    onClick={() => setModality(m)}
                    className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
                      modality === m
                        ? "bg-[var(--color-accent)] border-[var(--color-accent)] text-white"
                        : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:bg-[var(--color-card)]"
                    }`}
                  >
                    {m === "all" ? "All" : m.replace(" + prior", "")}
                    <span className="ml-1 opacity-60">{count}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
        {/* Company filter chips */}
        {presentCompanies.length > 1 && (
          <div className="flex flex-wrap gap-1 mt-3">
            <button
              onClick={() => setFilterCompany("")}
              className={`px-2 py-0.5 rounded text-xs border transition-colors ${
                filterCompany === ""
                  ? "bg-[var(--color-accent)]/15 border-[var(--color-accent)] text-[var(--color-text)]"
                  : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-accent)]"
              }`}
            >
              All
            </button>
            {presentCompanies.map((co) => {
              const active = filterCompany === co;
              const color = providerColor(co);
              return (
                <button
                  key={co}
                  onClick={() => setFilterCompany(active ? "" : co)}
                  className={`px-2 py-0.5 rounded text-xs border transition-colors flex items-center gap-1.5 ${
                    active
                      ? "bg-[var(--color-accent)]/10 border-[var(--color-accent)] text-[var(--color-text)]"
                      : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-accent)]"
                  }`}
                >
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  {co}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Result cards */}
      <div className="space-y-2">
        {groupedResults.map(([modelId, modelResults]) => (
          <div key={modelId}>
            {modelResults.map((r, i) => (
              <div
                key={`${r.model_id}-${r.modality}-${i}`}
                className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] mb-2"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <ModelBadge modelId={r.model_id} />
                    <span className="text-[11px] text-[var(--color-text-muted)] font-mono">
                      {r.modality}
                    </span>
                    {r.augment && (
                      <span className="text-[11px] text-yellow-400 font-mono">
                        {r.augment}
                      </span>
                    )}
                  </div>
                  {r.inference_metadata && (
                    <span className="text-[11px] text-[var(--color-text-muted)] font-mono">
                      {(r.inference_metadata.latency_ms / 1000).toFixed(1)}s
                    </span>
                  )}
                </div>
                <p className="text-sm leading-relaxed mb-2">
                  {r.enhanced_instruction}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  <ActionBadge
                    action={r.next_action}
                    gtAction={gt?.next_action}
                    acceptable={gt?.acceptable_actions}
                  />
                  <LcBadge
                    value={r.lane_change_required}
                    gtValue={gt?.lane_change_required}
                  />
                  {r.lanes_count != null && (
                    <span
                      className={`inline-block px-1.5 py-0.5 text-[11px] font-mono rounded ${
                        gt && r.lanes_count === gt.lanes_count
                          ? "bg-zinc-500/15 text-zinc-300"
                          : gt
                            ? "bg-red-500/15 text-red-400"
                            : "bg-zinc-500/15 text-zinc-300"
                      }`}
                    >
                      {r.lanes_count} lane{r.lanes_count !== 1 ? "s" : ""}
                    </span>
                  )}
                  {r.relevant_landmarks.slice(0, 4).map((l, i) => (
                    <span
                      key={i}
                      className="inline-block px-1.5 py-0.5 text-[11px] rounded bg-zinc-500/10 text-[var(--color-text-muted)]"
                    >
                      {l}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}

        {groupedResults.length === 0 && (
          <div className="text-center py-12 text-[var(--color-text-muted)] text-sm">
            No results for this modality.
          </div>
        )}
      </div>
    </div>
  );
}
