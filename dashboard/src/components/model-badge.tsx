"use client";

import { providerColor, providerFromModelId, formatModelName } from "@/lib/models";

export function ModelBadge({
  modelId,
  className = "",
}: {
  modelId: string;
  className?: string;
}) {
  const provider = providerFromModelId(modelId);
  const color = providerColor(provider);

  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className="inline-block w-2 h-2 rounded-full shrink-0"
        style={{ backgroundColor: color }}
      />
      <span className="font-mono text-xs">{formatModelName(modelId)}</span>
    </span>
  );
}
