const PROVIDER_COLORS: Record<string, string> = {
  openai: "#10a37f",
  google: "#4285F4",
  anthropic: "#D97757",
  "x-ai": "#EF4444",
  qwen: "#a855f7",
  Qwen: "#a855f7",
};

export function providerColor(provider: string): string {
  return PROVIDER_COLORS[provider] || "#71717a";
}

export function providerFromModelId(modelId: string): string {
  const prefix = modelId.split("/")[0] || "";
  if (prefix === "openai") return "openai";
  if (prefix === "google") return "google";
  if (prefix === "anthropic") return "anthropic";
  if (prefix === "x-ai") return "x-ai";
  if (prefix === "qwen" || prefix === "Qwen") return "qwen";
  return prefix;
}

export function formatModelName(modelId: string): string {
  // Strip provider prefix and common suffixes
  const name = modelId.split("/").pop() || modelId;
  return name
    .replace(/-instruct$/i, "")
    .replace(/-preview$/i, "")
    .replace(/-beta$/i, "");
}
