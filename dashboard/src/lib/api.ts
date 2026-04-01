import type {
  Sample,
  SampleDetail,
  SampleResults,
  CanonicalGt,
  Model,
  Stats,
} from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  stats: () => get<Stats>("/api/stats"),
  samples: () => get<Sample[]>("/api/samples"),
  sample: (id: string) => get<SampleDetail>(`/api/samples/${id}`),
  sampleResults: (id: string) =>
    get<SampleResults>(`/api/samples/${id}/results`),
  canonicalGt: (id: string) => get<CanonicalGt>(`/api/canonical-gt/${id}`),
  models: () => get<Model[]>("/api/models"),
};

/** Strip leading directory prefix (e.g. "frames/foo.jpg" -> "foo.jpg") */
function basename(path: string): string {
  return path.split("/").pop() || path;
}

export function frameUrl(path: string): string {
  return `/api/frames/${basename(path)}`;
}

export function mapUrl(path: string): string {
  return `/api/maps/${basename(path)}`;
}
