export interface Sample {
  id: string;
  route_id: string;
  step_index: number;
  city: string;
  maneuver: string;
  instruction: string;
  split: "train" | "val" | "test" | null;
  frame: string | null;
  map: string | null;
  distances: { step_distance_m?: number; remaining_distance_m?: number };
  result_count: number;
}

export interface SampleDetail extends Sample {
  frames: string[];
  geometry: Record<string, unknown>;
  osm_road: Record<string, unknown>;
  prev_id: string | null;
  next_id: string | null;
}

export interface InferenceResult {
  id: string;
  model_id: string;
  modality: string;
  enhanced_instruction: string;
  lane_change_required: string | boolean;
  lanes_count: number | null;
  next_action: string;
  relevant_landmarks: string[];
  potential_hazards: string[];
  reasoning: string | null;
  augment: string | null;
  inference_metadata?: {
    latency_ms: number;
    tokens_in: number;
    tokens_out: number;
  };
}

export interface SampleResults {
  sample_id: string;
  results: InferenceResult[];
}

export interface CanonicalGt {
  sample_id: string;
  enhanced_instruction: string;
  next_action: string;
  next_action_human?: string;
  lane_change_required: boolean;
  lanes_count: number;
  relevant_landmarks: string[];
  potential_hazards: string[];
  acceptable_actions: string[];
}

export interface Model {
  display_name: string;
  model_id: string;
  provider: string;
  params_b: number;
  is_moe: boolean;
}

export interface Stats {
  total_samples: number;
  total_models: number;
  total_results: number;
  cities: { name: string; count: number }[];
}
