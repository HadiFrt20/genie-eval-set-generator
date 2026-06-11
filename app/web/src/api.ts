// Typed fetch client for the FastAPI backend.
// All endpoints are relative — Vite proxies /api → :8000 in dev,
// in production the API is same-origin (FastAPI serves the static build).

export interface HealthResponse {
  host: string
  authed: boolean
  notebook_path: string
  obo_dev_fallback?: boolean
  // bundle-provided default catalog (--var=output_catalog), used as the form's uc_catalog default
  default_output_catalog?: string
}

export interface SpaceListItem {
  space_id: string
  title: string
  description: string
  n_tables: number
  warehouse_id: string
  table_identifiers: string[]
}

export interface SpaceDetail {
  id?: string
  title?: string
  description?: string
  table_identifiers?: string[]
  warehouse_id?: string
  [key: string]: unknown
}

export interface SubmitRunRequest {
  run_name: string
  uc_catalog: string
  uc_schema: string
  genie_space_id: string
  genie_tables: string
  mlflow_experiment_path: string
  generator_endpoint: string
  judge_endpoint: string
  ai_gateway_base_url: string
  questions_per_table: string
  min_hard_per_table: string
  max_distinct_values_per_column: string
  extra_space_instructions: string
  embedding_endpoint: string
  stability_runs: string
  include_historical_in_context: string
  run_prompt_optimization: string
  // v0.1.0 customization hooks
  extra_prompt_rules?: string
  custom_prompt_uri?: string
  custom_min_pooled_pass?: string
  custom_max_pooled_pass?: string
}

export interface SubmittedRun {
  run_id: number
  run_name: string
  experiment_path: string
  submitted_at: string
  submitted_by?: string
  params: Record<string, string>
  index_status?: string
}

export interface RunsIndexRow {
  run_id: number | string
  run_name: string
  space_id: string
  experiment_path: string
  submitted_at: string
  submitted_by: string
  params_json: string
}

export interface RunsIndexResponse {
  ok: boolean
  warehouse_id?: string
  table?: string
  rows: RunsIndexRow[]
}

export interface RunStatus {
  run_id: number
  run_name: string
  life_cycle_state: string
  result_state: string
  state_message?: string
  start_time?: number | null
  end_time?: number | null
  run_page_url?: string
}

export interface ScorecardData {
  // Descriptive embedding numbers (was "realism"). Not a verdict; only flag is leakage_risk.
  diversity: {
    mean_nn_cosine: number | null
    nn_cosine_p10: number | null
    gen_self_similarity: number | null
    ref_self_similarity: number | null
    leakage_risk: string | null
    computed: boolean
    n_history: string | null
    n_curated: string | null
    n_reference: string | null
  }
  // Test-retest reliability + precision (was "stability"; Krippendorff α removed).
  reliability: {
    rerun_agreement: number | null
    half_pp: number | null
    pooled: number | null
    gateable: string | null
    computed: boolean
    m_reruns: string | null
    n_questions: string | null
    n_units: number | null
    n_needed_for_target: number | null
    gate_max_half_width_pp: number | null
  }
  quality: {
    sql_executes: number | null
    nonempty: number | null
    clarity: number | null
    sql_answers: number | null
    grounded: number | null
  }
  // pass_rate_deterministic is row concordance vs synthetic SQL — a LOWER BOUND, not correctness.
  regression: {
    pass_rate_deterministic: number | null
    by_difficulty: Record<string, number>
    by_category: Record<string, number>
    by_difficulty_n: Record<string, number>
    by_category_n: Record<string, number>
    n_questions: number | null
    n_indeterminate: number | null
    n_same_engine: number | null
    difficulty_monotonic: string | null
    difficulty_label_match_rate: number | null
  }
}

export interface ScorecardResponse {
  ready: boolean
  reason?: string
  life_cycle_state?: string
  result_state?: string
  available_sections?: string[]
  // 'exact' = MLflow runs tagged with this job run_id; 'latest' = untagged fallback
  // (newest runs in the experiment — may belong to a different submission).
  attribution?: string
  scorecard: ScorecardData | null
}

export interface EvalSetResponse {
  ok: boolean
  warehouse_id?: string
  warehouse_name?: string
  table?: string
  columns: string[]
  rows: Array<Record<string, unknown>>
  error?: string
}

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!r.ok) {
    let detail = `HTTP ${r.status}`
    try {
      const j = await r.json()
      detail = j.detail ?? detail
    } catch {
      try {
        detail = await r.text()
      } catch {
        // ignore
      }
    }
    throw new ApiError(detail, r.status)
  }
  return r.json() as Promise<T>
}

export const api = {
  health: () => http<HealthResponse>('/api/health'),
  me: () => http<{ user_name: string }>('/api/me'),
  listSpaces: () => http<SpaceListItem[]>('/api/spaces'),
  getSpace: (id: string) => http<SpaceDetail>(`/api/spaces/${id}`),
  curatedCount: (id: string) =>
    http<{ count: number }>(`/api/spaces/${id}/curated-count`),
  conversationCount: (id: string) =>
    http<{ count: number }>(`/api/spaces/${id}/conversation-count`),
  submitRun: (body: SubmitRunRequest) =>
    http<SubmittedRun>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  getRun: (id: number) => http<RunStatus>(`/api/runs/${id}`),
  listRuns: (limit = 25) => http<RunStatus[]>(`/api/runs?limit=${limit}`),
  scorecard: (runId: number, experimentPath: string) =>
    http<ScorecardResponse>(
      `/api/scorecard/${runId}?experiment_path=${encodeURIComponent(experimentPath)}`,
    ),
  evalSet: (catalog: string, schema: string, limit = 200) =>
    http<EvalSetResponse>(
      `/api/eval-set?catalog=${encodeURIComponent(catalog)}&schema=${encodeURIComponent(
        schema,
      )}&limit=${limit}`,
    ),
  evalRunsIndex: (catalog: string, limit = 50) =>
    http<RunsIndexResponse>(
      `/api/eval-runs-index?catalog=${encodeURIComponent(catalog)}&limit=${limit}`,
    ),
}

export { ApiError }
