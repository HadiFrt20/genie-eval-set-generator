import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Loader2, Rocket } from 'lucide-react'
import { api, type SubmitRunRequest } from '@/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { useRunsStore } from '@/lib/runsStore'

interface FormState {
  uc_catalog: string
  uc_schema: string
  mlflow_experiment_path: string
  questions_per_table: string
  min_hard_per_table: string
  max_distinct_values_per_column: string
  extra_space_instructions: string
  stability_runs: string
  embedding_endpoint: string
  generator_endpoint: string
  judge_endpoint: string
  ai_gateway_base_url: string
  include_historical_in_context: boolean
  run_prompt_optimization: boolean
  run_name: string
  // v0.1.0 customization hooks
  extra_prompt_rules: string
  custom_prompt_uri: string
  custom_min_pooled_pass: string
  custom_max_pooled_pass: string
}

function pad(n: number) {
  return String(n).padStart(2, '0')
}
function timeStamp() {
  const d = new Date()
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`
}

const DEFAULT_GEN = 'databricks-llama-4-maverick'
const DEFAULT_JUDGE = 'databricks-claude-sonnet-4-5'
const DEFAULT_EMB = 'databricks-bge-large-en'

export default function ConfigurePage() {
  const [params] = useSearchParams()
  const spaceId = params.get('space_id') ?? ''
  const navigate = useNavigate()
  const addRun = useRunsStore((s) => s.add)

  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const me = useQuery({ queryKey: ['me'], queryFn: api.me })
  const detail = useQuery({
    queryKey: ['space', spaceId],
    queryFn: () => api.getSpace(spaceId),
    enabled: !!spaceId,
  })

  const tables = useMemo(() => {
    const t = (detail.data?.table_identifiers as string[] | undefined) ?? []
    return t
  }, [detail.data])

  const autoCatalog = tables[0]?.split('.')[0] ?? 'main'
  const host = health.data?.host ?? ''

  const [form, setForm] = useState<FormState | null>(null)

  // Initialize form once we have space details + health + the resolved user
  useEffect(() => {
    if (form || !spaceId || !me.data) return
    const user = me.data.user_name || 'shared'
    setForm({
      uc_catalog: autoCatalog,
      uc_schema: 'genie_eval',
      mlflow_experiment_path: `/Users/${user}/genie_eval_${spaceId.slice(0, 8)}`,
      questions_per_table: '14',
      min_hard_per_table: '5',
      max_distinct_values_per_column: '30',
      extra_space_instructions: '',
      stability_runs: '3',
      embedding_endpoint: DEFAULT_EMB,
      generator_endpoint: DEFAULT_GEN,
      judge_endpoint: DEFAULT_JUDGE,
      ai_gateway_base_url: host ? `${host}/serving-endpoints` : '/serving-endpoints',
      include_historical_in_context: false,
      run_prompt_optimization: false,
      run_name: `genie_eval_${spaceId.slice(0, 8)}_${timeStamp()}`,
      // v0.1.0 hooks default to empty / spec defaults
      extra_prompt_rules: '',
      custom_prompt_uri: '',
      custom_min_pooled_pass: '0.10',
      custom_max_pooled_pass: '0.90',
    })
  }, [autoCatalog, host, spaceId, form, me.data])

  // When health/detail land later, top up the catalog/gateway if user hasn't edited
  useEffect(() => {
    if (!form) return
    if (form.uc_catalog === 'main' && autoCatalog !== 'main') {
      setForm((f) => (f ? { ...f, uc_catalog: autoCatalog } : f))
    }
    if (
      form.ai_gateway_base_url === '/serving-endpoints' &&
      host
    ) {
      setForm((f) =>
        f ? { ...f, ai_gateway_base_url: `${host}/serving-endpoints` } : f,
      )
    }
  }, [autoCatalog, host, form])

  const submit = useMutation({
    mutationFn: (req: SubmitRunRequest) => api.submitRun(req),
    onSuccess: (run, vars) => {
      addRun(run)
      // Persist the catalog so /runs can query the runs-index next visit.
      try {
        localStorage.setItem('lastEvalCatalog', vars.uc_catalog)
      } catch {
        // ignore (private mode etc.)
      }
      navigate('/runs')
    },
  })

  if (!spaceId) {
    return (
      <Card>
        <CardContent className="p-5 text-sm text-muted-foreground">
          Pick a Genie space from the{' '}
          <a className="font-medium text-primary underline" href="/spaces">
            Spaces
          </a>{' '}
          page first.
        </CardContent>
      </Card>
    )
  }

  if (!form || detail.isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-8 w-1/3" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  const update = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((f) => (f ? { ...f, [k]: v } : f))

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form) return
    submit.mutate({
      run_name: form.run_name,
      uc_catalog: form.uc_catalog,
      uc_schema: form.uc_schema,
      genie_space_id: spaceId,
      genie_tables: '',
      mlflow_experiment_path: form.mlflow_experiment_path,
      generator_endpoint: form.generator_endpoint,
      judge_endpoint: form.judge_endpoint,
      ai_gateway_base_url: form.ai_gateway_base_url,
      questions_per_table: form.questions_per_table,
      min_hard_per_table: form.min_hard_per_table,
      max_distinct_values_per_column: form.max_distinct_values_per_column,
      extra_space_instructions: form.extra_space_instructions,
      embedding_endpoint: form.embedding_endpoint,
      stability_runs: form.stability_runs,
      include_historical_in_context: form.include_historical_in_context
        ? 'true'
        : 'false',
      run_prompt_optimization: form.run_prompt_optimization ? 'true' : 'false',
      // v0.1.0 hooks
      extra_prompt_rules: form.extra_prompt_rules,
      custom_prompt_uri: form.custom_prompt_uri,
      custom_min_pooled_pass: form.custom_min_pooled_pass,
      custom_max_pooled_pass: form.custom_max_pooled_pass,
    })
  }

  const title = (detail.data?.title as string) || '(untitled)'

  return (
    <form onSubmit={onSubmit} className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Configure run</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Submit a synthetic eval-set generation run for{' '}
          <span className="font-medium text-foreground">{title}</span>{' '}
          <span className="font-mono text-xs">({spaceId})</span>
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Space</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>
            <span className="text-muted-foreground">Tables registered:</span>{' '}
            <Badge variant="secondary">{tables.length}</Badge>
          </div>
          {tables.length > 0 ? (
            <div className="font-mono text-xs text-muted-foreground">
              {tables.slice(0, 5).join(', ')}
              {tables.length > 5 ? `, … (+${tables.length - 5} more)` : ''}
            </div>
          ) : null}
          {detail.data?.description ? (
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground">
                Space description
              </summary>
              <div className="mt-2 whitespace-pre-wrap rounded border bg-muted/40 p-3 leading-relaxed">
                {detail.data.description as string}
              </div>
            </details>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Output target</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="uc_catalog">UC catalog</Label>
            <Input
              id="uc_catalog"
              value={form.uc_catalog}
              onChange={(e) => update('uc_catalog', e.target.value)}
              required
            />
            <p className="text-xs text-muted-foreground">
              Where the eval-set + runs Delta tables will be written
            </p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="uc_schema">UC schema</Label>
            <Input
              id="uc_schema"
              value={form.uc_schema}
              onChange={(e) => update('uc_schema', e.target.value)}
              required
            />
          </div>
          <div className="space-y-1.5 md:col-span-2">
            <Label htmlFor="mlflow_path">MLflow experiment path</Label>
            <Input
              id="mlflow_path"
              value={form.mlflow_experiment_path}
              onChange={(e) => update('mlflow_experiment_path', e.target.value)}
              required
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Generation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="space-y-1.5">
              <Label htmlFor="qpt">questions_per_table</Label>
              <Input
                id="qpt"
                type="number"
                min={1}
                max={80}
                value={form.questions_per_table}
                onChange={(e) =>
                  update('questions_per_table', e.target.value)
                }
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="mhpt">min_hard_per_table</Label>
              <Input
                id="mhpt"
                type="number"
                min={0}
                value={form.min_hard_per_table}
                onChange={(e) =>
                  update('min_hard_per_table', e.target.value)
                }
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="mdvpc">max_distinct_values_per_column</Label>
              <Input
                id="mdvpc"
                type="number"
                min={1}
                max={100}
                value={form.max_distinct_values_per_column}
                onChange={(e) =>
                  update('max_distinct_values_per_column', e.target.value)
                }
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="extra">extra_space_instructions (optional)</Label>
            <Textarea
              id="extra"
              rows={3}
              value={form.extra_space_instructions}
              onChange={(e) =>
                update('extra_space_instructions', e.target.value)
              }
              placeholder="Free-form guidance to bias the question generator…"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Stability + realism</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="stab">stability_runs (M)</Label>
            <Select
              id="stab"
              value={form.stability_runs}
              onChange={(e) => update('stability_runs', e.target.value)}
            >
              {['1', '2', '3', '5'].map((v) => (
                <option key={v} value={v}>
                  {v}
                  {v === '3' ? ' — recommended' : v === '1' ? ' — skip stability' : ''}
                </option>
              ))}
            </Select>
            <p className="text-xs text-muted-foreground">
              3 = gateable scorecard. 1 = no stability axis.
            </p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="emb">embedding_endpoint</Label>
            <Input
              id="emb"
              value={form.embedding_endpoint}
              onChange={(e) => update('embedding_endpoint', e.target.value)}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Endpoints</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="gen">generator_endpoint</Label>
            <Input
              id="gen"
              value={form.generator_endpoint}
              onChange={(e) => update('generator_endpoint', e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="judge">judge_endpoint</Label>
            <Input
              id="judge"
              value={form.judge_endpoint}
              onChange={(e) => update('judge_endpoint', e.target.value)}
            />
          </div>
          <div className="space-y-1.5 md:col-span-2">
            <Label htmlFor="gw">ai_gateway_base_url</Label>
            <Input
              id="gw"
              value={form.ai_gateway_base_url}
              onChange={(e) => update('ai_gateway_base_url', e.target.value)}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Advanced</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Checkbox
            label="Inject historical_qa into prompt (off = avoid memorization)"
            checked={form.include_historical_in_context}
            onChange={(e) =>
              update('include_historical_in_context', e.target.checked)
            }
          />
          <Checkbox
            label="Run prompt optimization (GEPA Phase 7) — adds ~15min"
            checked={form.run_prompt_optimization}
            onChange={(e) =>
              update('run_prompt_optimization', e.target.checked)
            }
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Customization hooks</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            All optional. See{' '}
            <a
              className="underline"
              href="https://github.com/HadiFrt20/genie-eval-set-generator#key-parameters"
              target="_blank"
              rel="noreferrer"
            >
              README → Customization hooks
            </a>{' '}
            for details.
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="epr">extra_prompt_rules</Label>
            <Textarea
              id="epr"
              rows={3}
              value={form.extra_prompt_rules}
              onChange={(e) => update('extra_prompt_rules', e.target.value)}
              placeholder='e.g. "Never generate questions about claims past 2020."'
            />
            <p className="text-xs text-muted-foreground">
              Free-text rule appended after rule #8 in the question-gen prompt.
            </p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cpu">custom_prompt_uri</Label>
            <Input
              id="cpu"
              value={form.custom_prompt_uri}
              onChange={(e) => update('custom_prompt_uri', e.target.value)}
              placeholder="prompts:/your.uc.catalog.your_prompt@production"
            />
            <p className="text-xs text-muted-foreground">
              MLflow Prompt Registry URI to use instead of the default. Must define
              the same template variables as the built-in prompt — the notebook
              fail-fasts at startup if any are missing.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="cmnp">custom_min_pooled_pass</Label>
              <Input
                id="cmnp"
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={form.custom_min_pooled_pass}
                onChange={(e) => update('custom_min_pooled_pass', e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cmxp">custom_max_pooled_pass</Label>
              <Input
                id="cmxp"
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={form.custom_max_pooled_pass}
                onChange={(e) => update('custom_max_pooled_pass', e.target.value)}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Stability gate's degenerate-eval guard rails. Tighten or loosen the
            range that counts as "in-distribution" for <span className="font-mono">gateable=True</span>.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-3 p-5">
          <div className="space-y-1.5">
            <Label htmlFor="rn">Run name</Label>
            <Input
              id="rn"
              value={form.run_name}
              onChange={(e) => update('run_name', e.target.value)}
              required
            />
          </div>
          {submit.isError ? (
            <div className="rounded border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
              Submit failed: {(submit.error as Error).message}
            </div>
          ) : null}
          <Button
            type="submit"
            size="lg"
            className="w-full"
            disabled={submit.isPending}
          >
            {submit.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Rocket className="h-4 w-4" />
            )}
            Submit run
          </Button>
        </CardContent>
      </Card>
    </form>
  )
}
