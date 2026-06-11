import { useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'
import { api } from '@/api'
import { useRunsStore } from '@/lib/runsStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { formatNumber, formatPercent, cn } from '@/lib/utils'

function MetricTile({
  label,
  value,
  hint,
  className,
}: {
  label: string
  value: React.ReactNode
  hint?: string
  className?: string
}) {
  return (
    <div className={cn('rounded-lg border bg-card p-4', className)}>
      <div className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {hint ? (
        <div className="mt-1 text-xs text-muted-foreground">{hint}</div>
      ) : null}
    </div>
  )
}

export default function ScorecardPage() {
  const { run_id: runIdParam } = useParams()
  const [search] = useSearchParams()
  const submitted = useRunsStore((s) => s.submitted)

  const runId = runIdParam ? Number(runIdParam) : submitted[0]?.run_id ?? null
  const submittedRecord = useMemo(
    () => submitted.find((r) => r.run_id === runId),
    [submitted, runId],
  )
  const expPathFromQuery = search.get('experiment_path')
  const expPath = expPathFromQuery ?? submittedRecord?.experiment_path ?? ''

  const sc = useQuery({
    queryKey: ['scorecard', runId, expPath],
    queryFn: () => api.scorecard(runId!, expPath),
    enabled: !!runId && !!expPath,
    refetchInterval: (query: any) => {
      const data = query.state.data
      if (data?.ready) return false
      if (data?.life_cycle_state && data.life_cycle_state !== 'TERMINATED')
        return 30_000
      return 15_000
    },
  })

  const [pageIdx, setPageIdx] = useState(0)
  const PAGE = 25
  const params = submittedRecord?.params
  const evalSet = useQuery({
    queryKey: ['eval-set', params?.uc_catalog, params?.uc_schema],
    queryFn: () =>
      api.evalSet(params!.uc_catalog as string, params!.uc_schema as string, 200),
    enabled: !!params?.uc_catalog && !!params?.uc_schema && !!sc.data?.ready,
  })

  if (!runId) {
    return (
      <Card>
        <CardContent className="p-5 text-sm text-muted-foreground">
          No run selected. Submit a run from the{' '}
          <a className="text-primary underline" href="/spaces">
            Spaces
          </a>{' '}
          page or pick one from the{' '}
          <a className="text-primary underline" href="/runs">
            Runs
          </a>{' '}
          page.
        </CardContent>
      </Card>
    )
  }

  if (!expPath) {
    return (
      <Card>
        <CardContent className="p-5 text-sm text-muted-foreground">
          Missing <code>experiment_path</code>. Open the scorecard from the Runs
          page or include it in the URL.
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Eval-set scorecard
        </h1>
        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span>
            Run: <code className="font-mono text-foreground">{runId}</code>
          </span>
          <span>
            Experiment:{' '}
            <code className="font-mono text-foreground">{expPath}</code>
          </span>
        </div>
      </div>

      {sc.isLoading ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-lg" />
          ))}
        </div>
      ) : sc.isError ? (
        <Card>
          <CardContent className="p-5 text-sm text-rose-600">
            Failed to load: {(sc.error as Error).message}
          </CardContent>
        </Card>
      ) : !sc.data?.ready ? (
        <Card>
          <CardContent className="space-y-3 p-5">
            <div className="flex items-center gap-2 text-sm">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <span className="font-medium">Scorecard not ready</span>
            </div>
            <p className="text-sm text-muted-foreground">
              {sc.data?.reason ??
                'The notebook run is still in progress or scorecard runs have not been written to MLflow yet.'}
            </p>
            {sc.data?.life_cycle_state ? (
              <div className="text-xs text-muted-foreground">
                Life cycle:{' '}
                <Badge variant="muted">{sc.data.life_cycle_state}</Badge>
                {sc.data.result_state ? (
                  <>
                    {' · '}Result:{' '}
                    <Badge variant="muted">{sc.data.result_state}</Badge>
                  </>
                ) : null}
              </div>
            ) : null}
            <Button size="sm" variant="outline" onClick={() => sc.refetch()}>
              Check again
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          {sc.data.attribution === 'latest' ? (
            <Card>
              <CardContent className="flex items-center gap-2 p-3 text-xs text-amber-700">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>
                  Showing the <span className="font-medium">latest</span> scorecard runs in this
                  experiment — they could not be attributed to this specific job run (older
                  notebook version or interactive run). If you submitted multiple runs to the
                  same experiment, verify in MLflow.
                </span>
              </CardContent>
            </Card>
          ) : null}
          <ScorecardBody data={sc.data.scorecard!} />
        </>
      )}

      {sc.data?.ready && params ? (
        <Card>
          <CardHeader>
            <CardTitle>Generated eval set</CardTitle>
            <p className="text-xs text-muted-foreground">
              From{' '}
              <code className="font-mono">
                {params.uc_catalog}.{params.uc_schema}.genie_eval_set
              </code>
            </p>
          </CardHeader>
          <CardContent>
            {evalSet.isLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : evalSet.isError ? (
              <div className="text-sm text-rose-600">
                {(evalSet.error as Error).message}
              </div>
            ) : !evalSet.data?.ok ? (
              <div className="text-sm text-amber-600">
                {evalSet.data?.error ?? 'Could not read table.'}
              </div>
            ) : evalSet.data.rows.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                Eval set is empty.
              </div>
            ) : (
              <EvalSetTable
                rows={evalSet.data.rows}
                columns={evalSet.data.columns}
                pageIdx={pageIdx}
                setPageIdx={setPageIdx}
                pageSize={PAGE}
              />
            )}
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}

function ScorecardBody({
  data,
}: {
  data: NonNullable<ReturnType<typeof useQuery<any, any>>['data']> | any
}) {
  const d = data.diversity ?? {}
  const rel = data.reliability ?? {}
  const ver = data.verification ?? {}
  const verComputed = ver.computed === true || String(ver.computed) === 'true'
  const q = data.quality ?? {}
  const reg = data.regression ?? {}
  const diversityComputed = d.computed !== false
  const relComputed = rel.computed === true || String(rel.computed) === 'true'
  const leakage = String(d.leakage_risk).toLowerCase() === 'true'
  const gateableTrue = String(rel.gateable).toLowerCase() === 'true'

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricTile
          label="Leakage risk"
          value={
            !diversityComputed ? (
              <Badge variant="muted" className="text-base">
                —
              </Badge>
            ) : leakage ? (
              <Badge variant="danger" className="text-base">
                leakage
              </Badge>
            ) : (
              <Badge variant="success" className="text-base">
                ok
              </Badge>
            )
          }
          hint={
            !diversityComputed
              ? 'diversity not computed'
              : `mean NN cos ${formatNumber(d.mean_nn_cosine)}`
          }
        />
        <MetricTile
          label="Gateable"
          value={
            !relComputed ? (
              <span className="text-base text-muted-foreground">not computed</span>
            ) : (
              <span className="inline-flex items-center gap-1.5">
                {gateableTrue ? (
                  <CheckCircle2 className="h-5 w-5 text-emerald-600" />
                ) : (
                  <XCircle className="h-5 w-5 text-rose-600" />
                )}
                <span>{String(rel.gateable)}</span>
              </span>
            )
          }
          hint={!relComputed ? 'set stability_runs ≥ 2' : undefined}
        />
        {verComputed ? (
          <MetricTile
            label="Verified pairs"
            value={formatPercent(ver.verified_fraction)}
            hint="execution-verified, two model families"
          />
        ) : null}
        <MetricTile
          label="Row concordance"
          value={formatPercent(reg.pass_rate_deterministic)}
          hint={verComputed ? 'on verified pairs only' : 'lower bound — vs synthetic SQL'}
        />
        <MetricTile label="SQL executes" value={formatPercent(q.sql_executes)} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Trust gate — answer-key verification</CardTitle>
          <p className="text-xs text-muted-foreground">
            k independent solvers across two model families re-derive each answer blind (with the
            space's declared semantics) and execution results are compared. Certificates, not
            accuracy. Quarantined pairs are kept + flagged and excluded from the regression.
          </p>
        </CardHeader>
        <CardContent>
          {!verComputed ? (
            <p className="text-sm text-muted-foreground">
              Not computed — set <code>verification_k</code> ≥ 2 to enable the trust gate.
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricTile
                label="Verified"
                value={formatPercent(ver.verified_fraction)}
                hint={`of ${ver.n_gated ?? '—'} gated pairs`}
              />
              <MetricTile
                label="Gold / Verified"
                value={`${ver.gold_count ?? '—'} / ${ver.verified_count ?? '—'}`}
                hint="unanimous / majority"
              />
              <MetricTile
                label="Quarantine"
                value={String(ver.quarantine_count ?? '—')}
                hint="review queue (excluded from regression)"
              />
              <MetricTile
                label="Repaired"
                value={String(ver.repaired_count ?? '—')}
                hint={`mean agreement ${formatNumber(ver.mean_panel_agreement, 2) ?? '—'}`}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Diversity &amp; leakage</CardTitle>
          <p className="text-xs text-muted-foreground">
            Descriptive embedding numbers — not a verdict. The only flag is leakage
            (synthetic questions near-verbatim to reference questions).
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {!diversityComputed ? (
            <p className="text-sm text-muted-foreground">
              Not computed — no embedding endpoint, or the reference corpus is below the
              minimum.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <MetricTile
                  label="mean NN cosine"
                  value={formatNumber(d.mean_nn_cosine)}
                  hint="≈1.0 = copies"
                />
                <MetricTile
                  label="NN cosine p10"
                  value={formatNumber(d.nn_cosine_p10)}
                  hint="low = outliers"
                />
                <MetricTile
                  label="gen self-sim"
                  value={formatNumber(d.gen_self_similarity)}
                  hint="lower = diverse"
                />
                <MetricTile
                  label="ref self-sim"
                  value={formatNumber(d.ref_self_similarity)}
                  hint="context only"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                n_reference = {d.n_reference ?? '—'} ({d.n_history ?? '—'} history +{' '}
                {d.n_curated ?? '—'} curated)
                {leakage
                  ? ' · ⚠ leakage risk: synthetic set is near-verbatim to reference'
                  : ''}
              </p>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Reliability</CardTitle>
          <p className="text-xs text-muted-foreground">
            Test-retest agreement across reruns + precision of the concordance rate (Wilson
            CI on N questions, not N×M cells).
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {!relComputed ? (
            <p className="text-sm text-muted-foreground">
              Not computed — set <code>stability_runs</code> ≥ 2 to measure reliability.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <MetricTile
                  label="Rerun agreement"
                  value={formatNumber(rel.rerun_agreement)}
                  hint="need ≥ 0.90"
                />
                <MetricTile
                  label="CI half-width"
                  value={`${formatNumber(rel.half_pp, 2)}pp`}
                  hint={
                    rel.gate_max_half_width_pp
                      ? `target ≤ ${formatNumber(rel.gate_max_half_width_pp, 1)}pp`
                      : undefined
                  }
                />
                <MetricTile label="Concordance" value={formatPercent(rel.pooled)} />
                <MetricTile
                  label="M reruns / N"
                  value={`${rel.m_reruns ?? '—'} / ${rel.n_questions ?? '—'}`}
                />
              </div>
              {!gateableTrue && rel.n_needed_for_target ? (
                <p className="text-xs text-muted-foreground">
                  To reach the half-width target at this concordance you need ~
                  <span className="font-medium text-foreground">
                    {rel.n_needed_for_target}
                  </span>{' '}
                  questions (raise <code>questions_per_table</code>).
                </p>
              ) : null}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Eval-set quality</CardTitle>
          <p className="text-xs text-muted-foreground">
            <code>sql_executes</code> / <code>nonempty</code> are direct checks. clarity /
            sql_answers / grounded are <span className="font-medium">uncalibrated</span>{' '}
            LLM/heuristic scores — read as opinions, not measurements.
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricTile label="question_clarity *" value={formatNumber(q.clarity, 2)} />
            <MetricTile label="sql_answers_q *" value={formatNumber(q.sql_answers, 2)} />
            <MetricTile label="grounded_literals †" value={formatNumber(q.grounded, 2)} />
            <MetricTile label="nonempty" value={formatNumber(q.nonempty, 2)} />
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            * uncalibrated LLM judge · † deterministic heuristic (single-quoted literals only)
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Row concordance (lower bound)</CardTitle>
          <p className="text-xs text-muted-foreground">
            Descriptive per-bucket profiles — NOT calibration/discrimination metrics. Buckets
            with n &lt; 5 are noise; treat them as indicative only.
          </p>
          {reg.n_same_engine != null || reg.n_indeterminate != null ? (
            <p className="text-xs text-muted-foreground">
              {reg.n_same_engine != null && reg.n_questions != null
                ? `${reg.n_same_engine}/${reg.n_questions} expected rows executed same-engine as Genie · `
                : ''}
              {reg.n_indeterminate != null
                ? `${reg.n_indeterminate} indeterminate (row cap)`
                : ''}
              {reg.difficulty_monotonic != null
                ? ` · difficulty profile ${
                    String(reg.difficulty_monotonic).toLowerCase() === 'true'
                      ? 'monotonic'
                      : 'non-monotonic'
                  }`
                : ''}
            </p>
          ) : null}
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.keys(reg.by_difficulty ?? {}).length > 0 ? (
            <ProfileTable
              title="By difficulty"
              label="Difficulty"
              order={['easy', 'medium', 'hard']}
              rates={reg.by_difficulty}
              counts={reg.by_difficulty_n ?? {}}
            />
          ) : null}
          {Object.keys(reg.by_category ?? {}).length > 0 ? (
            <ProfileTable
              title="By category"
              label="Category"
              rates={reg.by_category}
              counts={reg.by_category_n ?? {}}
            />
          ) : null}
        </CardContent>
      </Card>
    </div>
  )
}

function ProfileTable({
  title,
  label,
  rates,
  counts,
  order,
}: {
  title: string
  label: string
  rates: Record<string, number>
  counts: Record<string, number>
  order?: string[]
}) {
  const keys = order
    ? order.filter((k) => rates[k] !== undefined)
    : Object.keys(rates).sort((a, b) => (rates[b] as number) - (rates[a] as number))
  return (
    <div>
      <div className="mb-2 text-sm font-medium">{title}</div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{label}</TableHead>
            <TableHead className="text-right">n</TableHead>
            <TableHead className="text-right">Concordance</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {keys.map((k) => {
            const n = counts[k]
            const small = typeof n === 'number' && n < 5
            return (
              <TableRow key={k}>
                <TableCell className="capitalize">{k}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {n ?? '—'}
                  {small ? (
                    <AlertTriangle className="ml-1 inline h-3 w-3 text-amber-600" />
                  ) : null}
                </TableCell>
                <TableCell
                  className={cn(
                    'text-right tabular-nums',
                    small && 'text-muted-foreground',
                  )}
                >
                  {formatPercent(rates[k])}
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
}

function EvalSetTable({
  rows,
  columns,
  pageIdx,
  setPageIdx,
  pageSize,
}: {
  rows: Array<Record<string, unknown>>
  columns: string[]
  pageIdx: number
  setPageIdx: (n: number) => void
  pageSize: number
}) {
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize))
  const slice = rows.slice(pageIdx * pageSize, (pageIdx + 1) * pageSize)
  return (
    <div className="space-y-3">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((c) => (
              <TableHead key={c}>{c}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {slice.map((row, i) => (
            <TableRow key={i}>
              {columns.map((c) => (
                <TableCell
                  key={c}
                  className={cn(
                    'align-top text-xs',
                    c === 'expected_sql' || c === 'question'
                      ? 'max-w-[420px] whitespace-pre-wrap break-words font-mono'
                      : 'whitespace-nowrap',
                  )}
                >
                  {formatCell(row[c])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          Showing {pageIdx * pageSize + 1}–
          {Math.min((pageIdx + 1) * pageSize, rows.length)} of {rows.length}
        </span>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={pageIdx === 0}
            onClick={() => setPageIdx(pageIdx - 1)}
          >
            Prev
          </Button>
          <span>
            {pageIdx + 1} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={pageIdx + 1 >= totalPages}
            onClick={() => setPageIdx(pageIdx + 1)}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  )
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (typeof v === 'string') return v
  return JSON.stringify(v)
}
