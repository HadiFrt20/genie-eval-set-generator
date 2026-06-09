import { Link } from 'react-router-dom'
import { useEffect, useMemo, useState } from 'react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { ExternalLink, RefreshCcw, BarChart3 } from 'lucide-react'
import { api, type RunStatus, type RunsIndexRow } from '@/api'
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
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { epochToDateTime } from '@/lib/utils'

function StateBadge({ run }: { run: RunStatus | undefined }) {
  if (!run) return <Badge variant="muted">…</Badge>
  const lc = run.life_cycle_state
  const rs = run.result_state
  if (lc === 'TERMINATED') {
    if (rs === 'SUCCESS') return <Badge variant="success">SUCCESS</Badge>
    if (rs === 'FAILED') return <Badge variant="danger">FAILED</Badge>
    if (rs === 'CANCELED') return <Badge variant="muted">CANCELED</Badge>
    return <Badge variant="warning">{rs || 'TERMINATED'}</Badge>
  }
  if (lc === 'RUNNING') return <Badge variant="info">RUNNING</Badge>
  if (lc === 'PENDING') return <Badge variant="warning">PENDING</Badge>
  if (lc === 'INTERNAL_ERROR') return <Badge variant="danger">INTERNAL_ERROR</Badge>
  return <Badge variant="muted">{lc}</Badge>
}

const LAST_CATALOG_KEY = 'lastEvalCatalog'

function readLastCatalog(): string {
  try {
    return localStorage.getItem(LAST_CATALOG_KEY) || ''
  } catch {
    return ''
  }
}

export default function RunsPage() {
  // Catalog input (persisted across sessions). User edits to switch view.
  const [catalog, setCatalog] = useState<string>(readLastCatalog)

  useEffect(() => {
    if (catalog) {
      try {
        localStorage.setItem(LAST_CATALOG_KEY, catalog)
      } catch {
        // ignore
      }
    }
  }, [catalog])

  // In-session zustand — used as a "just-submitted" overlay so the page feels
  // instant after a submit, even before the index round-trip resolves.
  const submittedThisSession = useRunsStore((s) => s.submitted)

  // Authoritative source: the UC runs-index for the chosen catalog.
  const indexQ = useQuery({
    queryKey: ['eval-runs-index', catalog],
    queryFn: () => api.evalRunsIndex(catalog, 50),
    enabled: catalog.length > 0,
    staleTime: 30_000,
  })

  // Merge: prefer index entries; for runs that exist in zustand but not yet in
  // the index (write delay or write-failure), display from zustand.
  const merged = useMemo(() => {
    type Row = {
      run_id: number
      run_name: string
      experiment_path: string
      submitted_at: string
      submitted_by?: string
    }
    const byId = new Map<number, Row>()
    for (const r of indexQ.data?.rows || []) {
      const id = Number(r.run_id)
      byId.set(id, {
        run_id: id,
        run_name: r.run_name,
        experiment_path: r.experiment_path,
        submitted_at: r.submitted_at,
        submitted_by: r.submitted_by,
      })
    }
    for (const r of submittedThisSession) {
      if (!byId.has(r.run_id)) {
        byId.set(r.run_id, {
          run_id: r.run_id,
          run_name: r.run_name,
          experiment_path: r.experiment_path,
          submitted_at: r.submitted_at,
          submitted_by: r.submitted_by,
        })
      }
    }
    return Array.from(byId.values()).sort((a, b) =>
      a.submitted_at < b.submitted_at ? 1 : -1,
    )
  }, [indexQ.data, submittedThisSession])

  // Live status for each merged run; refetch every 30s while any RUNNING.
  const queries = useQueries({
    queries: merged.map((r) => ({
      queryKey: ['run', r.run_id],
      queryFn: () => api.getRun(r.run_id),
      refetchInterval: (query: any) => {
        const data = query.state.data as RunStatus | undefined
        if (!data) return 15_000
        return data.life_cycle_state === 'TERMINATED' ? false : 30_000
      },
    })),
  })

  const statusByRun = useMemo(() => {
    const m = new Map<number, RunStatus>()
    queries.forEach((q, idx) => {
      if (q.data) m.set(merged[idx].run_id, q.data)
    })
    return m
  }, [queries, merged])

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Run status</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Reads the persistent runs index in UC. Auto-refreshes every 30 seconds while any run is RUNNING.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            indexQ.refetch()
            queries.forEach((q) => q.refetch())
          }}
        >
          <RefreshCcw className="mr-1 h-3.5 w-3.5" />
          Refresh now
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Catalog</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Label htmlFor="cat">UC catalog where the runs-index lives</Label>
          <div className="flex items-center gap-2">
            <Input
              id="cat"
              value={catalog}
              placeholder="e.g. main"
              onChange={(e) => setCatalog(e.target.value.trim())}
              className="max-w-sm"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => indexQ.refetch()}
              disabled={!catalog}
            >
              Load
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Reads <span className="font-mono">{catalog || '<catalog>'}.genie_eval.genie_eval_runs_index</span>.
            Switch to a different catalog to view runs from another deployment.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Eval runs</CardTitle>
        </CardHeader>
        <CardContent>
          {!catalog ? (
            <div className="rounded border border-dashed p-6 text-center text-sm text-muted-foreground">
              Set a catalog above to load the persistent runs index, or submit a
              run from the Configure page (the catalog will auto-populate).
            </div>
          ) : indexQ.isLoading ? (
            <div className="text-sm text-muted-foreground">Loading runs index…</div>
          ) : indexQ.isError ? (
            <div className="text-sm text-rose-600">
              {(indexQ.error as Error).message}
            </div>
          ) : merged.length === 0 ? (
            <div className="rounded border border-dashed p-6 text-center text-sm text-muted-foreground">
              No runs found for catalog <span className="font-mono">{catalog}</span>.
              Submit one from the Configure page.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run ID</TableHead>
                  <TableHead>Run name</TableHead>
                  <TableHead>State</TableHead>
                  <TableHead>Submitted</TableHead>
                  <TableHead>By</TableHead>
                  <TableHead>Experiment</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {merged.map((r) => {
                  const st = statusByRun.get(r.run_id)
                  return (
                    <TableRow key={r.run_id}>
                      <TableCell className="font-mono text-xs">{r.run_id}</TableCell>
                      <TableCell className="max-w-[260px] truncate text-sm">
                        {r.run_name}
                      </TableCell>
                      <TableCell>
                        <StateBadge run={st} />
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {r.submitted_at}
                      </TableCell>
                      <TableCell className="max-w-[160px] truncate text-xs text-muted-foreground">
                        {r.submitted_by || '—'}
                      </TableCell>
                      <TableCell className="max-w-[260px] truncate font-mono text-xs">
                        {r.experiment_path}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="inline-flex items-center gap-1">
                          {st?.run_page_url ? (
                            <Button
                              asChild
                              variant="ghost"
                              size="sm"
                              title="Open in Databricks"
                            >
                              <a href={st.run_page_url} target="_blank" rel="noreferrer">
                                <ExternalLink className="h-3.5 w-3.5" />
                              </a>
                            </Button>
                          ) : null}
                          <Button asChild size="sm" variant="outline">
                            <Link
                              to={`/scorecard/${r.run_id}?experiment_path=${encodeURIComponent(r.experiment_path)}`}
                            >
                              <BarChart3 className="mr-1 h-3.5 w-3.5" />
                              Scorecard
                            </Link>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
