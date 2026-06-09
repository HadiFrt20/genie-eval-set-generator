import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowRight,
  Database,
  MessageSquare,
  Star,
  TableProperties,
} from 'lucide-react'
import { api, type SpaceListItem } from '@/api'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

function CountBadge({
  spaceId,
  kind,
  icon: Icon,
}: {
  spaceId: string
  kind: 'curated' | 'conversations'
  icon: React.ComponentType<{ className?: string }>
}) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['space-count', spaceId, kind],
    queryFn: () =>
      kind === 'curated'
        ? api.curatedCount(spaceId)
        : api.conversationCount(spaceId),
    staleTime: 5 * 60_000,
  })
  return (
    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
      <Icon className="h-3.5 w-3.5" />
      {isLoading ? (
        <Skeleton className="h-3 w-6" />
      ) : isError ? (
        '—'
      ) : (
        (data?.count ?? 0)
      )}
      <span className="text-[10px] uppercase tracking-wide">
        {kind === 'curated' ? 'curated' : 'conversations'}
      </span>
    </span>
  )
}

function SpaceCard({ s }: { s: SpaceListItem }) {
  const navigate = useNavigate()
  return (
    <Card
      className="group relative cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5"
      onClick={() =>
        navigate(`/configure?space_id=${encodeURIComponent(s.space_id)}`)
      }
    >
      <CardContent className="space-y-3 p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold leading-tight">
              {s.title}
            </div>
            <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
              {s.space_id}
            </div>
          </div>
          <Badge variant="secondary" className="shrink-0">
            <TableProperties className="mr-1 h-3 w-3" />
            {s.n_tables}
          </Badge>
        </div>
        <p className="line-clamp-3 min-h-[3.6em] text-xs leading-snug text-muted-foreground">
          {s.description || (
            <span className="italic">No description</span>
          )}
        </p>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t pt-3">
          <CountBadge spaceId={s.space_id} kind="curated" icon={Star} />
          <CountBadge
            spaceId={s.space_id}
            kind="conversations"
            icon={MessageSquare}
          />
          {s.warehouse_id ? (
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <Database className="h-3.5 w-3.5" />
              <span className="truncate font-mono text-[11px]">
                {s.warehouse_id.slice(0, 12)}…
              </span>
            </span>
          ) : null}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="absolute right-3 top-3 opacity-0 transition-opacity group-hover:opacity-100"
        >
          Configure
          <ArrowRight className="ml-1 h-3.5 w-3.5" />
        </Button>
      </CardContent>
    </Card>
  )
}

export default function SpacesPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['spaces'],
    queryFn: api.listSpaces,
  })

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Genie spaces in this workspace
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Pick a space to generate an eval set against. Counts populate live
          from the Genie REST API.
        </p>
      </div>

      {isError ? (
        <Card>
          <CardContent className="p-5 text-sm text-rose-600">
            Failed to load spaces: {(error as Error).message}
          </CardContent>
        </Card>
      ) : null}

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i}>
              <CardContent className="space-y-3 p-5">
                <Skeleton className="h-5 w-2/3" />
                <Skeleton className="h-3 w-1/2" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : null}

      {data && data.length === 0 ? (
        <Card>
          <CardContent className="p-5 text-sm text-muted-foreground">
            No Genie spaces found in this workspace.
          </CardContent>
        </Card>
      ) : null}

      {data && data.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.map((s) => (
            <SpaceCard key={s.space_id} s={s} />
          ))}
        </div>
      ) : null}
    </div>
  )
}
