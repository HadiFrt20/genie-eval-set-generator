import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { FlaskConical } from 'lucide-react'
import { api } from '@/api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import SpacesPage from '@/pages/SpacesPage'
import ConfigurePage from '@/pages/ConfigurePage'
import RunsPage from '@/pages/RunsPage'
import ScorecardPage from '@/pages/ScorecardPage'

function NavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
          isActive
            ? 'bg-secondary text-secondary-foreground'
            : 'text-muted-foreground hover:bg-secondary/60 hover:text-foreground',
        )
      }
    >
      {label}
    </NavLink>
  )
}

function HealthIndicator() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 60_000,
  })
  if (isLoading) return <Badge variant="muted">…</Badge>
  if (isError || !data) return <Badge variant="danger">unreachable</Badge>
  if (!data.authed) return <Badge variant="warning">no auth</Badge>
  const workspace = data.host.replace(/^https?:\/\//, '').split('.')[0]
  return (
    <Badge variant="success" className="font-mono">
      {workspace}
    </Badge>
  )
}

export default function App() {
  const loc = useLocation()
  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur">
        <div className="container flex h-14 items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <FlaskConical className="h-4 w-4" />
            </div>
            <div>
              <div className="text-sm font-semibold leading-none">
                Genie Eval Generator
              </div>
              <div className="mt-0.5 text-xs text-muted-foreground">
                Synthetic NL→SQL eval sets for Genie spaces
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-1">
            <NavItem to="/spaces" label="Spaces" />
            <NavItem
              to={`/configure${loc.pathname.startsWith('/configure') ? loc.search : ''}`}
              label="Configure"
            />
            <NavItem to="/runs" label="Runs" />
            <NavItem to="/scorecard" label="Scorecard" />
          </nav>
          <HealthIndicator />
        </div>
      </header>
      <main className="container flex-1 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/spaces" replace />} />
          <Route path="/spaces" element={<SpacesPage />} />
          <Route path="/configure" element={<ConfigurePage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/scorecard" element={<ScorecardPage />} />
          <Route path="/scorecard/:run_id" element={<ScorecardPage />} />
          <Route path="*" element={<Navigate to="/spaces" replace />} />
        </Routes>
      </main>
      <footer className="border-t py-3 text-center text-xs text-muted-foreground">
        Frontend over the{' '}
        <code className="font-mono text-[11px]">genie_eval_set_generator</code>{' '}
        notebook
      </footer>
    </div>
  )
}
