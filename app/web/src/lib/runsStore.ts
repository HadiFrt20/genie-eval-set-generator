// In-session store of runs the user submitted. Survives page navigation, not
// browser refresh — same lifetime as the Streamlit st.session_state list.
import { create } from 'zustand'
import type { SubmittedRun } from '@/api'

interface RunsState {
  submitted: SubmittedRun[]
  add: (r: SubmittedRun) => void
  clear: () => void
}

export const useRunsStore = create<RunsState>((set) => ({
  submitted: [],
  add: (r) => set((s) => ({ submitted: [r, ...s.submitted].slice(0, 50) })),
  clear: () => set({ submitted: [] }),
}))
