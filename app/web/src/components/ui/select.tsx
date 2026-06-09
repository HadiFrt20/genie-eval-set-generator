import * as React from 'react'
import { cn } from '@/lib/utils'

// Lightweight native select with shadcn-ish styling. We deliberately avoid
// pulling in @radix-ui/react-select to keep the bundle small — we don't need
// virtualization or async option loading for the few choices we have.
export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      'flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm',
      'focus:outline-none focus:ring-1 focus:ring-ring',
      'disabled:cursor-not-allowed disabled:opacity-50',
      className,
    )}
    {...props}
  >
    {children}
  </select>
))
Select.displayName = 'Select'
