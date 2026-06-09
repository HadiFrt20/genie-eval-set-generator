import * as React from 'react'
import { cn } from '@/lib/utils'

interface CheckboxProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
}

export const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, id, ...props }, ref) => {
    const inputId = id ?? React.useId()
    return (
      <div className="flex items-center gap-2">
        <input
          ref={ref}
          id={inputId}
          type="checkbox"
          className={cn(
            'h-4 w-4 rounded border-input text-primary focus:ring-1 focus:ring-ring',
            className,
          )}
          {...props}
        />
        {label ? (
          <label htmlFor={inputId} className="text-sm font-medium leading-none">
            {label}
          </label>
        ) : null}
      </div>
    )
  },
)
Checkbox.displayName = 'Checkbox'
