/* Adapted from Nageoffer Ragent frontend/src/components/ui/table.tsx under Apache-2.0. */
import { forwardRef, type HTMLAttributes } from 'react';

export const Table = forwardRef<HTMLTableElement, HTMLAttributes<HTMLTableElement>>(
  ({className, ...props}, ref) => (
    <div className="ui-table-wrap">
      <table ref={ref} className={['ui-table', className].filter(Boolean).join(' ')} {...props} />
    </div>
  )
);
Table.displayName = 'Table';
