import { type ReactNode, useMemo, useState } from "react";

import { ChevronDown, ChevronUp } from "lucide-react";

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  sortable?: boolean;
  sortValue?: (row: T) => string | number;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyFn: (row: T) => string;
  onRowClick?: (row: T) => void;
  pageSize?: number;
  emptyMessage?: string;
}

export function DataTable<T>({
  columns,
  data,
  keyFn,
  onRowClick,
  pageSize = 10,
  emptyMessage = "No data.",
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(0);

  const sorted = useMemo(() => {
    if (!sortKey) return data;
    const col = columns.find((c) => c.key === sortKey);
    if (!col?.sortValue) return data;
    const fn = col.sortValue;
    return [...data].sort((a, b) => {
      const av = fn(a);
      const bv = fn(b);
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir, columns]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const paged = sorted.slice(page * pageSize, (page + 1) * pageSize);

  function toggleSort(key: string) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
    setPage(0);
  }

  if (data.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-sm text-text-muted">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="min-w-[600px] w-full text-left text-sm">
          <thead>
            <tr className="border-b border-border">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-4 py-3 text-xs font-medium text-text-secondary ${col.className ?? ""} ${col.sortable ? "cursor-pointer select-none hover:text-text-primary" : ""}`}
                  onClick={col.sortable ? () => toggleSort(col.key) : undefined}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.header}
                    {col.sortable && sortKey === col.key && (
                      sortDir === "asc" ? <ChevronUp size={12} /> : <ChevronDown size={12} />
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row) => (
              <tr
                key={keyFn(row)}
                className={`border-b border-border transition-colors hover:bg-surface-2 ${onRowClick ? "cursor-pointer" : ""}`}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-4 py-3 ${col.className ?? ""}`}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex flex-col items-center justify-between gap-2 border-t border-border px-4 py-3 text-xs text-text-muted sm:flex-row">
          <span>
            {page * pageSize + 1}–{Math.min((page + 1) * pageSize, sorted.length)}{" "}
            of {sorted.length}
          </span>
          <div className="flex gap-1">
            <button
              className="rounded px-2 py-1 hover:bg-surface-3 disabled:opacity-40"
              onClick={() => setPage((p) => p - 1)}
              disabled={page === 0}
            >
              Prev
            </button>
            <button
              className="rounded px-2 py-1 hover:bg-surface-3 disabled:opacity-40"
              onClick={() => setPage((p) => p + 1)}
              disabled={page >= totalPages - 1}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
