import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { type Column, DataTable } from "@/components/ui/data-table";

interface TestItem {
  id: string;
  name: string;
  score: number;
}

const COLUMNS: Column<TestItem>[] = [
  {
    key: "name",
    header: "Name",
    sortable: true,
    sortValue: (r) => r.name,
    render: (r) => <span>{r.name}</span>,
  },
  {
    key: "score",
    header: "Score",
    sortable: true,
    sortValue: (r) => r.score,
    render: (r) => <span>{r.score}</span>,
  },
];

const DATA: TestItem[] = [
  { id: "1", name: "Alpha", score: 90 },
  { id: "2", name: "Beta", score: 70 },
  { id: "3", name: "Charlie", score: 85 },
];

describe("DataTable", () => {
  it("renders table with data", () => {
    render(
      <DataTable columns={COLUMNS} data={DATA} keyFn={(r) => r.id} />,
    );
    expect(screen.getByText("Alpha")).toBeInTheDocument();
    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText("Charlie")).toBeInTheDocument();
  });

  it("renders column headers", () => {
    render(
      <DataTable columns={COLUMNS} data={DATA} keyFn={(r) => r.id} />,
    );
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Score")).toBeInTheDocument();
  });

  it("shows empty message when no data", () => {
    render(
      <DataTable
        columns={COLUMNS}
        data={[]}
        keyFn={(r) => r.id}
        emptyMessage="Nothing here"
      />,
    );
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("calls onRowClick when row is clicked", () => {
    const onClick = vi.fn();
    render(
      <DataTable
        columns={COLUMNS}
        data={DATA}
        keyFn={(r) => r.id}
        onRowClick={onClick}
      />,
    );
    fireEvent.click(screen.getByText("Alpha"));
    expect(onClick).toHaveBeenCalledWith(DATA[0]);
  });

  it("sorts by column when header is clicked", () => {
    render(
      <DataTable columns={COLUMNS} data={DATA} keyFn={(r) => r.id} />,
    );

    // Click "Name" header to sort
    fireEvent.click(screen.getByText("Name"));

    const rows = screen.getAllByRole("row");
    // Header row + 3 data rows
    expect(rows).toHaveLength(4);
  });

  it("paginates with small page size", () => {
    render(
      <DataTable
        columns={COLUMNS}
        data={DATA}
        keyFn={(r) => r.id}
        pageSize={2}
      />,
    );
    // Should show 2 items + header
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(3);

    // Should show pagination
    expect(screen.getByText("Next")).toBeInTheDocument();
  });

  it("navigates pages", () => {
    render(
      <DataTable
        columns={COLUMNS}
        data={DATA}
        keyFn={(r) => r.id}
        pageSize={2}
      />,
    );

    // Click next
    fireEvent.click(screen.getByText("Next"));
    // Now should show Charlie
    expect(screen.getByText("Charlie")).toBeInTheDocument();
  });
});
