import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select } from "@/components/ui/input";

interface BudgetCreateDialogProps {
  open: boolean;
  onClose: () => void;
}

export function BudgetCreateDialog({
  open,
  onClose,
}: BudgetCreateDialogProps) {
  const [scope, setScope] = useState("global");
  const [scopeValue, setScopeValue] = useState("*");
  const [period, setPeriod] = useState("monthly");
  const [limit, setLimit] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onClose();
    setScope("global");
    setScopeValue("*");
    setPeriod("monthly");
    setLimit("");
  }

  return (
    <Dialog open={open} onClose={onClose} title="Create Budget Rule">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Select
          label="Scope"
          value={scope}
          onChange={(e) => setScope(e.target.value)}
          options={[
            { value: "global", label: "Global" },
            { value: "provider", label: "Provider" },
            { value: "category", label: "Category" },
            { value: "service", label: "Service" },
          ]}
        />
        <Input
          label="Scope Value"
          value={scopeValue}
          onChange={(e) => setScopeValue(e.target.value)}
          placeholder="* (all)"
          required
        />
        <Select
          label="Period"
          value={period}
          onChange={(e) => setPeriod(e.target.value)}
          options={[
            { value: "daily", label: "Daily" },
            { value: "weekly", label: "Weekly" },
            { value: "monthly", label: "Monthly" },
          ]}
        />
        <Input
          label="Limit (EUR)"
          type="number"
          value={limit}
          onChange={(e) => setLimit(e.target.value)}
          placeholder="5000"
          required
        />
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="ghost" type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit">Create</Button>
        </div>
      </form>
    </Dialog>
  );
}
