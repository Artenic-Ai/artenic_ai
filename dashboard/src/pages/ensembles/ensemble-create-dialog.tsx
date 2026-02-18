import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select } from "@/components/ui/input";

interface EnsembleCreateDialogProps {
  open: boolean;
  onClose: () => void;
}

export function EnsembleCreateDialog({
  open,
  onClose,
}: EnsembleCreateDialogProps) {
  const [name, setName] = useState("");
  const [service, setService] = useState("sentiment");
  const [strategy, setStrategy] = useState("weighted_average");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onClose();
    setName("");
    setService("sentiment");
    setStrategy("weighted_average");
  }

  return (
    <Dialog open={open} onClose={onClose} title="Create Ensemble">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-ensemble"
          required
        />
        <Select
          label="Service"
          value={service}
          onChange={(e) => setService(e.target.value)}
          options={[
            { value: "sentiment", label: "Sentiment" },
            { value: "ner", label: "NER" },
            { value: "vision", label: "Vision" },
            { value: "fraud", label: "Fraud Detection" },
            { value: "detection", label: "Object Detection" },
          ]}
        />
        <Select
          label="Strategy"
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          options={[
            { value: "weighted_average", label: "Weighted Average" },
            { value: "simple_average", label: "Simple Average" },
            { value: "stacking", label: "Stacking" },
            { value: "cascade", label: "Cascade" },
            { value: "voting", label: "Majority Voting" },
          ]}
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
