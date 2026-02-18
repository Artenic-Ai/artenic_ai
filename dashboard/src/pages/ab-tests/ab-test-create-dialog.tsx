import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select } from "@/components/ui/input";

interface ABTestCreateDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ABTestCreateDialog({
  open,
  onClose,
}: ABTestCreateDialogProps) {
  const [name, setName] = useState("");
  const [service, setService] = useState("sentiment");
  const [metric, setMetric] = useState("accuracy");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onClose();
    setName("");
    setService("sentiment");
    setMetric("accuracy");
  }

  return (
    <Dialog open={open} onClose={onClose} title="Create A/B Test">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="model-a-vs-model-b"
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
          ]}
        />
        <Select
          label="Primary Metric"
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          options={[
            { value: "accuracy", label: "Accuracy" },
            { value: "f1_score", label: "F1 Score" },
            { value: "entity_f1", label: "Entity F1" },
            { value: "auc", label: "AUC" },
            { value: "latency", label: "Latency" },
          ]}
        />
        <Input
          label="Min Samples"
          type="number"
          defaultValue="5000"
          placeholder="5000"
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
