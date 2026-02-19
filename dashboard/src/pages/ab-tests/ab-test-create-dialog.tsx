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
  const [minSamples, setMinSamples] = useState("5000");
  const [errors, setErrors] = useState<Record<string, string>>({});

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const newErrors: Record<string, string> = {};
    if (!name.trim()) {
      newErrors.name = "Name is required";
    }
    if (Number(minSamples) <= 0) {
      newErrors.min_samples = "Min samples must be greater than 0";
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setErrors({});
    onClose();
    setName("");
    setService("sentiment");
    setMetric("accuracy");
    setMinSamples("5000");
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
          error={errors.name}
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
          value={minSamples}
          onChange={(e) => setMinSamples(e.target.value)}
          placeholder="5000"
          error={errors.min_samples}
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
