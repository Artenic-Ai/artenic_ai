import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select } from "@/components/ui/input";

interface TrainingDispatchDialogProps {
  open: boolean;
  onClose: () => void;
}

export function TrainingDispatchDialog({
  open,
  onClose,
}: TrainingDispatchDialogProps) {
  const [service, setService] = useState("");
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("gcp");
  const [instanceType, setInstanceType] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const newErrors: Record<string, string> = {};
    if (!service.trim()) {
      newErrors.service = "Service is required";
    }
    if (!model.trim()) {
      newErrors.model = "Model is required";
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setErrors({});
    onClose();
    setService("");
    setModel("");
    setProvider("gcp");
    setInstanceType("");
  }

  return (
    <Dialog open={open} onClose={onClose} title="Dispatch Training Job">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Service"
          value={service}
          onChange={(e) => setService(e.target.value)}
          placeholder="sentiment"
          required
          error={errors.service}
        />
        <Input
          label="Model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="sentiment-bert-v3"
          required
          error={errors.model}
        />
        <Select
          label="Provider"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          options={[
            { value: "gcp", label: "GCP" },
            { value: "aws", label: "AWS" },
            { value: "vastai", label: "VastAI" },
            { value: "runpod", label: "RunPod" },
            { value: "local", label: "Local" },
          ]}
        />
        <Input
          label="Instance Type"
          value={instanceType}
          onChange={(e) => setInstanceType(e.target.value)}
          placeholder="n1-standard-8-t4"
        />
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="ghost" type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit">Dispatch</Button>
        </div>
      </form>
    </Dialog>
  );
}
