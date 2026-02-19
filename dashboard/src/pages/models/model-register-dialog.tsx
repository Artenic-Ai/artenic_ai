import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select } from "@/components/ui/input";

interface ModelRegisterDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ModelRegisterDialog({
  open,
  onClose,
}: ModelRegisterDialogProps) {
  const [name, setName] = useState("");
  const [version, setVersion] = useState("");
  const [modelType, setModelType] = useState("");
  const [framework, setFramework] = useState("pytorch");
  const [errors, setErrors] = useState<Record<string, string>>({});

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const newErrors: Record<string, string> = {};
    if (!name.trim()) {
      newErrors.name = "Name is required";
    }
    if (!/^\d+\.\d+\.\d+$/.test(version)) {
      newErrors.version = "Version must be in semver format (e.g. 1.0.0)";
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setErrors({});
    // Demo mode: just close
    onClose();
    setName("");
    setVersion("");
    setModelType("");
    setFramework("pytorch");
  }

  return (
    <Dialog open={open} onClose={onClose} title="Register Model">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-model-v1"
          required
          error={errors.name}
        />
        <Input
          label="Version"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
          placeholder="1.0.0"
          required
          error={errors.version}
        />
        <Input
          label="Model Type"
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
          placeholder="NLP/classification"
          required
        />
        <Select
          label="Framework"
          value={framework}
          onChange={(e) => setFramework(e.target.value)}
          options={[
            { value: "pytorch", label: "PyTorch" },
            { value: "tensorflow", label: "TensorFlow" },
            { value: "onnx", label: "ONNX" },
            { value: "xgboost", label: "XGBoost" },
            { value: "lightgbm", label: "LightGBM" },
            { value: "sklearn", label: "scikit-learn" },
          ]}
        />
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="ghost" type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit">Register</Button>
        </div>
      </form>
    </Dialog>
  );
}
