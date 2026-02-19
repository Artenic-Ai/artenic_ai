import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select, Textarea } from "@/components/ui/input";
import { useStorageOptions } from "@/hooks/use-datasets";

interface DatasetCreateDialogProps {
  open: boolean;
  onClose: () => void;
}

export function DatasetCreateDialog({
  open,
  onClose,
}: DatasetCreateDialogProps) {
  const [name, setName] = useState("");
  const [format, setFormat] = useState("csv");
  const [storageBackend, setStorageBackend] = useState("filesystem");
  const [description, setDescription] = useState("");
  const [source, setSource] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const { data: storageOptions } = useStorageOptions();

  const storageSelectOptions = (storageOptions ?? []).map((opt) => ({
    value: opt.id,
    label: opt.available ? opt.label : `${opt.label} (unavailable)`,
  }));

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const newErrors: Record<string, string> = {};
    if (!name.trim()) {
      newErrors.name = "Name is required";
    }
    const selectedStorage = storageOptions?.find(
      (s) => s.id === storageBackend,
    );
    if (selectedStorage && !selectedStorage.available) {
      newErrors.storage = "Selected storage backend is not available";
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setErrors({});
    // Demo mode: just close
    onClose();
    setName("");
    setFormat("csv");
    setStorageBackend("filesystem");
    setDescription("");
    setSource("");
  }

  return (
    <Dialog open={open} onClose={onClose} title="Create Dataset">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-dataset"
          required
          error={errors.name}
        />
        <Select
          label="Format"
          value={format}
          onChange={(e) => setFormat(e.target.value)}
          options={[
            { value: "csv", label: "CSV" },
            { value: "parquet", label: "Parquet" },
            { value: "json", label: "JSON" },
            { value: "jsonl", label: "JSONL" },
            { value: "image", label: "Image" },
            { value: "audio", label: "Audio" },
            { value: "mixed", label: "Mixed" },
            { value: "other", label: "Other" },
          ]}
        />
        <Select
          label="Storage Backend"
          value={storageBackend}
          onChange={(e) => setStorageBackend(e.target.value)}
          options={
            storageSelectOptions.length > 0
              ? storageSelectOptions
              : [{ value: "filesystem", label: "Local Filesystem" }]
          }
        />
        {errors.storage && (
          <p className="text-xs text-danger">{errors.storage}</p>
        )}
        <Textarea
          label="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Brief description of this dataset..."
          rows={3}
        />
        <Input
          label="Source"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="e.g. internal-datalake, https://..."
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
