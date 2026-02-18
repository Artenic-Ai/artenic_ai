import { useState } from "react";

interface JsonEditorProps {
  label?: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
  placeholder?: string;
}

export function JsonEditor({
  label,
  value,
  onChange,
  rows = 10,
  placeholder = '{\n  "text": "Hello world"\n}',
}: JsonEditorProps) {
  const [error, setError] = useState<string | null>(null);

  function handleChange(raw: string) {
    onChange(raw);
    if (!raw.trim()) {
      setError(null);
      return;
    }
    try {
      JSON.parse(raw);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid JSON");
    }
  }

  return (
    <div className="space-y-1.5">
      {label && (
        <label className="text-sm font-medium text-text-secondary">
          {label}
        </label>
      )}
      <textarea
        value={value}
        onChange={(e) => handleChange(e.target.value)}
        rows={rows}
        placeholder={placeholder}
        spellCheck={false}
        className={`w-full rounded-md border bg-surface-2 px-3 py-2 font-mono text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-1 ${
          error
            ? "border-danger focus:border-danger focus:ring-danger"
            : "border-border focus:border-accent focus:ring-accent"
        }`}
      />
      {error && <p className="text-xs text-danger">{error}</p>}
    </div>
  );
}
