import { useState } from "react";

import { Play, Zap } from "lucide-react";

import { PageShell } from "@/components/layout/page-shell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { JsonEditor } from "@/components/ui/json-editor";
import { Spinner } from "@/components/ui/spinner";

function buildDemoResponse(service: string): unknown {
  const timestamp = new Date().toISOString();
  const responses: Record<string, unknown> = {
    sentiment: {
      prediction: { label: "positive", confidence: 0.94 },
      model_id: "mdl-a1b2c3d4",
      service: "sentiment",
      timestamp,
      inference_time_ms: 42,
    },
    ner: {
      prediction: {
        entities: [
          { text: "Artenic", type: "ORG", start: 0, end: 7 },
          { text: "Zurich", type: "LOC", start: 20, end: 26 },
        ],
      },
      model_id: "mdl-e5f6g7h8",
      service: "ner",
      timestamp,
      inference_time_ms: 68,
    },
    vision: {
      prediction: { class: "laptop", confidence: 0.87, bbox: [120, 45, 380, 290] },
      model_id: "mdl-i9j0k1l2",
      service: "vision",
      timestamp,
      inference_time_ms: 35,
    },
    fraud: {
      prediction: { is_fraud: false, score: 0.12 },
      model_id: "mdl-u1v2w3x4",
      service: "fraud",
      timestamp,
      inference_time_ms: 5,
    },
  };
  return responses[service] ?? responses.sentiment;
}

export function InferencePlaygroundPage() {
  const [service, setService] = useState("sentiment");
  const [modelId, setModelId] = useState("");
  const [input, setInput] = useState('{\n  "text": "This product is amazing!"\n}');
  const [response, setResponse] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handlePredict() {
    setError(null);

    try {
      JSON.parse(input);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid JSON input.");
      return;
    }

    setLoading(true);
    setResponse(null);

    // Simulate delay
    await new Promise((resolve) => setTimeout(resolve, 200 + Math.random() * 300));

    const demo = buildDemoResponse(service);
    setResponse(JSON.stringify(demo, null, 2));
    setLoading(false);
  }

  return (
    <PageShell
      title="Inference Playground"
      description="Test model predictions with custom input."
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Request">
          <div className="space-y-4">
            <Select
              label="Service"
              value={service}
              onChange={(e) => setService(e.target.value)}
              options={[
                { value: "sentiment", label: "Sentiment Analysis" },
                { value: "ner", label: "Named Entity Recognition" },
                { value: "vision", label: "Vision Classification" },
                { value: "fraud", label: "Fraud Detection" },
                { value: "detection", label: "Object Detection" },
                { value: "embedding", label: "Text Embedding" },
              ]}
            />
            <Input
              label="Model ID (optional)"
              placeholder="Leave empty for default"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            />
            <JsonEditor
              label="Input Data"
              value={input}
              onChange={setInput}
              rows={12}
            />
            {error && (
              <p className="text-sm text-danger">{error}</p>
            )}
            <Button onClick={handlePredict} disabled={loading} className="w-full">
              {loading ? (
                <Spinner size="sm" />
              ) : (
                <>
                  <Play size={16} />
                  Run Prediction
                </>
              )}
            </Button>
          </div>
        </Card>

        <Card title="Response">
          {response ? (
            <pre className="overflow-auto rounded-md border border-border bg-surface-2 p-4 font-mono text-sm text-text-primary">
              {response}
            </pre>
          ) : (
            <div className="flex h-64 flex-col items-center justify-center text-text-muted">
              <Zap size={32} className="mb-2" />
              <p className="text-sm">Run a prediction to see results</p>
            </div>
          )}
        </Card>
      </div>
    </PageShell>
  );
}
