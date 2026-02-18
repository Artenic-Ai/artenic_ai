import { AlertTriangle } from "lucide-react";

import { Button } from "./button";

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

export function ErrorState({
  message = "Something went wrong.",
  onRetry,
}: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <AlertTriangle size={40} className="mb-4 text-danger" />
      <h3 className="text-lg font-medium text-text-primary">Error</h3>
      <p className="mt-1 max-w-sm text-sm text-text-secondary">{message}</p>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry} className="mt-4">
          Retry
        </Button>
      )}
    </div>
  );
}
