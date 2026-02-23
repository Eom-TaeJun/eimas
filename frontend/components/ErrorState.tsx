"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle,
  RefreshCw,
  WifiOff,
  ServerCrash,
  FileX,
  Settings,
} from "lucide-react";

interface ErrorStateProps {
  error?: Error | string;
  onRetry?: () => void;
  type?: "network" | "server" | "notfound" | "generic";
  title?: string;
  message?: string;
}

const errorConfig = {
  network: {
    icon: WifiOff,
    title: "Connection Error",
    message: "Unable to connect to the EIMAS backend. Please check if the FastAPI server is running.",
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/20",
  },
  server: {
    icon: ServerCrash,
    title: "Server Error",
    message: "The EIMAS analysis engine encountered an error. Please try again or contact support.",
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/20",
  },
  notfound: {
    icon: FileX,
    title: "Data Not Found",
    message: "No analysis results found. Please run an EIMAS analysis first.",
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/20",
  },
  generic: {
    icon: AlertTriangle,
    title: "Error Occurred",
    message: "An unexpected error occurred while loading the data.",
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/20",
  },
};

export function ErrorState({
  error,
  onRetry,
  type = "generic",
  title,
  message,
}: ErrorStateProps) {
  const config = errorConfig[type];
  const Icon = config.icon;

  const errorMessage =
    error instanceof Error
      ? error.message
      : typeof error === "string"
      ? error
      : message || config.message;

  const displayTitle = title || config.title;

  return (
    <Card className={`bg-surface border-2 ${config.borderColor}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className={`w-6 h-6 ${config.color}`} />
          <span className="text-white">{displayTitle}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`${config.bgColor} rounded-lg p-4 border ${config.borderColor}`}>
          <p className="text-gray-300">{errorMessage}</p>
        </div>

        {type === "network" && (
          <div className="space-y-2">
            <p className="text-sm font-semibold text-gray-300">Troubleshooting Steps:</p>
            <ul className="space-y-1 text-sm text-gray-400">
              <li className="flex items-start gap-2">
                <span className="text-purple-400">1.</span>
                <span>Verify FastAPI server is running on <code className="text-xs bg-gray-800 px-1 py-0.5 rounded">localhost:8000</code></span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400">2.</span>
                <span>Check that at least one EIMAS analysis has been run</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400">3.</span>
                <span>Ensure your network connection is stable</span>
              </li>
            </ul>
          </div>
        )}

        {type === "notfound" && (
          <div className="space-y-2">
            <p className="text-sm font-semibold text-gray-300">Quick Start:</p>
            <ul className="space-y-1 text-sm text-gray-400">
              <li className="flex items-start gap-2">
                <span className="text-purple-400">1.</span>
                <span>
                  Run EIMAS analysis:{" "}
                  <code className="text-xs bg-gray-800 px-1 py-0.5 rounded">
                    python main.py --quick
                  </code>
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400">2.</span>
                <span>Wait for analysis to complete</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400">3.</span>
                <span>Refresh this page</span>
              </li>
            </ul>
          </div>
        )}

        <div className="flex items-center gap-3 pt-2">
          {onRetry && (
            <Button
              onClick={onRetry}
              className="bg-secondary hover:bg-secondary-600 text-white"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => window.open("http://localhost:8000/docs", "_blank")}
            className="bg-surface-card border-border text-gray-300 hover:bg-secondary hover:text-white"
          >
            <Settings className="w-4 h-4 mr-2" />
            API Docs
          </Button>
        </div>

        {error instanceof Error && error.stack && (
          <details className="mt-4">
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
              Show error details
            </summary>
            <pre className="mt-2 p-3 bg-gray-900 rounded text-xs text-gray-400 overflow-auto max-h-40">
              {error.stack}
            </pre>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

export function InlineError({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
      <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
      <span className="text-sm text-red-300">{message}</span>
    </div>
  );
}
