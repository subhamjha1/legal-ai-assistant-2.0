"use client";

import { useRef } from "react";
import { Check, Loader2, Trash2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { DOCUMENT_TYPE_LABELS, type DocumentSummary, type PipelineStep } from "@/lib/types";

interface DocumentLibraryProps {
  documents: DocumentSummary[];
  onUpload: (file: File) => void;
  onDelete: (documentId: string) => void;
  pipeline: { filename: string; steps: PipelineStep[] } | null;
}

const PIPELINE_LABELS: Record<PipelineStep["key"], string> = {
  upload: "Parsing pages",
  chunk: "Structuring chunks",
  vector_index: "Embedding & indexing",
  keyword_index: "Indexing keywords",
};

export function DocumentLibrary({ documents, onUpload, onDelete, pipeline }: DocumentLibraryProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-hairline p-4">
        <p className="font-display text-lg text-parchment">Ledger</p>
        <p className="mt-0.5 text-xs text-parchment-dim">Legal research workbench</p>
      </div>

      <div className="border-b border-hairline p-4">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
        <Button
          variant="outline"
          className="w-full justify-center"
          onClick={() => fileInputRef.current?.click()}
          disabled={pipeline !== null}
        >
          <Upload className="size-4" />
          Upload document
        </Button>

        {pipeline && <PipelineProgress filename={pipeline.filename} steps={pipeline.steps} />}
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {documents.length === 0 && !pipeline ? (
          <p className="px-2 py-6 text-center text-sm text-parchment-dim">
            No documents yet. Upload an Act, Judgment, Tax Document, or POV Document to
            begin.
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {documents.map((doc) => (
              <li
                key={doc.document_id}
                className="group flex items-center gap-2 rounded-md px-2 py-2 hover:bg-panel-raised/60"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] text-parchment" title={doc.original_filename}>
                    {doc.original_filename}
                  </p>
                  <div className="mt-1 flex items-center gap-1.5">
                    <Badge variant="brass">{DOCUMENT_TYPE_LABELS[doc.document_type]}</Badge>
                    <span className="font-mono text-[10px] text-parchment-dim">
                      {doc.total_pages}p
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => onDelete(doc.document_id)}
                  className="rounded-sm p-1 text-parchment-dim opacity-0 transition-opacity hover:text-brick group-hover:opacity-100"
                  aria-label={`Delete ${doc.original_filename}`}
                >
                  <Trash2 className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function PipelineProgress({ filename, steps }: { filename: string; steps: PipelineStep[] }) {
  return (
    <div className="animate-rise-in mt-3 rounded-md border border-hairline bg-panel-raised/50 p-3">
      <p className="truncate text-[12px] text-parchment-dim" title={filename}>
        {filename}
      </p>
      <ul className="mt-2 flex flex-col gap-1.5">
        {steps.map((step) => (
          <li key={step.key} className="flex items-center gap-2 text-[12px]">
            <StepIcon status={step.status} />
            <span
              className={cn(
                step.status === "pending" && "text-parchment-dim/60",
                step.status === "active" && "text-parchment",
                step.status === "done" && "text-ledger",
                step.status === "error" && "text-brick"
              )}
            >
              {PIPELINE_LABELS[step.key]}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function StepIcon({ status }: { status: PipelineStep["status"] }) {
  if (status === "done") return <Check className="size-3 text-ledger" />;
  if (status === "active") return <Loader2 className="size-3 animate-spin text-brass" />;
  if (status === "error") return <X className="size-3 text-brick" />;
  return <span className="block size-3 rounded-full border border-hairline-strong" />;
}
