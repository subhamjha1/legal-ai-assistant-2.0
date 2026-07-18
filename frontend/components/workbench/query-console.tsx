"use client";

import { FormEvent, useRef } from "react";
import { ArrowUp, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AnswerText } from "./answer-text";
import { cn } from "@/lib/utils";

export type QueryPhase = "idle" | "retrieving" | "streaming" | "done" | "error";

const EXAMPLE_QUERIES = [
  "What are the conditions for claiming a Section 80G deduction?",
  "Summarize the court's holding on the disallowed deduction.",
  "What case number and court heard this matter?",
];

interface QueryConsoleProps {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  phase: QueryPhase;
  answerText: string;
  hasSufficientEvidence: boolean | null;
  activeRef: string | null;
  onSelectRef: (ref: string) => void;
  documentCount: number;
}

export function QueryConsole({
  query,
  onQueryChange,
  onSubmit,
  phase,
  answerText,
  hasSufficientEvidence,
  activeRef,
  onSelectRef,
  documentCount,
}: QueryConsoleProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const isBusy = phase === "retrieving" || phase === "streaming";

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim() || isBusy) return;
    onSubmit();
  }

  return (
    <div className="flex h-full flex-col">
      <form onSubmit={handleSubmit} className="border-b border-hairline p-4">
        <div className="flex items-center gap-2 rounded-md border border-hairline-strong bg-panel-raised/60 px-3 py-2 focus-within:border-brass/50">
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            disabled={isBusy || documentCount === 0}
            placeholder={
              documentCount === 0
                ? "Upload a document to begin..."
                : "Ask a question about your documents..."
            }
            className="flex-1 bg-transparent font-mono text-sm text-parchment placeholder:text-parchment-dim/60 outline-none disabled:cursor-not-allowed"
          />
          <Button
            type="submit"
            size="icon"
            disabled={isBusy || !query.trim() || documentCount === 0}
            aria-label="Ask"
          >
            <ArrowUp />
          </Button>
        </div>
      </form>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {phase === "idle" && answerText === "" && (
          <EmptyState
            documentCount={documentCount}
            onPickExample={(q) => {
              onQueryChange(q);
              inputRef.current?.focus();
            }}
          />
        )}

        {phase === "retrieving" && (
          <div className="flex items-center gap-2 font-mono text-sm text-parchment-dim">
            <span>Consulting the record</span>
            <span className="animate-ink-caret">_</span>
          </div>
        )}

        {(phase === "streaming" || phase === "done") && answerText !== "" && (
          <div className="animate-rise-in max-w-[68ch]">
            {hasSufficientEvidence === false && phase === "done" ? (
              <div className="flex items-start gap-3 rounded-md border border-brick/30 bg-brick-wash px-4 py-3">
                <ShieldAlert className="mt-0.5 size-4 shrink-0 text-brick" />
                <p className="text-[15px] leading-relaxed text-parchment">{answerText}</p>
              </div>
            ) : (
              <AnswerText text={answerText} activeRef={activeRef} onSelectRef={onSelectRef} />
            )}
            {phase === "streaming" && (
              <span className="animate-ink-caret ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-brass" />
            )}
          </div>
        )}

        {phase === "error" && (
          <div className="rounded-md border border-brick/30 bg-brick-wash px-4 py-3 text-sm text-parchment">
            Something went wrong reaching the research pipeline. Check that the backend is
            running and try again.
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({
  documentCount,
  onPickExample,
}: {
  documentCount: number;
  onPickExample: (q: string) => void;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <p className="font-display text-2xl italic text-parchment-dim">
        {documentCount === 0
          ? "Upload a document to open the record."
          : "Ask anything grounded in what you\u2019ve uploaded."}
      </p>
      {documentCount > 0 && (
        <div className="mt-6 flex flex-col gap-2">
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => onPickExample(q)}
              className={cn(
                "rounded-md border border-hairline px-3 py-2 text-left text-sm text-parchment-dim",
                "hover:border-hairline-strong hover:text-parchment hover:bg-panel-raised/60 transition-colors"
              )}
            >
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
