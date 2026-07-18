"use client";

import { Fragment } from "react";
import { cn } from "@/lib/utils";

const TAG_PATTERN = /(\[C\d+\])/g;

interface AnswerTextProps {
  text: string;
  activeRef: string | null;
  onSelectRef: (ref: string) => void;
}

/**
 * Splits the current accumulated answer text on [Cx] tags and renders each
 * as a small brass citation chip instead of raw bracket text. Runs on the
 * whole accumulated string on every render (cheap for answer-length text),
 * which sidesteps the token-boundary edge case where a tag like "[C1]"
 * arrives split across two streamed deltas - it simply renders as literal
 * characters until the closing bracket appears, then resolves to a chip.
 */
export function AnswerText({ text, activeRef, onSelectRef }: AnswerTextProps) {
  const parts = text.split(TAG_PATTERN);

  return (
    <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-parchment">
      {parts.map((part, i) => {
        const match = part.match(/^\[C(\d+)\]$/);
        if (!match) return <Fragment key={i}>{part}</Fragment>;

        const ref = `C${match[1]}`;
        return (
          <button
            key={i}
            onClick={() => onSelectRef(ref)}
            className={cn(
              "mx-0.5 inline-flex h-[1.1em] min-w-[1.4em] translate-y-[-0.15em] items-center justify-center rounded-sm px-1 font-mono text-[10px] font-semibold align-middle transition-colors",
              activeRef === ref
                ? "bg-brass text-ink"
                : "bg-hairline-strong text-parchment hover:bg-brass/60 hover:text-ink"
            )}
            aria-label={`Jump to exhibit ${ref}`}
          >
            {ref}
          </button>
        );
      })}
    </p>
  );
}
