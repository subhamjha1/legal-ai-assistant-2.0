"use client";

import { ChevronRight, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Citation } from "@/lib/types";

interface ExhibitLedgerProps {
  citations: Citation[];
  activeRef: string | null;
  onSelect: (ref: string | null) => void;
  hasAnswer: boolean;
  isNoEvidence: boolean;
}

/**
 * The one deliberately bold element in this design (see design brief:
 * "spend your boldness in one place"). Each citation the model used
 * appears as a numbered tab, echoing the physical exhibit dividers used in
 * litigation binders - a real artifact of the subject's own world, not a
 * decorative motif. The numbering here is legitimate (not a generic
 * 01/02/03 device): it's literally the order in which the answer first
 * cites each passage.
 */
export function ExhibitLedger({ citations, activeRef, onSelect, hasAnswer, isNoEvidence }: ExhibitLedgerProps) {
  if (!hasAnswer) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <div className="mb-3 h-px w-8 bg-hairline-strong" />
        <p className="font-mono text-[11px] uppercase tracking-wider text-parchment-dim">
          Exhibit Ledger
        </p>
        <p className="mt-2 max-w-[20ch] text-sm text-parchment-dim">
          Citations will appear here as the answer is written.
        </p>
      </div>
    );
  }

  if (isNoEvidence) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <p className="font-mono text-[11px] uppercase tracking-wider text-brick">No Exhibits</p>
        <p className="mt-2 max-w-[22ch] text-sm text-parchment-dim">
          No passage in the record supported an answer, so none were cited.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-hairline px-4 py-3">
        <p className="font-mono text-[11px] uppercase tracking-wider text-parchment-dim">
          Exhibit Ledger
        </p>
        <p className="mt-0.5 font-mono text-[11px] text-parchment-dim/70">
          {citations.length} passage{citations.length === 1 ? "" : "s"} cited
        </p>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-3">
        <ul className="flex flex-col gap-2">
          {citations.map((citation) => (
            <ExhibitTab
              key={citation.chunk_ref}
              citation={citation}
              isActive={activeRef === citation.chunk_ref}
              onSelect={() =>
                onSelect(activeRef === citation.chunk_ref ? null : citation.chunk_ref)
              }
            />
          ))}
        </ul>
      </div>
    </div>
  );
}

function ExhibitTab({
  citation,
  isActive,
  onSelect,
}: {
  citation: Citation;
  isActive: boolean;
  onSelect: () => void;
}) {
  const pageLabel =
    citation.page_start === citation.page_end
      ? `p. ${citation.page_start}`
      : `pp. ${citation.page_start}\u2013${citation.page_end}`;

  return (
    <li
      id={`exhibit-${citation.chunk_ref}`}
      className={cn(
        "animate-rise-in overflow-hidden rounded-md border transition-all",
        isActive
          ? "border-brass/50 bg-brass-wash shadow-[0_0_0_1px_rgba(201,162,39,0.15)]"
          : "border-hairline bg-panel-raised/40 hover:border-hairline-strong"
      )}
    >
      <button
        onClick={onSelect}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left"
        aria-expanded={isActive}
      >
        <span
          className={cn(
            "flex h-6 min-w-6 items-center justify-center rounded-sm px-1 font-mono text-[11px] font-semibold",
            isActive ? "bg-brass text-ink" : "bg-hairline-strong text-parchment"
          )}
        >
          {citation.chunk_ref}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-[13px] text-parchment">
            <FileText className="size-3.5 shrink-0 text-parchment-dim" />
            <span className="truncate">{citation.document}</span>
          </div>
          <p className="mt-0.5 font-mono text-[11px] text-parchment-dim">
            {pageLabel}
            {citation.structural_label ? ` \u00b7 ${citation.structural_label}` : ""}
          </p>
        </div>
        <ChevronRight
          className={cn(
            "size-4 shrink-0 text-parchment-dim transition-transform",
            isActive && "rotate-90 text-brass"
          )}
        />
      </button>
      {isActive && (
        <div className="animate-rise-in border-t border-hairline-strong/60 px-3 py-2.5">
          <p className="text-[13px] leading-relaxed text-parchment-dim">{citation.snippet}</p>
        </div>
      )}
    </li>
  );
}
