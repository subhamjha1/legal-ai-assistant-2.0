"use client";

import { useEffect, useRef, useState } from "react";
import {
  createChunks,
  deleteDocument,
  indexKeyword,
  indexVector,
  listDocuments,
  streamAnswer,
  uploadDocument,
} from "@/lib/api";
import type { Citation, DocumentSummary, PipelineStep, StreamEvent } from "@/lib/types";
import { DocumentLibrary } from "@/components/workbench/document-library";
import { QueryConsole, type QueryPhase } from "@/components/workbench/query-console";
import { ExhibitLedger } from "@/components/workbench/exhibit-ledger";

const INITIAL_STEPS: PipelineStep[] = [
  { key: "upload", label: "Parsing pages", status: "pending" },
  { key: "chunk", label: "Structuring chunks", status: "pending" },
  { key: "vector_index", label: "Embedding & indexing", status: "pending" },
  { key: "keyword_index", label: "Indexing keywords", status: "pending" },
];

export default function Home() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [pipeline, setPipeline] = useState<{ filename: string; steps: PipelineStep[] } | null>(
    null
  );

  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<QueryPhase>("idle");
  const [answerText, setAnswerText] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [hasSufficientEvidence, setHasSufficientEvidence] = useState<boolean | null>(null);
  const [activeRef, setActiveRef] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    refreshDocuments();
  }, []);

  async function refreshDocuments() {
    try {
      setDocuments(await listDocuments());
    } catch {
      // Backend not reachable yet - the empty-state UI covers this gracefully.
    }
  }

  function updateStep(key: PipelineStep["key"], status: PipelineStep["status"]) {
    setPipeline((prev) =>
      prev
        ? { ...prev, steps: prev.steps.map((s) => (s.key === key ? { ...s, status } : s)) }
        : prev
    );
  }

  async function handleUpload(file: File) {
    setPipeline({ filename: file.name, steps: INITIAL_STEPS.map((s) => ({ ...s })) });

    try {
      updateStep("upload", "active");
      const uploaded = await uploadDocument(file);
      updateStep("upload", "done");

      updateStep("chunk", "active");
      await createChunks(uploaded.document_id);
      updateStep("chunk", "done");

      updateStep("vector_index", "active");
      await indexVector(uploaded.document_id);
      updateStep("vector_index", "done");

      updateStep("keyword_index", "active");
      await indexKeyword(uploaded.document_id);
      updateStep("keyword_index", "done");

      await refreshDocuments();
      setTimeout(() => setPipeline(null), 900);
    } catch (err) {
      setPipeline((prev) => {
        if (!prev) return prev;
        const firstIncomplete = prev.steps.find((s) => s.status !== "done");
        return {
          ...prev,
          steps: prev.steps.map((s) =>
            s.key === firstIncomplete?.key ? { ...s, status: "error" } : s
          ),
        };
      });
      console.error(err);
    }
  }

  async function handleDelete(documentId: string) {
    await deleteDocument(documentId);
    await refreshDocuments();
  }

  async function handleAsk() {
    const q = query.trim();
    if (!q) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("retrieving");
    setAnswerText("");
    setCitations([]);
    setHasSufficientEvidence(null);
    setActiveRef(null);

    try {
      let started = false;
      await streamAnswer(
        q,
        5,
        (event: StreamEvent) => {
          if (event.type === "token") {
            if (!started) {
              started = true;
              setPhase("streaming");
            }
            setAnswerText((prev) => prev + event.text);
          } else {
            setCitations(event.citations);
            setHasSufficientEvidence(event.has_sufficient_evidence);
            setPhase("done");
          }
        },
        controller.signal
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        console.error(err);
        setPhase("error");
      }
    }
  }

  return (
    <main className="mx-auto flex h-screen w-full max-w-[1600px] flex-1 overflow-hidden">
      <aside className="w-[280px] shrink-0 border-r border-hairline">
        <DocumentLibrary
          documents={documents}
          onUpload={handleUpload}
          onDelete={handleDelete}
          pipeline={pipeline}
        />
      </aside>

      <section className="min-w-0 flex-1 border-r border-hairline">
        <QueryConsole
          query={query}
          onQueryChange={setQuery}
          onSubmit={handleAsk}
          phase={phase}
          answerText={answerText}
          hasSufficientEvidence={hasSufficientEvidence}
          activeRef={activeRef}
          onSelectRef={(ref) => {
            setActiveRef(ref);
            document
              .getElementById(`exhibit-${ref}`)
              ?.scrollIntoView({ behavior: "smooth", block: "center" });
          }}
          documentCount={documents.length}
        />
      </section>

      <aside className="w-[320px] shrink-0">
        <ExhibitLedger
          citations={citations}
          activeRef={activeRef}
          onSelect={setActiveRef}
          hasAnswer={answerText !== ""}
          isNoEvidence={hasSufficientEvidence === false}
        />
      </aside>
    </main>
  );
}
