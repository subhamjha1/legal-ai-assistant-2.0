import type {
  AnswerResponse,
  DocumentSummary,
  DocumentType,
  StreamEvent,
  UploadResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Request failed (${res.status}): ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadDocument(
  file: File,
  documentType?: DocumentType
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const url = new URL(`${API_BASE}/api/v1/documents/upload`);
  if (documentType) url.searchParams.set("document_type", documentType);

  const res = await fetch(url.toString(), { method: "POST", body: form });
  return handle<UploadResponse>(res);
}

export async function createChunks(documentId: string) {
  const res = await fetch(`${API_BASE}/api/v1/documents/${documentId}/chunks`, {
    method: "POST",
  });
  return handle(res);
}

export async function indexVector(documentId: string) {
  const res = await fetch(`${API_BASE}/api/v1/documents/${documentId}/index`, {
    method: "POST",
  });
  return handle(res);
}

export async function indexKeyword(documentId: string) {
  const res = await fetch(`${API_BASE}/api/v1/documents/${documentId}/keyword-index`, {
    method: "POST",
  });
  return handle(res);
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const res = await fetch(`${API_BASE}/api/v1/documents`);
  return handle<DocumentSummary[]>(res);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v1/documents/${documentId}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`Failed to delete document (${res.status})`);
  }
}

export async function askQuestion(query: string, topK = 5): Promise<AnswerResponse> {
  const res = await fetch(`${API_BASE}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  return handle<AnswerResponse>(res);
}

/**
 * Consumes the /query/stream Server-Sent Events endpoint. Uses a manual
 * fetch()+ReadableStream reader rather than the browser's EventSource API,
 * since EventSource only supports GET requests and this endpoint needs a
 * POST body (the query text, filters, etc).
 */
export async function streamAnswer(
  query: string,
  topK: number,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v1/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? ""; // keep the last, possibly-incomplete chunk

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const jsonText = trimmed.slice("data:".length).trim();
      if (!jsonText) continue;
      try {
        onEvent(JSON.parse(jsonText) as StreamEvent);
      } catch {
        // Ignore malformed lines rather than aborting the whole stream.
      }
    }
  }
}
