export type DocumentType = "act" | "judgment" | "tax_document" | "pov_document" | "unknown";

export const DOCUMENT_TYPE_LABELS: Record<DocumentType, string> = {
  act: "Act",
  judgment: "Judgment",
  tax_document: "Tax Document",
  pov_document: "POV Document",
  unknown: "Unclassified",
};

export interface DocumentSummary {
  document_id: string;
  original_filename: string;
  document_type: DocumentType;
  total_pages: number;
  uploaded_at: string;
}

export interface UploadResponse {
  document_id: string;
  original_filename: string;
  document_type: DocumentType;
  document_type_source: "user_tagged" | "auto_suggested";
  type_suggestion: {
    suggested_type: DocumentType;
    confidence: number;
    method: string;
    reasoning: string | null;
  } | null;
  total_pages: number;
  pages_requiring_ocr: number;
  processing_time_seconds: number;
}

export interface Citation {
  chunk_ref: string;
  document: string;
  page_start: number;
  page_end: number;
  structural_label: string | null;
  snippet: string;
}

export interface AnswerResponse {
  query: string;
  answer: string;
  citations: Citation[];
  has_sufficient_evidence: boolean;
  chunks_considered: number;
  model_used: string;
}

export type StreamEvent =
  | { type: "token"; text: string }
  | {
      type: "done";
      citations: Citation[];
      has_sufficient_evidence: boolean;
      chunks_considered: number;
      model_used: string;
    };

export type PipelineStepStatus = "pending" | "active" | "done" | "error";

export interface PipelineStep {
  key: "upload" | "chunk" | "vector_index" | "keyword_index";
  label: string;
  status: PipelineStepStatus;
}
