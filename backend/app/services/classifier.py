"""
Document type classification service.

Why this exists:
You chose "user can tag, system suggests if blank." A pure LLM call for every
untagged upload is slow and costs money on documents where the answer is
obvious from the first page (e.g. a judgment almost always says "IN THE
COURT OF..." near the top). So we run a fast keyword heuristic first; only
when its confidence is below threshold do we escalate to an LLM call on the
first page of text. This mirrors a real production cost/latency tradeoff.
"""
import re

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.document import DocumentType, DocumentTypeSuggestion

logger = get_logger(__name__)

# Keyword signals per document type, checked against the first ~2 pages of
# extracted text. Order matters only for tie-breaking (first match wins on
# equal score).
_KEYWORD_SIGNALS: dict[DocumentType, list[str]] = {
    DocumentType.JUDGMENT: [
        r"\bIN THE (SUPREME|HIGH|DISTRICT) COURT\b",
        r"\bJUDG(E)?MENT\b",
        r"\bPETITIONER(S)?\b",
        r"\bRESPONDENT(S)?\b",
        r"\bCASE NO\.?\b",
        r"\bBEFORE THE HON'?BLE\b",
        r"\bAPPEAL(ED)? FROM\b",
    ],
    DocumentType.ACT: [
        r"\bAN ACT\b",
        r"\bBE IT ENACTED\b",
        r"\bSECTION \d+\b",
        r"\bSHORT TITLE\b",
        r"\bCHAPTER \d+\b",
        r"\bENACTED BY\b",
    ],
    DocumentType.TAX_DOCUMENT: [
        r"\bINTERNAL REVENUE (SERVICE|CODE)\b",
        r"\bFORM 1040\b",
        r"\bTAXABLE INCOME\b",
        r"\bIRS\b",
        r"\bTAX (RETURN|LIABILITY|ASSESSMENT)\b",
        r"\bDEDUCTION(S)?\b",
    ],
    DocumentType.POV_DOCUMENT: [
        r"\bPOINT OF VIEW\b",
        r"\bIN THE OPINION OF\b",
        r"\bLEGAL OPINION\b",
        r"\bMEMORANDUM OF ADVICE\b",
        r"\bWE ARE OF THE VIEW\b",
    ],
}


class DocumentClassifier:
    """Suggests a document type when the user leaves it unspecified."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def suggest_type(self, first_pages_text: str) -> DocumentTypeSuggestion:
        """
        Run the keyword heuristic; escalate to LLM only if confidence is low.
        `first_pages_text` should be the concatenated text of roughly the
        first 2 pages (enough signal, cheap to scan).
        """
        heuristic_result = self._keyword_heuristic(first_pages_text)

        if heuristic_result.confidence >= self.settings.classification_min_confidence:
            return heuristic_result

        logger.info(
            "Heuristic confidence %.2f below threshold %.2f; escalating to LLM.",
            heuristic_result.confidence,
            self.settings.classification_min_confidence,
        )
        try:
            return self._llm_classify(first_pages_text, heuristic_result)
        except Exception as exc:
            # LLM classification is a nice-to-have, never a hard dependency.
            # If it fails, fall back to the best heuristic guess rather than
            # blocking the upload.
            logger.warning("LLM classification failed (%s); using heuristic guess.", exc)
            return heuristic_result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _keyword_heuristic(text: str) -> DocumentTypeSuggestion:
        upper_text = text.upper()
        scores: dict[DocumentType, int] = {}

        for doc_type, patterns in _KEYWORD_SIGNALS.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, upper_text))
            if matches:
                scores[doc_type] = matches

        if not scores:
            return DocumentTypeSuggestion(
                suggested_type=DocumentType.UNKNOWN,
                confidence=0.0,
                method="keyword_heuristic",
                reasoning="No recognizable legal-domain keyword signals found in the first pages.",
            )

        best_type = max(scores, key=lambda t: scores[t])
        total_signals = len(_KEYWORD_SIGNALS[best_type])
        confidence = min(scores[best_type] / total_signals, 1.0)

        return DocumentTypeSuggestion(
            suggested_type=best_type,
            confidence=round(confidence, 2),
            method="keyword_heuristic",
            reasoning=f"Matched {scores[best_type]}/{total_signals} keyword signals for {best_type.value}.",
        )

    def _llm_classify(
        self, text: str, fallback: DocumentTypeSuggestion
    ) -> DocumentTypeSuggestion:
        """
        Escalation path for ambiguous documents. Wired to a real Anthropic
        call; kept isolated here so it's the only place Milestone 1 talks to
        an LLM, and easy to mock in tests.
        """
        if not self.settings.anthropic_api_key:
            logger.info("No Anthropic API key configured; skipping LLM classification.")
            return fallback

        import anthropic  # local import: optional dependency for this path only

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        prompt = (
            "Classify this legal document excerpt into exactly one category: "
            "act, judgment, tax_document, pov_document, or unknown. "
            "Respond with ONLY a JSON object: "
            '{"type": "<category>", "confidence": <0-1 float>, "reasoning": "<one sentence>"}.\n\n'
            f"Excerpt:\n{text[:3000]}"
        )
        response = client.messages.create(
            model=self.settings.llm_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in response.content if hasattr(block, "text"))

        import json

        parsed = json.loads(raw.strip().strip("`"))
        return DocumentTypeSuggestion(
            suggested_type=DocumentType(parsed["type"]),
            confidence=float(parsed["confidence"]),
            method="llm",
            reasoning=parsed.get("reasoning"),
        )
