# Legal AI Assistant — Frontend

## Milestone 8: Frontend (complete)

A real Next.js 15 (App Router) application — not a prototype — built with
TypeScript, Tailwind CSS v4, and hand-authored shadcn-style components
(Radix primitives + `class-variance-authority`, vendored directly into the
repo rather than pulled from `shadcn`'s CLI registry, which this
environment's network allow-list couldn't reach — see "A note on this
sandbox" below).

### What it does

- **Document library** (left rail): upload a PDF, watch it move through the
  real backend pipeline (parse → chunk → vector index → keyword index) with
  live per-step status, then browse/delete indexed documents by type.
- **Query console** (center): ask a question, watch the answer **stream in
  token-by-token** over Server-Sent Events, with inline `[C1]`-style
  citation tags rendered as clickable brass chips as they arrive.
- **Exhibit Ledger** (right rail): the citations the model actually used,
  presented as numbered tabs — click one to expand the source snippet, or
  click a chip in the answer text to jump to its tab. Every page number and
  document name shown here comes from the backend's citation formatter,
  which sources them from the retrieval pipeline's own data, never from the
  model's text (see Milestone 7).

### Design system

Built around the actual physical materiality of legal work — litigation
binders, exhibit dividers, brass courtroom fixtures — rather than generic
chat-app or SaaS-dashboard defaults:

| Token | Value | Use |
|---|---|---|
| `ink` | `#0b0d10` | page background |
| `panel` / `panel-raised` | `#14171c` / `#1b1f26` | rail & card surfaces |
| `parchment` | `#e8e6df` | primary text (warm off-white, not pure white) |
| `brass` | `#c9a227` | primary accent — citations, active states |
| `ledger` | `#4c8a71` | confidence / success |
| `brick` | `#c0563d` | no-evidence / low-confidence states |

Typography: **Fraunces** (display serif, used sparingly for headings and
empty-state copy), **Public Sans** (UI/body — the U.S. government design
system's typeface, a deliberate, subject-fitting choice over the more
common Inter), **IBM Plex Mono** (citations, page references, document
metadata). All three are self-hosted via `@fontsource` npm packages rather
than a Google Fonts `<link>` tag — one less third-party request, no
font-loading layout shift, and no build-time dependency on
`fonts.googleapis.com` (which this sandbox's network couldn't reach anyway).

The signature element is the **Exhibit Ledger**: citations rendered as
numbered tabs echoing physical exhibit dividers in a litigation binder — a
real artifact of the subject's own world, not a decorative motif, and the
numbering is literal (the order the answer first cited each passage), not
arbitrary.

### A note on this sandbox's constraints (same honest pattern as the backend)

- **npm registry, GitHub, self-hosted fonts**: all reachable — this app was
  genuinely `npm install`'d, `npm run build`'d, `npm run lint`'d (both
  clean), and visually verified via Playwright screenshots of a live
  `npm run start` server, all in this sandbox.
- **`ui.shadcn.com`, `fonts.googleapis.com`**: blocked (confirmed 403).
  Shadcn's CLI fetches its component templates from `ui.shadcn.com`'s
  registry; since shadcn components are just vendored source you own
  anyway (not a runtime package), the components in `components/ui/` are
  hand-authored using the same underlying approach (Radix primitives +
  `cva` + `tailwind-merge`) rather than CLI-generated — functionally
  identical output.
- **The actual LLM answer in any screenshot you're shown was a stubbed
  response**, not a real Claude/GPT/Gemini call — this sandbox has no
  `ANTHROPIC_API_KEY` configured (see backend README: the *network path* to
  Anthropic is open here, just not the credential). The retrieval pipeline
  underneath (real parsing, real chunking, real hybrid search, real MMR,
  real re-ranking) is 100% genuine; only the final answer-generation call
  was stubbed for that one verification pass, using a throwaway harness
  script that is **not** part of this repository.

### Running it for real

```bash
cd frontend
npm install
cp .env.example .env.local   # points at http://localhost:8000 by default
npm run dev
```

Requires the backend running (see `../backend/README.md`) with at least
one document uploaded and indexed. With a real `ANTHROPIC_API_KEY` set on
the backend, the streaming answers you see will be genuine Claude output.

### Project structure

```
frontend/
├── app/
│   ├── layout.tsx       # root layout, self-hosted fonts via @fontsource
│   ├── globals.css      # design tokens (Tailwind v4 @theme block)
│   └── page.tsx         # main workbench - owns all state
├── components/
│   ├── ui/              # hand-authored shadcn-style primitives
│   │   ├── button.tsx / badge.tsx / card.tsx / scroll-area.tsx
│   └── workbench/
│       ├── document-library.tsx   # left rail: upload + pipeline progress
│       ├── query-console.tsx      # center: input + streaming answer
│       ├── answer-text.tsx        # parses [Cx] tags into citation chips
│       └── exhibit-ledger.tsx     # right rail: the signature element
└── lib/
    ├── api.ts    # typed client for every backend endpoint, incl. SSE parsing
    ├── types.ts  # mirrors backend Pydantic schemas
    └── utils.ts  # cn() helper (clsx + tailwind-merge)
```

### What's next (Milestone 9)

Evaluation: a golden dataset of ~100 questions with ground-truth answers,
sources, and pages, run against this full pipeline and scored with
RAGAS/DeepEval (retrieval accuracy, faithfulness, context precision,
answer correctness) — turning "it looks like it works" into measured
numbers.
