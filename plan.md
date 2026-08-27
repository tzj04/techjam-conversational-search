## Plan: Conversational Search Agent

TL;DR: Upgrade `starter/agent.py` from stateless SQLite FTS lookup to a standard-library, offline-capable stateful agent. Track slots and exclusions per session, detect intent overrides, retrieve and rerank catalog candidates with exact/field-aware lexical scoring, and choose evaluator-compatible clarification attributes for browsing turns. Validate the API and evaluator behavior locally; defer full benchmark scoring until `data/catalog.jsonl` is available.

**Steps**
1. Establish the current baseline and constraints: read the API contract, evaluator, tests, submission rules, and catalog schema; run the existing unit tests and baseline evaluator where the catalog is available. Record that `catalog.jsonl` is currently absent and avoid adding runtime dependencies that violate offline submission rules.
2. Define a compact internal state model in `starter/agent.py` (or a small standard-library helper module only if needed): session profile, raw message history, active positive slots, budget bounds, exclusions, last asked attribute, and current intent/category. Ensure `reset()` fully initializes or replaces state for the requested session.
3. Implement deterministic slot extraction from profile and user text. Cover product/category terms, material, color, size, style, brand, budget, feature, use case, negation/exclusion language, and “actually/forget/instead” override language. Make category changes clear stale category-dependent slots and preserve only compatible preferences.
4. Implement candidate retrieval using the existing SQLite catalog index. Build multiple query forms from the latest message and active slots, use strict and relaxed searches with fallback for zero-result or niche requests, and ensure every emitted ID comes from the loaded catalog. Keep the implementation standard-library-only and robust to missing/optional catalog fields.
5. Add field-aware reranking for the retrieved pool: reward exact product/category phrases and active hard constraints, apply budget compatibility and exclusion penalties, then use stable rating/popularity tie-breakers. Deduplicate by `parent_asin`, cap output at `top_k`, and make ordering deterministic.
6. Add active clarification selection for vague sessions. Estimate attribute diversity/entropy over the current candidate pool for unfilled allowed attributes, prioritize attributes the evaluator can reveal and that materially partition candidates, and return a concise question with a valid `ask_attribute`. Continue returning useful recommendations while asking; return `null` when the request is sufficiently specific or after the candidate set is confidently focused.
7. Implement boundary and override behavior explicitly: “no preference” answers remove the pending constraint rather than filtering everything out; out-of-catalog requests relax soft constraints while preserving the broadest compatible category/use case; new intent replaces obsolete category and dependent constraints before retrieval.
8. Harden the response contract: validate allowed `ask_attribute` values, valid unique catalog IDs, `message` type, and optional usage fields. Handle unknown sessions predictably and keep session state isolated across `session_id`s.
9. Add focused tests for reset/isolation, slot extraction and negation, override replacement, browsing clarification, boundary relaxation, deterministic valid recommendations, and response-schema normalization. Reuse evaluator test helpers or add small synthetic catalog/session fixtures without depending on the unavailable 50k catalog.
10. Run focused unit tests first, then `python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results.json` once the catalog is present. Report HitRate@10, MRR, MTTC, efficiency, and technical score, plus scenario breakdowns and any residual gaps.

**Relevant files**
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/starter/agent.py` — primary implementation surface; reuse `Agent._build_index`, `_text`, and `_terms`, replacing the current stateless `reset`/`respond` path.
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/evaluator/local_evaluator.py` — source of truth for turn simulation, hidden-attribute reveal behavior, recommendation normalization, metrics, and scenario handling; use `customer_reply`, `behavior_for`, and `normalize_recommendations` as implementation constraints.
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/tests/test_evaluator.py` — existing contract/metric tests and place for focused regression coverage.
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/docs/agent_api_contract.json` — response schema and allowed attributes.
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/docs/competition_specification.md` — catalog fields, dialogue behavior, and challenge requirements.
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/docs/submission_rules.md` — offline/dependency/package constraints.
- `/Users/soma/Desktop/tttechjam/techjam-conversational-search/data/public_set.jsonl` — public sessions for evaluator validation; the catalog expected by those sessions is not currently present.

**Verification**
1. Run `python3 -m unittest discover -s tests -v` before and after changes.
2. Use a tiny synthetic catalog fixture to exercise `reset()` plus several `respond()` turns, including vague browsing followed by the requested attribute, an intent override, a negation, and a no-preference boundary answer.
3. Assert every response has a valid allowed `ask_attribute`, unique IDs, IDs drawn from the loaded catalog, and no exception across turns 1-10.
4. Run the full evaluator command with `data/catalog.jsonl` when the release data is available; inspect the JSON output with `python3 -m json.tool results.json`.
5. Compare aggregate and scenario metrics against `docs/baseline_results.json`; investigate misses by turn and scenario before any optimization iteration.

**Decisions**
- Use deterministic standard-library methods rather than adding BM25, embeddings, FAISS, or an LLM dependency, because the repository has no dependency manifest and submission rules require offline execution. The existing SQLite FTS index is the retrieval foundation.
- Treat evaluator semantics as authoritative: `ask_attribute` drives information revelation, recommendations are scored only by exact catalog `parent_asin`, and invalid/duplicate IDs do not count.
- Keep changes focused on the agent and narrow tests; do not modify evaluator scoring or fabricate the missing catalog.
- Full 50k-catalog benchmark results cannot be produced in this workspace until `data/catalog.jsonl` is supplied.

**Further Considerations**
1. Dense embeddings and cross-encoder reranking are excluded from the initial implementation because they conflict with the current dependency/offline shape; add them only if the submission environment explicitly permits vendored model assets and the lexical baseline has been measured.
2. The public dataset should be used for regression scoring, but synthetic fixtures are required for fast development because the actual catalog is unavailable here.
