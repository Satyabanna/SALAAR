# SALAAR

> You think. You drop the ideas. The agent connects the dots.

An idea canvas where the edges are the product. Every connection the agent proposes carries a
**type**, a **rationale quoting both ideas**, a **confidence**, a **semantic similarity**, and a
**gravitational pull** — and none of it enters the graph until you press **✓**.

```bash
export OLLAMA_API=...                      # any OpenAI-compatible key
export BASE_URL=https://ollama.com/v1
export SALAAR_MODEL=gpt-oss:120b
export API_KEY_VAR=OLLAMA_API
export LLM_TIMEOUT=90                      # see "Read this before you demo"
python3 app.py                             # http://localhost:8000
```

No dependencies — Python 3 stdlib only. No key? It still runs: every model path degrades to a
deterministic local result and the badge reads `offline mode`.

## Read this before you demo

**`LLM_TIMEOUT` defaults to 5 seconds, and a real relationship-typing call takes longer than
that.** Measured here: the 7-node demo canvas types in **6.5s**. At the default the socket is
cut at 5s, the exception is caught, and the canvas silently serves the scripted fallback —
`mode: fallback` with a valid key, and nothing in the UI says so except the badge. Set
`LLM_TIMEOUT=90` or you will demo the offline path by accident.

| `LLM_TIMEOUT` | Result on the 7-node seed |
|---|---|
| `5` (default) | `mode: fallback`, 10 scripted/lexical edges, 5.2s |
| `90` | `mode: llm`, 10 typed edges, 6.5s |

## Run it

| | |
|---|---|
| Local | `python3 app.py` — stdlib HTTP server, no framework, no dependencies |
| Docker | `docker build -t salaar . && docker run -p 8000:8000 -e OLLAMA_API=$OLLAMA_API -v $PWD/data:/data salaar` |
| Render / Fly / Cloud Run | Dockerfile as-is; honours `$PORT`. Mount a volume at `/data` to keep canvases. |
| Self-check | `python3 test_app.py` — no network, no key, ~0.07s |

Each canvas is a URL: `localhost:8000/#ward-triage`.

## The design

### Embeddings are real, and they are optional

Every node carries a cached `semantic_representation`: a unit vector, the model that produced
it, and a content hash. Edit the idea and the hash changes, which invalidates that node's
vector and nothing else.

With no configuration the app uses a **local deterministic embedder**
(`local-semantic-hash-v1/192`): hashed word features, character 3-grams for morphology, and
eight hand-written semantic-family axes (healthcare, operations, prediction, agent,
collaboration, product, research, sustainability). It is genuinely modest — but it is a
fixed-size embedding, it needs no key, and it makes Tier 0 work offline. Set
`SALAAR_EMBEDDING_MODEL` to use any OpenAI-compatible `/embeddings` endpoint instead; if that
endpoint misbehaves the app disables it for the process and recaches locally rather than
turning a canvas click into a spinner.

### Gravity decides which pairs are worth a model call

```
pull = sqrt(m1*m2) * affinity^1.5 / (2*(1-cosine) + softening)
```

`2(1-cosine)` is the squared distance between unit embeddings. `affinity` is cosine rescaled
above `GRAVITY_THRESHOLD`, so unrelated vectors exert nothing. The softening term stops
duplicate ideas producing infinite force, and the result is capped.

Mass is semantic density plus **approved** graph evidence — never text length. Pending
suggestions add no mass, so the agent cannot inflate its own confidence by proposing edges.

Each node contributes its top `TOP_K` neighbours by pull, bounding model work at O(n·k), with
`MAX_PAIRS` as a hard global cap. Pairs already typed and pairs you rejected are filtered out
before the call, so adding one idea to a connected canvas costs a small call about that idea.

### One batched typing call

All surviving candidates go to the model in a single structured request returning a type from a
fixed vocabulary, a rationale, a confidence, and a `flip` flag. Edges are directed and read
`A <type> B`; `flip` is how the model says the relationship holds the other way round, which is
what stops *"Slack specializes AI-for-hospitals"* pointing backwards.

The response is treated as untrusted input twice over: `_validate_schema` checks the JSON shape,
then `_validated_typed_edges` rejects duplicate pair indices, out-of-range indices, non-boolean
`flip`, unknown relationship types, and non-finite confidences. Anything under 0.35 confidence
is dropped.

### Clusters are the accepted graph, then semantics

Union-find over **accepted, non-adversarial** edges gives the base components — `contradicts`
and `duplicates` never merge, because two ideas that fight are not one idea, and pending
proposals never merge, because the user has not agreed yet. Those components are then
agglomerated while their centres of gravity stay within `CLUSTER_SIMILARITY`.

Each cluster reports its centre of gravity, the member nearest that centre (its **core**), mass,
cohesion, and per-member pull. There is deliberately **no global centre** — one would collapse
unrelated themes into a single blob.

### Layout is a suggestion, never a reset

`POST /api/gravity` runs a deterministic force simulation — pair springs from gravitational
pull, per-cluster attraction to each cluster's own centre, spatial repulsion, and a weak anchor
back to where you put things. **It never mutates saved coordinates.** The browser animates to
the suggestion only when you click *Arrange by gravity*.

### Your decisions survive; the agent's guesses do not

Editing an idea invalidates the analysis that depended on it, and the two cases are handled
differently on purpose:

- A **pending** AI proposal about the edited idea is discarded. It was reasoning about wording
  that no longer exists.
- An **accepted** edge is kept and flagged `needs_review` with a reason and timestamp. Deleting
  it would silently throw away a human decision. It renders amber and dashed, is excluded from
  cluster mass, and waits for your ✓, ✎, or ✗.

Rejecting an edge writes the pair to a denylist stamped with both endpoints' content hashes, so
it is never re-proposed — until one of those ideas is reworded, which retires the denylist entry
along with the wording it was about.

### Concurrency

Every canvas has a `revision`. A debounced browser save that arrives after an analysis action
has committed is refused and handed the current graph, so a slow model call can never overwrite
newer work. Model and embedding calls run **outside** the lock; each committing path re-reads
and compares the revision before writing, and reports `stale: true` rather than clobbering.

## API

| Method | Path | Does |
|---|---|---|
| `GET` | `/api/graph/<id>` | Load a canvas, migrating legacy shapes forward |
| `PUT` | `/api/graph/<id>` | Save, guarded by `revision` |
| `POST` | `/api/connect/<id>` | Candidate selection → one typing call → merge proposals |
| `POST` | `/api/synthesize/<id>` | One call: core, gaps, contradictions, refined concept, next action |
| `POST` | `/api/gravity/<id>` | Suggest a layout; returns positions and cluster geometry |
| `POST` | `/api/clusters/<id>` | Recompute clusters and centres of gravity |
| `POST` | `/api/edge/<id>` | Accept, edit, or reject one proposal by `edge_id` |
| `POST` | `/api/chat/<id>` | Ask a question with the canvas as context |
| `POST` | `/api/extract/<id>` | Transcript → ideas, added as `team_chat` nodes |

Canvas ids are validated against `[A-Za-z0-9_-]{1,64}`, so no path traversal (verified: `404`).

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `BASE_URL` | NVIDIA NIM | Any OpenAI-compatible `/chat/completions` |
| `SALAAR_MODEL` | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | `detailed thinking off` is sent automatically for Nemotron |
| `API_KEY_VAR` | `NVIDIA_API_KEY` | Name of the variable holding the key, not the key |
| `LLM_TIMEOUT` | `5` | **Too low. See above.** |
| `SALAAR_EMBEDDING_MODEL` | unset | Unset = local deterministic embedder |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY_VAR` / `EMBEDDING_TIMEOUT` | inherit chat config | A chat model is not an embedding model |
| `EMBEDDING_DIMENSIONS` | `192` | Local embedder width |
| `TOP_K` / `MAX_PAIRS` | `4` / `60` | Neighbours per node; hard cap on pairs per call |
| `CLUSTER_SIMILARITY` | `0.18` | Cluster merge threshold |
| `GRAVITY_THRESHOLD` / `GRAVITY_SOFTENING` / `GRAVITY_CAP` | `0.03` / `0.28` / `8` | Force shape |
| `CANDIDATE_SCORE_FLOOR` | `0.015` | Floor below which a neighbour is noise |
| `DATA_DIR` / `PORT` | `data` / `8000` | |

## Observed on this machine

Ollama Cloud, 7-node demo canvas, 2026-09-12.

| Check | Result |
|---|---|
| `python3 test_app.py` | `ok`, 0.07s |
| `connect`, `LLM_TIMEOUT=5` | `mode: fallback`, 10 edges, 5.2s |
| `connect`, `LLM_TIMEOUT=90` | `mode: llm`, 10 edges, 6.5s |
| `connect` again | `mode: cached`, 0 new — typed pairs are not re-sent |
| `gravity` | 7 positions, 3 layout clusters, radius 341.7 on the 5-node cluster |
| `clusters` | mass 6.62, cohesion 0.69, core `s5` |
| `edge` accept | `state: accepted`, endpoints promoted to `connected` |
| `edge` reject | denylist written; pair not re-proposed on reconnect |
| `synthesize` | model-backed: *"An AI-driven agent that predicts and explains upcoming bottlenecks in a single hospital ward"* |
| `extract` | 2 ideas added from a Slack snippet as `team_chat` |
| traversal / bad state | `404` / `400 state must be accepted or rejected` |

Model comparison from an earlier run, one sample each — not a benchmark: `gpt-oss:120b` was
fastest and gave the only rationales that explained rather than restated; `nemotron-3-super`
reasoned well but ran ~3× slower; `nemotron-3-nano:30b` template-filled.

## Where the BRD landed

| BRD | Here |
|---|---|
| FR-01…03 canvas, persistence | `static/index.html`, `PUT /api/graph` with revisions |
| FR-04 semantic representation | Cached per-node embeddings, hash-invalidated |
| FR-05 relationship discovery | Gravity-ranked candidates → one batched typing call |
| FR-06 visualisation | SVG edges coloured by relationship family, cluster halos, COG markers |
| FR-07 clustering | Accepted-edge union-find, then COG agglomeration |
| FR-08/09/10 core, critique, optimize | One `/api/synthesize` call |
| FR-11 evidence | `why` must quote both ideas; carried as `evidence` |
| FR-12 chat | `/api/chat` |
| FR-13 conversation extraction | `/api/extract`, source `team_chat` |
| FR-14 human control | `pending` → ✓ / ✎ / ✗, denylist, `needs_review` on edit |
| §6.1 Tier 0 | Scripted demo edges + local embedder; `test_app.py` asserts the degraded paths |
| §9 data model | `content`, `position`, `source`, `timestamp`, `status`, `semantic_representation`; edges carry `source_node`, `target_node`, `relationship_type`, `evidence`, `confidence` |
| §9 idea state | `raw → understood → clustered → connected` |

**Deliberately not built:** auth, multi-user, a database, WebSockets, undo history, a real Slack
app. Add auth and Postgres the day a second team uses it.

## Known issues

1. **`LLM_TIMEOUT=5` guarantees silent degradation.** One-line fix, documented rather than
   changed, since the default may be someone's deliberate choice.
2. **The HTTP handler has no automated test.** `test_app.py` covers the engine thoroughly but
   calls no endpoint — which is why the crash below reached a running server.
3. **Fixed in this pass:** `POST /api/edge` with `state: rejected` raised
   `unhashable type: 'list'` on every call (`sorted()` returns a list, tested against a set of
   tuples), so ✗ was dead and FR-14 was half-broken. Now a tuple.

## Demo script (2 min)

1. `Seed demo` — 7 disconnected ideas.
2. `Connect the dots` — typed edges, dashed until approved, rationale on hover.
3. `✓` two, `✎` one to correct its type, `✗` one — the rejected pair is now permanently out.
4. `Arrange by gravity` — clusters separate, each around its own centre; halos show cohesion.
5. Select the ward cluster → `Synthesize` → core, gaps, contradictions, refined concept.
6. **Ask a judge for one new idea.** Double-click, type it, `Connect the dots` — only the new
   pairs are sent.
7. Edit one idea's wording and watch its accepted edges turn amber for review.
8. Paste a Slack snippet into *Import a team conversation* → same graph, new source.
