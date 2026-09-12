"""SALAAR — idea canvas that types and explains the edges between ideas.

System shape (one process, no services):

    browser ──PUT /api/graph/<id>──▶  graph store (JSON file per canvas, one lock)
            ──POST /api/connect ───▶  pass 1: cached embeddings + gravity-ranked neighbours
                                      pass 2: ONE batched LLM call     (type + rationale + confidence)
                                      merge: skip typed pairs & user-rejected pairs (incremental)
            ──POST /api/synthesize─▶  ONE LLM call over a cluster -> core / gaps / contradictions / refined
            ──POST /api/chat ──────▶  ONE LLM call, canvas as context
            ──POST /api/extract ───▶  ONE LLM call, transcript -> nodes (Slack = just another source)

Every LLM path degrades to a deterministic local result instead of failing, so the
demo cannot break on stage (BRD 6.1 Tier 0).
"""

import hashlib
import json
import math
import os
import re
import urllib.request
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Any OpenAI-compatible endpoint. Defaults to NVIDIA NIM / Nemotron.
BASE_URL = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("SALAAR_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
KEY_VAR = os.getenv("API_KEY_VAR", "NVIDIA_API_KEY")
API_KEY = os.getenv(KEY_VAR) or os.getenv("OPENAI_API_KEY") or ""
DATA = Path(os.getenv("DATA_DIR", "data"))
STATIC = Path(__file__).parent / "static"
TOP_K = max(1, int(os.getenv("TOP_K", "4")))    # top semantic neighbours per node
MAX_PAIRS = max(0, int(os.getenv("MAX_PAIRS", "60")))  # global candidate safety cap; 0 = no cap

# Embeddings deliberately have their own configuration: a chat model is not
# necessarily an embedding model. Set SALAAR_EMBEDDING_MODEL to use any
# OpenAI-compatible /embeddings endpoint. With no setting, the app uses the
# deterministic local semantic embedder below, so Tier 0 still works offline.
REMOTE_EMBEDDING_MODEL = os.getenv("SALAAR_EMBEDDING_MODEL", "").strip()
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", BASE_URL)
EMBEDDING_KEY_VAR = os.getenv("EMBEDDING_API_KEY_VAR", KEY_VAR)
EMBEDDING_API_KEY = os.getenv(EMBEDDING_KEY_VAR) or API_KEY
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "5"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "5"))
EMBEDDING_DIMENSIONS = max(48, int(os.getenv("EMBEDDING_DIMENSIONS", "192")))
LOCAL_EMBEDDING_MODEL = f"local-semantic-hash-v1/{EMBEDDING_DIMENSIONS}"
CLUSTER_SIMILARITY = float(os.getenv("CLUSTER_SIMILARITY", "0.18"))
GRAVITY_THRESHOLD = float(os.getenv("GRAVITY_THRESHOLD", "0.03"))
GRAVITY_SOFTENING = float(os.getenv("GRAVITY_SOFTENING", "0.28"))
GRAVITY_CAP = float(os.getenv("GRAVITY_CAP", "8"))
CANDIDATE_SCORE_FLOOR = float(os.getenv("CANDIDATE_SCORE_FLOOR", "0.015"))

VOCAB = ["relates_to", "expands", "specializes", "depends_on", "contradicts",
         "duplicates", "solves", "causes", "can_combine_with", "implements"]

# ponytail: one global lock over all canvases. Per-canvas locks only if two
# demos ever run concurrently on one box.
LOCK = threading.Lock()
LOCAL_EMBEDDING_CACHE = {}
REMOTE_EMBEDDINGS_DISABLED = False


# ---------------------------------------------------------------- graph store

def _file(gid):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", gid):  # trust boundary: no path traversal
        raise ValueError("bad canvas id")
    DATA.mkdir(parents=True, exist_ok=True)
    return DATA / f"{gid}.json"


def load(gid):
    f = _file(gid)
    if f.exists():
        return json.loads(f.read_text())
    return {"nodes": [], "edges": [], "rejected": []}


def save(gid, g):
    _file(gid).write_text(json.dumps(g))
    return g


def _revision(graph):
    try:
        return max(0, int(graph.get("revision", 0)))
    except (AttributeError, TypeError, ValueError):
        return 0


def commit(gid, graph):
    """Advance the server revision for an API mutation before persisting it."""
    graph["revision"] = _revision(graph) + 1
    return save(gid, graph)


# ------------------------------------------------- pass 1: embeddings + gravity

# The local model is intentionally modest but useful: stable hashed word and
# character features preserve lexical evidence, while a tiny public vocabulary
# gives common brainstorm synonyms a shared direction. It is a real fixed-size
# embedding, not TF-IDF, and lets the product remain dependency- and key-free.
SEMANTIC_FAMILIES = (
    ("healthcare", {"hospital", "clinic", "ward", "patient", "nurse", "doctor", "medical", "health", "healthcare"}),
    ("operations", {"wait", "waiting", "queue", "bottleneck", "workflow", "process", "triage", "capacity", "dashboard"}),
    ("prediction", {"predict", "prediction", "forecast", "detect", "early", "monitor", "signal", "measure", "metric"}),
    ("agent", {"agent", "assistant", "ai", "automation", "automate", "investigate", "reason", "analysis", "intelligence"}),
    ("collaboration", {"slack", "chat", "team", "conversation", "meeting", "message", "collaborate", "communication"}),
    ("product", {"product", "user", "customer", "market", "mvp", "feature", "prototype", "build", "launch"}),
    ("research", {"research", "study", "evidence", "experiment", "hypothesis", "learn", "data", "insight"}),
    ("sustainability", {"climate", "carbon", "energy", "sustainable", "sustainability", "waste", "environment"}),
)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "we", "with", "will", "would", "should", "could", "not", "just",
}


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _node_text(node):
    """Read both the v1 `text` field and the BRD's canonical `content` field."""
    value = node.get("content")
    if value is None:
        value = node.get("text", "")
    return str(value).strip()


def _terms(text):
    """Small, deterministic normaliser shared by local embeddings and fallback."""
    out = []
    for raw in re.findall(r"[a-z0-9]{2,}", str(text).lower()):
        token = raw
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token not in STOP_WORDS:
            out.append(token)
    return out


def _text_digest(text):
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def _unit(values):
    """Return a finite, L2-normalised vector or None for an invalid cache entry."""
    try:
        vector = [float(x) for x in values]
    except (TypeError, ValueError):
        return None
    if not vector or not all(math.isfinite(x) for x in vector):
        return None
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else None


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _feature_index(namespace, value, dimensions):
    digest = hashlib.blake2b(f"{namespace}:{value}".encode(), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % dimensions, -1.0 if number & 1 else 1.0


def local_embedding(text):
    """A deterministic semantic embedding for offline/demo operation.

    Every value has stable dimensions, so it can be cached with the node and
    compared across page reloads. Character n-grams make morphology robust;
    semantic-family axes handle high-value, obvious paraphrases without a model.
    """
    cache_key = (LOCAL_EMBEDDING_MODEL, _text_digest(text))
    cached = LOCAL_EMBEDDING_CACHE.get(cache_key)
    if cached:
        return list(cached)

    family_dimensions = len(SEMANTIC_FAMILIES)
    hashed_dimensions = max(16, EMBEDDING_DIMENSIONS - family_dimensions)
    vector = [0.0] * EMBEDDING_DIMENSIONS
    terms = _terms(text)
    for term in terms:
        index, sign = _feature_index("word", term, hashed_dimensions)
        vector[index] += sign
        # Subword features catch hospital/hospitals and investigate/investigation.
        grams = [term[i:i + 3] for i in range(max(0, len(term) - 2))]
        for gram in grams:
            index, sign = _feature_index("gram", gram, hashed_dimensions)
            vector[index] += sign * 0.16
        for family_index, (_, vocabulary) in enumerate(SEMANTIC_FAMILIES):
            if term in vocabulary:
                vector[hashed_dimensions + family_index] += 1.8

    # An empty node has no direction. It is safely ignored by candidate ranking.
    embedding = _unit(vector) or [0.0] * EMBEDDING_DIMENSIONS
    LOCAL_EMBEDDING_CACHE[cache_key] = tuple(embedding)
    return embedding


def _remote_embeddings(texts):
    if not REMOTE_EMBEDDING_MODEL or not EMBEDDING_API_KEY:
        raise RuntimeError("embedding endpoint is not configured")
    request = urllib.request.Request(
        EMBEDDING_BASE_URL.rstrip("/") + "/embeddings",
        data=json.dumps({"model": REMOTE_EMBEDDING_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {EMBEDDING_API_KEY}"})
    with urllib.request.urlopen(request, timeout=EMBEDDING_TIMEOUT) as response:
        payload = json.load(response)
    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [_unit(item.get("embedding", [])) for item in data]
    if len(vectors) != len(texts) or any(vector is None for vector in vectors):
        raise ValueError("embedding response was incomplete or invalid")
    return vectors


def _active_embedding_model():
    return LOCAL_EMBEDDING_MODEL if REMOTE_EMBEDDINGS_DISABLED or not REMOTE_EMBEDDING_MODEL else REMOTE_EMBEDDING_MODEL


def _ensure_node_fields(node, source="canvas"):
    """Migrate v1 nodes lazily without breaking existing saved canvases."""
    text = _node_text(node)
    node["text"] = text
    node.setdefault("content", text)
    position = node.get("position") if isinstance(node.get("position"), dict) else {}
    node.setdefault("x", position.get("x", 0))
    node.setdefault("y", position.get("y", 0))
    try:
        node["x"] = float(node["x"])
        node["y"] = float(node["y"])
    except (TypeError, ValueError):
        node["x"], node["y"] = 0, 0
    if not math.isfinite(node["x"]) or not math.isfinite(node["y"]):
        node["x"], node["y"] = 0, 0
    node["position"] = position
    node["position"]["x"] = node["x"]
    node["position"]["y"] = node["y"]
    node.setdefault("source", source)
    node.setdefault("timestamp", _now())
    node.setdefault("status", "raw")
    return node


def _prepare_graph(graph):
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])
    graph.setdefault("rejected", [])
    graph.setdefault("clusters", [])
    graph.setdefault("revision", 0)
    for node in graph["nodes"]:
        _ensure_node_fields(node)
    by_id = {node["id"]: node for node in graph["nodes"]}
    seen_edge_ids = set()
    for index, edge in enumerate(graph["edges"]):
        # Edge IDs make approval actions address a stable proposal rather than a
        # mutable array slot.  Legacy graphs get a deterministic ID once and
        # then persist it on their next write.
        edge_id = edge.get("id")
        if not isinstance(edge_id, str) or not edge_id or edge_id in seen_edge_ids:
            edge_id = "e-" + hashlib.sha1(
                f"{index}|{edge.get('src', edge.get('source_node', ''))}|"
                f"{edge.get('dst', edge.get('target_node', ''))}|{edge.get('why', '')}".encode()
            ).hexdigest()[:16]
            while edge_id in seen_edge_ids:
                edge_id += "x"
            edge["id"] = edge_id
        seen_edge_ids.add(edge_id)
        edge.setdefault("src", edge.get("source_node"))
        edge.setdefault("dst", edge.get("target_node"))
        edge.setdefault("source_node", edge.get("src"))
        edge.setdefault("target_node", edge.get("dst"))
        edge.setdefault("type", edge.get("relationship_type", "relates_to"))
        edge.setdefault("relationship_type", edge.get("type"))
        edge.setdefault("why", edge.get("evidence", ""))
        edge.setdefault("evidence", edge.get("why", ""))
        edge.setdefault("conf", edge.get("confidence", 0.5))
        edge.setdefault("confidence", edge.get("conf"))
        source, target = by_id.get(edge.get("src")), by_id.get(edge.get("dst"))
        # Legacy edges predate hashes. Their current endpoint text is their
        # baseline; all new edges retain the hashes that created them.
        if source:
            edge.setdefault("source_content_hash", _text_digest(_node_text(source)))
        if target:
            edge.setdefault("target_content_hash", _text_digest(_node_text(target)))
        edge["needs_review"] = bool(edge.get("needs_review", False))
    return graph


def _rejection_pair(entry):
    if isinstance(entry, dict):
        pair = entry.get("pair") or (entry.get("src"), entry.get("dst"))
    else:
        pair = entry
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    return tuple(sorted((str(pair[0]), str(pair[1]))))


def _drop_incident_relations(graph, changed_ids):
    """Discard stale AI proposals but retain human decisions for explicit review."""
    if not changed_ids:
        return
    retained = []
    for edge in graph["edges"]:
        incident = edge.get("src") in changed_ids or edge.get("dst") in changed_ids
        if not incident:
            retained.append(edge)
        elif edge.get("state", "accepted") == "accepted":
            # An accepted relationship belongs to the user.  Its old evidence
            # may no longer apply after a typo/rewrite, but deleting it would
            # silently discard a deliberate decision.  Keep it visible and
            # exclude it from semantic mass until the user confirms or edits it.
            edge["needs_review"] = True
            edge["review_reason"] = "An endpoint idea changed"
            edge["review_requested_at"] = _now()
            retained.append(edge)
    graph["edges"] = retained
    graph["rejected"] = [entry for entry in graph["rejected"]
                         if not (pair := _rejection_pair(entry)) or not (set(pair) & set(changed_ids))]


def _drop_stale_relations(graph):
    """Drop stale proposals while turning stale accepted edges into review items."""
    by_id = {node["id"]: node for node in graph["nodes"]}
    valid_edges = []
    for edge in graph["edges"]:
        source, target = by_id.get(edge.get("src")), by_id.get(edge.get("dst"))
        if not source or not target:
            continue
        current = (edge.get("source_content_hash") == _text_digest(_node_text(source)) and
                   edge.get("target_content_hash") == _text_digest(_node_text(target)))
        if not current:
            if edge.get("state", "accepted") == "accepted":
                edge["needs_review"] = True
                edge.setdefault("review_reason", "An endpoint idea changed")
                valid_edges.append(edge)
            # Pending AI suggestions are based on the old wording and should
            # be regenerated rather than presented as current evidence.
            continue
        valid_edges.append(edge)
    graph["edges"] = valid_edges

    valid_rejections = []
    for entry in graph["rejected"]:
        if not isinstance(entry, dict):  # legacy denylist has no text baseline
            valid_rejections.append(entry)
            continue
        source, target = by_id.get(entry.get("src")), by_id.get(entry.get("dst"))
        if not source or not target:
            continue
        if (entry.get("source_content_hash") != _text_digest(_node_text(source)) or
                entry.get("target_content_hash") != _text_digest(_node_text(target))):
            continue
        valid_rejections.append(entry)
    graph["rejected"] = valid_rejections


def _current_rejected_pairs(graph):
    return {pair for entry in graph["rejected"] if (pair := _rejection_pair(entry))}


def ensure_embeddings(nodes):
    """Fill or reuse per-node cached embeddings, invalidating them on text/model change."""
    global REMOTE_EMBEDDINGS_DISABLED
    for node in nodes:
        _ensure_node_fields(node)

    expected_model = _active_embedding_model()
    missing = []
    vectors = {}
    for index, node in enumerate(nodes):
        text_hash = _text_digest(_node_text(node))
        semantic = node.get("semantic_representation") or node.get("semantic") or {}
        cached = _unit(semantic.get("vector", [])) if isinstance(semantic, dict) else None
        if (cached and semantic.get("content_hash") == text_hash and
                semantic.get("model") == expected_model):
            vectors[node["id"]] = cached
        else:
            missing.append(index)

    if missing and expected_model != LOCAL_EMBEDDING_MODEL:
        try:
            remote = _remote_embeddings([_node_text(nodes[index]) for index in missing])
            for index, vector in zip(missing, remote):
                vectors[nodes[index]["id"]] = vector
        except Exception as exc:
            # A misconfigured embedding endpoint must not turn a canvas action into
            # a spinner. Disable it for this process and consistently recache local.
            print(f"[salaar] embedding fallback: {exc!r}")
            REMOTE_EMBEDDINGS_DISABLED = True
            return ensure_embeddings(nodes)

    for index in missing:
        node = nodes[index]
        vector = vectors.get(node["id"])
        if vector is None:
            vector = local_embedding(_node_text(node))
            vectors[node["id"]] = vector
        node["semantic_representation"] = {
            "model": _active_embedding_model(),
            "content_hash": _text_digest(_node_text(node)),
            "dimensions": len(vector),
            "vector": [round(value, 7) for value in vector],
            "updated_at": _now(),
        }
        # "understood" is a fact about embedding, not an implicit acceptance.
        if node.get("status") == "raw":
            node["status"] = "understood"
    return vectors


def _accepted_strengths(nodes, edges):
    """Only human-approved, non-adversarial edges contribute conceptual mass."""
    strength = {node["id"]: 0.0 for node in nodes}
    for edge in edges:
        if (edge.get("state", "accepted") != "accepted" or edge.get("needs_review") or
                edge.get("type") in ("contradicts", "duplicates")):
            continue
        try:
            confidence = max(0.0, min(1.0, float(edge.get("conf", edge.get("confidence", 0.5)))))
        except (TypeError, ValueError):
            continue
        for node_id in (edge.get("src"), edge.get("dst")):
            if node_id in strength:
                strength[node_id] += confidence
    return strength


def _node_masses(nodes, vectors, edges):
    """Mass reflects semantic density plus approved graph evidence, never text length."""
    accepted = _accepted_strengths(nodes, edges)
    masses = {}
    for node in nodes:
        vector = vectors[node["id"]]
        neighbours = sorted((_cosine(vector, vectors[other["id"]]) for other in nodes if other is not node), reverse=True)
        positive = [similarity for similarity in neighbours if similarity > 0][:3]
        density = sum(positive) / len(positive) if positive else 0.0
        mass = 1.0 + 0.75 * density + 0.75 * min(1.0, accepted[node["id"]] / 2.0)
        masses[node["id"]] = max(1.0, min(3.0, mass))
    return masses


def centre_of_gravity(vectors, weights=None):
    """Unit weighted centroid in embedding space; the conceptual centre of mass."""
    vectors = [vector for vector in vectors if vector]
    if not vectors:
        return []
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        return []
    weights = list(weights or [1.0] * len(vectors))
    if len(weights) != len(vectors):
        raise ValueError("one centre-of-gravity weight is required per embedding")
    total = sum(max(0.0, float(weight)) for weight in weights) or 1.0
    centre = [0.0] * dimensions
    for vector, weight in zip(vectors, weights):
        weight = max(0.0, float(weight))
        for index, value in enumerate(vector):
            centre[index] += value * weight / total
    return _unit(centre) or [0.0] * dimensions


# American spelling is convenient for integrations; the BRD-facing spelling above
# makes the product concept explicit.
center_of_gravity = centre_of_gravity


def gravitational_pull(similarity, mass_a=1.0, mass_b=1.0, threshold=None):
    """Bounded attraction in semantic space.

    `r² = 2(1-cosine)` is the squared distance between unit embeddings. A
    softening term prevents duplicate ideas from producing infinite force; the
    affinity threshold prevents unrelated vectors being pulled together.
    """
    threshold = GRAVITY_THRESHOLD if threshold is None else threshold
    similarity = max(-1.0, min(1.0, float(similarity)))
    affinity = max(0.0, (similarity - threshold) / max(1e-9, 1.0 - threshold))
    if not affinity:
        return 0.0
    distance_squared = 2.0 * (1.0 - similarity)
    pull = math.sqrt(max(0.0, mass_a) * max(0.0, mass_b)) * affinity ** 1.5
    return min(GRAVITY_CAP, pull / (distance_squared + GRAVITY_SOFTENING))


def union_find(nodes, edges):
    """Approved positive edges form graph components; pending proposals never do."""
    parent = {node["id"]: node["id"] for node in nodes}

    def find(node_id):
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    for edge in edges:
        if (edge.get("state", "accepted") != "accepted" or edge.get("needs_review") or
                edge.get("type") in ("contradicts", "duplicates")):
            continue
        source, target = edge.get("src"), edge.get("dst")
        if source not in parent or target not in parent:
            continue
        left, right = find(source), find(target)
        if left != right:
            parent[left] = right
    groups = {}
    for node_id in parent:
        groups.setdefault(find(node_id), []).append(node_id)
    return list(groups.values())


def _semantic_groups(nodes, vectors, edges):
    """Agglomerate semantically compatible centres, seeded by approved graph groups."""
    by_id = {node["id"]: node for node in nodes}
    groups = [set(group) for group in union_find(nodes, edges)]
    masses = _node_masses(nodes, vectors, edges)

    def centroid(group):
        ids = sorted(group)
        return centre_of_gravity([vectors[node_id] for node_id in ids], [masses[node_id] for node_id in ids])

    while len(groups) > 1:
        best = None
        for left in range(len(groups)):
            left_centre = centroid(groups[left])
            for right in range(left + 1, len(groups)):
                score = _cosine(left_centre, centroid(groups[right]))
                if best is None or score > best[0]:
                    best = (score, left, right)
        if not best or best[0] < CLUSTER_SIMILARITY:
            break
        _, left, right = best
        groups[left] |= groups[right]
        groups.pop(right)
    return groups, masses, by_id


def semantic_clusters(nodes, edges, vectors=None):
    """Return metadata for embedding clusters and their conceptual centres."""
    if not nodes:
        return []
    vectors = vectors or ensure_embeddings(nodes)
    groups, masses, by_id = _semantic_groups(nodes, vectors, edges)
    clusters = []
    for group in sorted(groups, key=lambda item: (min(item), len(item))):
        ids = sorted(group)
        centre = centre_of_gravity([vectors[node_id] for node_id in ids], [masses[node_id] for node_id in ids])
        affinities = {node_id: _cosine(vectors[node_id], centre) for node_id in ids}
        core_id = max(ids, key=lambda node_id: (affinities[node_id], masses[node_id], node_id))
        cluster_mass = sum(masses[node_id] for node_id in ids)
        pulls = {
            node_id: gravitational_pull(affinities[node_id], masses[node_id], cluster_mass)
            for node_id in ids
        }
        cohesion = sum(affinities.values()) / len(ids)
        label = _node_text(by_id[core_id])[:64] or "Untitled idea"
        cluster_id = "cluster-" + hashlib.sha1("|".join(ids).encode()).hexdigest()[:10]
        clusters.append({
            "id": cluster_id,
            "node_ids": ids,
            "label": label,
            "core_concept": "",
            "unresolved_questions": [],
            "mass": round(cluster_mass, 3),
            "cohesion": round(cohesion, 3),
            "centre_of_gravity": {
                "core_node_id": core_id,
                "embedding": [round(value, 5) for value in centre],
                "member_pull": {node_id: round(pulls[node_id], 3) for node_id in ids},
            },
        })
    return clusters


def candidates(nodes, k=TOP_K, edges=None):
    """Embedding candidates ranked by softened gravitational pull.

    Each node contributes at most `k` attractive neighbours, keeping LLM work
    O(n·k) while prioritising ideas nearest to a conceptual centre of gravity.
    When k covers every other node, this naturally becomes exhaustive.
    """
    if len(nodes) < 2:
        return []
    edges = edges or []
    vectors = ensure_embeddings(nodes)
    masses = _node_masses(nodes, vectors, edges)
    count = len(nodes)
    records = []
    per_node = [[] for _ in nodes]
    for left in range(count):
        for right in range(left + 1, count):
            similarity = _cosine(vectors[nodes[left]["id"]], vectors[nodes[right]["id"]])
            pull = gravitational_pull(similarity, masses[nodes[left]["id"]], masses[nodes[right]["id"]])
            record = {
                "a": nodes[left], "b": nodes[right], "sim": round(similarity, 4),
                "gravity": round(pull, 4), "score": round(pull + max(0.0, similarity) * 0.08, 4),
            }
            records.append(record)
            per_node[left].append(record)
            per_node[right].append(record)

    picked, seen = [], set()
    exhaustive = count - 1 <= max(1, k)
    for row in per_node:
        ordered = sorted(row, key=lambda pair: (pair["score"], pair["sim"], pair["gravity"]), reverse=True)
        for pair in ordered[:max(1, k)]:
            if not exhaustive and pair["score"] <= CANDIDATE_SCORE_FLOOR:
                continue
            key = tuple(sorted((pair["a"]["id"], pair["b"]["id"])))
            if key not in seen:
                seen.add(key)
                picked.append(pair)
    picked = sorted(picked, key=lambda pair: (pair["score"], pair["a"]["id"], pair["b"]["id"]), reverse=True)
    return picked[:MAX_PAIRS] if MAX_PAIRS else picked


def refresh_clusters(graph):
    """Persist semantic cluster/centre metadata after an intelligence action."""
    _prepare_graph(graph)
    previous = {cluster.get("id"): cluster for cluster in graph.get("clusters", []) if isinstance(cluster, dict)}
    vectors = ensure_embeddings(graph["nodes"])
    clusters = semantic_clusters(graph["nodes"], graph["edges"], vectors)
    for cluster in clusters:
        old = previous.get(cluster["id"])
        if old:
            cluster["core_concept"] = old.get("core_concept", "")
            cluster["unresolved_questions"] = old.get("unresolved_questions", [])
    graph["clusters"] = clusters
    graph["analysis"] = {"embedding_model": _active_embedding_model(), "updated_at": _now()}
    for node in graph["nodes"]:
        node.pop("cluster_id", None)
        node.pop("gravity_pull", None)
    for cluster in clusters:
        for node_id in cluster["node_ids"]:
            node = next((item for item in graph["nodes"] if item["id"] == node_id), None)
            if node:
                node["cluster_id"] = cluster["id"]
                node["gravity_pull"] = cluster["centre_of_gravity"]["member_pull"][node_id]
                if len(cluster["node_ids"]) > 1 and node.get("status") == "understood":
                    node["status"] = "clustered"
    return clusters


def _stable_direction(left_id, right_id):
    """Non-random direction for nodes that begin at exactly the same position."""
    digest = hashlib.blake2b(f"{left_id}|{right_id}".encode(), digest_size=4).digest()
    angle = int.from_bytes(digest, "big") / (2 ** 32) * math.tau
    return math.cos(angle), math.sin(angle)


def gravity_layout(graph, ids=None, iterations=72):
    """Suggest a deterministic physical layout from embedding-space gravity.

    Similarity becomes attraction only after it clears `GRAVITY_THRESHOLD`; each
    semantic cluster then pulls toward its own spatial projection of the embedding
    centre of gravity. Short-range repulsion and a weak anchor preserve readable,
    user-owned positions. This function *never mutates* node coordinates: the
    browser applies the suggestion only after the user clicks Arrange by gravity.
    """
    _prepare_graph(graph)
    selected = set(ids or [])
    nodes = [node for node in graph["nodes"] if not selected or node["id"] in selected]
    if not nodes:
        return {"positions": [], "clusters": [], "embedding_model": _active_embedding_model()}
    if len(nodes) == 1:
        node = nodes[0]
        return {"positions": [{"id": node["id"], "x": node["x"], "y": node["y"]}],
                "clusters": [], "embedding_model": _active_embedding_model()}

    ids_set = {node["id"] for node in nodes}
    edges = [edge for edge in graph["edges"] if edge.get("src") in ids_set and edge.get("dst") in ids_set]
    vectors = ensure_embeddings(nodes)
    masses = _node_masses(nodes, vectors, edges)
    clusters = semantic_clusters(nodes, edges, vectors)
    by_id = {node["id"]: node for node in nodes}
    positions = {
        node["id"]: [float(node.get("x", 0)), float(node.get("y", 0))]
        for node in nodes
    }
    original = {node_id: value[:] for node_id, value in positions.items()}
    velocity = {node["id"]: [0.0, 0.0] for node in nodes}

    # A stack of new nodes is common after chat import. Give it a stable seed so
    # force calculations can separate it without introducing visual randomness.
    if len({(round(value[0], 3), round(value[1], 3)) for value in positions.values()}) == 1:
        origin_x, origin_y = next(iter(positions.values()))
        for index, node in enumerate(sorted(nodes, key=lambda item: item["id"])):
            angle = math.tau * index / len(nodes)
            positions[node["id"]] = [origin_x + math.cos(angle) * 90, origin_y + math.sin(angle) * 90]

    pulls = candidates(nodes, k=TOP_K, edges=edges)
    for _ in range(max(1, min(int(iterations), 180))):
        force = {node["id"]: [0.0, 0.0] for node in nodes}
        # Pair springs implement the embedding gravitational pull.
        for pair in pulls:
            left, right = pair["a"]["id"], pair["b"]["id"]
            pull = pair["gravity"]
            if pull <= 0:
                continue
            dx = positions[right][0] - positions[left][0]
            dy = positions[right][1] - positions[left][1]
            distance = math.hypot(dx, dy)
            if distance < 0.001:
                dx, dy = _stable_direction(left, right)
                distance = 1.0
            target = 118 + 210 * (1 - max(0.0, pair["sim"]))
            spring = (distance - target) * 0.012 * min(3.5, pull)
            ux, uy = dx / distance, dy / distance
            force[left][0] += ux * spring
            force[left][1] += uy * spring
            force[right][0] -= ux * spring
            force[right][1] -= uy * spring

        # Each conceptual cluster has its own COG; there is deliberately no
        # global centre that would collapse unrelated themes into one blob.
        for cluster in clusters:
            member_ids = cluster["node_ids"]
            if len(member_ids) < 2:
                continue
            total_mass = sum(masses[node_id] for node_id in member_ids)
            centre_x = sum(positions[node_id][0] * masses[node_id] for node_id in member_ids) / total_mass
            centre_y = sum(positions[node_id][1] * masses[node_id] for node_id in member_ids) / total_mass
            member_pull = cluster["centre_of_gravity"]["member_pull"]
            for node_id in member_ids:
                pull = min(4.0, member_pull[node_id])
                force[node_id][0] += (centre_x - positions[node_id][0]) * 0.006 * pull
                force[node_id][1] += (centre_y - positions[node_id][1]) * 0.006 * pull

        # Collision avoidance stays spatial, independent of semantic similarity.
        for left, node_left in enumerate(nodes):
            for node_right in nodes[left + 1:]:
                a, b = node_left["id"], node_right["id"]
                dx = positions[b][0] - positions[a][0]
                dy = positions[b][1] - positions[a][1]
                distance = math.hypot(dx, dy)
                if distance < 0.001:
                    dx, dy = _stable_direction(a, b)
                    distance = 1.0
                if distance < 218:
                    repulsion = (218 - distance) * 0.032
                    ux, uy = dx / distance, dy / distance
                    force[a][0] -= ux * repulsion
                    force[a][1] -= uy * repulsion
                    force[b][0] += ux * repulsion
                    force[b][1] += uy * repulsion

        for node in nodes:
            node_id = node["id"]
            # The weak anchor makes this an arrangement suggestion, not a reset.
            force[node_id][0] += (original[node_id][0] - positions[node_id][0]) * 0.004
            force[node_id][1] += (original[node_id][1] - positions[node_id][1]) * 0.004
            velocity[node_id][0] = velocity[node_id][0] * 0.68 + force[node_id][0]
            velocity[node_id][1] = velocity[node_id][1] * 0.68 + force[node_id][1]
            positions[node_id][0] += velocity[node_id][0]
            positions[node_id][1] += velocity[node_id][1]

    rendered_clusters = []
    for cluster in clusters:
        member_ids = cluster["node_ids"]
        mass = sum(masses[node_id] for node_id in member_ids)
        centre_x = sum(positions[node_id][0] * masses[node_id] for node_id in member_ids) / mass
        centre_y = sum(positions[node_id][1] * masses[node_id] for node_id in member_ids) / mass
        radius = max(140.0, max(math.hypot(positions[node_id][0] - centre_x, positions[node_id][1] - centre_y)
                                for node_id in member_ids) + 118)
        copy = dict(cluster)
        # Canvas nodes are positioned by their top-left corner, while halos and
        # wires use the visual node centre (190×44). Return that same coordinate
        # system so a COG halo is not visibly offset after arrangement.
        copy["spatial_centre"] = {"x": round(centre_x + 95, 2), "y": round(centre_y + 22, 2),
                                  "radius": round(radius, 2)}
        rendered_clusters.append(copy)
    return {
        "positions": [{"id": node_id, "x": round(value[0], 2), "y": round(value[1], 2)}
                      for node_id, value in sorted(positions.items())],
        "clusters": rendered_clusters,
        "embedding_model": _active_embedding_model(),
        "formula": "pull = sqrt(m1*m2)*affinity^1.5/(2*(1-cosine)+softening)",
    }


# ------------------------------------------------------------------ pass 2: LLM

def _post(prompt, json_mode, max_tokens):
    """One OpenAI-compatible chat call over stdlib urllib.

    Every serious provider speaks this shape, so the app has no SDK and no
    dependency: point BASE_URL/MODEL/API_KEY at NVIDIA NIM (default), OpenAI,
    Groq, OpenRouter, or a local Ollama and nothing else changes.
    """
    if not API_KEY:
        raise RuntimeError(f"{KEY_VAR} not set")
    msgs = [{"role": "user", "content": prompt}]
    if "nemotron" in MODEL.lower():
        # Nemotron's documented reasoning switch. Measured on nemotron-3-nano:30b:
        # 12x less reasoning output for the same answer. Without it a 6-pair
        # connect call spends ~20s thinking and blows the <5s interaction budget.
        msgs.insert(0, {"role": "system", "content": "detailed thinking off"})
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0.3, "messages": msgs}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"]


def _clean(text):
    # Nemotron and other reasoning models narrate before they answer; the answer
    # is what comes after, and fenced code is still fenced.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M).strip()


def _validate_schema(value, schema, path="$"):
    """Strictly validate the small JSON shapes requested from models."""
    expected = schema.get("type")
    type_ok = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected in type_ok and not type_ok[expected](value):
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed value")
    if expected == "object":
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{path} missing keys {missing}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ValueError(f"{path} has unexpected keys {sorted(extras)}")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{path}.{key}")
    elif expected == "array":
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")


def llm_json(prompt, schema, max_tokens=8000):
    raw = _clean(_post(
        f"{prompt}\n\nReply with JSON only, matching this schema exactly:\n"
        f"{json.dumps(schema)}", True, max_tokens))
    m = re.search(r"[\[{].*[\]}]", raw, re.S)          # tolerate a stray preamble
    data = json.loads(m.group(0) if m else raw)
    _validate_schema(data, schema)
    return data


def llm_text(prompt, max_tokens=2000):
    return _clean(_post(prompt, False, max_tokens))


EDGE_SCHEMA = {
    "type": "object",
    "properties": {"edges": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "pair": {"type": "integer"},
            "type": {"type": "string", "enum": VOCAB + ["none"]},
            "why": {"type": "string"},
            "conf": {"type": "number"},
            "flip": {"type": "boolean"},
        },
        "required": ["pair", "type", "why", "conf", "flip"], "additionalProperties": False}}},
    "required": ["edges"], "additionalProperties": False,
}

TYPING_RULES = """Pick the most specific type that holds. Decision rule, in order:
- depends_on: one idea must exist/be true before the other can work.
- causes: one produces the other as an effect.
- solves: one is a remedy for a problem named by the other.
- implements / specializes: one is the concrete or narrower form of the other.
- expands / can_combine_with: one broadens the other, or they compose into something neither is alone.
- contradicts / duplicates: opposite intent, or the same intent restated.
- relates_to: ONLY when nothing above applies. Overusing it makes the graph worthless.
Return "none" for a pair with no real relationship — dropping a weak pair is better than inventing one.

Edges are DIRECTED and read "A <type> B". Check the direction before you answer: if the
relationship only holds the other way round ("B <type> A"), keep the type and set flip=true.
"Slack" is not a specialization of "AI for hospitals"; the deployment is the narrower thing.

"why" is one sentence, under 25 words, quoting a phrase from BOTH ideas, and written in the
direction you chose — when flip=true it must read B first, then A. Say WHY the relationship
holds, never just restate the type: "nurses reject dashboards, so Slack removes the new
surface" is useful; "A contradicts B" is not. conf is 0-1."""


def type_pairs(pairs):
    """One batched call for every candidate pair. Never one call per pair."""
    listing = "\n".join(
        f'{i}. A="{p["a"]["text"]}" | B="{p["b"]["text"]}"' for i, p in enumerate(pairs))
    out = llm_json(
        f"""You are typing edges in an idea graph. For each numbered pair, decide how idea A relates to idea B.

{TYPING_RULES}

Pairs:
{listing}""", EDGE_SCHEMA)
    return out["edges"]


def _validated_typed_edges(raw_edges, pair_count):
    """Treat model JSON as untrusted input despite the requested schema."""
    if not isinstance(raw_edges, list):
        return []
    cleaned, seen = [], set()
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        pair = edge.get("pair")
        relation = edge.get("type")
        why = edge.get("why")
        flip = edge.get("flip", False)
        try:
            confidence = float(edge.get("conf"))
        except (TypeError, ValueError):
            continue
        if (isinstance(pair, bool) or not isinstance(pair, int) or not 0 <= pair < pair_count or
                pair in seen or relation not in VOCAB + ["none"] or not isinstance(why, str) or
                not why.strip() or not isinstance(flip, bool) or not math.isfinite(confidence)):
            continue
        seen.add(pair)
        cleaned.append({"pair": pair, "type": relation, "why": why.strip()[:500],
                        "conf": max(0.0, min(1.0, confidence)), "flip": flip})
    return cleaned


def type_pairs_batched(pairs, size=None):
    """One structured relationship-typing call for the whole bounded candidate set.

    `size` remains accepted for backward compatibility with early builds, but is
    intentionally ignored: BRD §8.1 calls for one batched request after top-k
    candidate generation. `MAX_PAIRS` is the explicit bound that keeps the
    prompt and latency safe instead of silently multiplying model requests.
    """
    if not pairs:
        return [], "none"
    try:
        raw = type_pairs(pairs)
        cleaned = _validated_typed_edges(raw, len(pairs))
        if not isinstance(raw, list) or (raw and not cleaned):
            raise ValueError("relationship batch did not contain valid edges")
        return cleaned, "llm"
    except Exception as exc:
        print(f"[salaar] relationship batch fell back: {exc!r}")
        return _validated_typed_edges(fallback_pairs(pairs), len(pairs)), "fallback"


def fallback_pairs(pairs):
    """Deterministic Tier-0 fallback, including rehearsed demo responses."""
    demo_edges = (
        ("ai for hospitals", "patient waiting time is the real pain", "solves", 0.76,
         '"AI for hospitals" can reduce the "patient waiting time" pain.'),
        ("predict bottlenecks before they form", "patient waiting time is the real pain", "solves", 0.78,
         '"Predict bottlenecks" before they form to reduce "patient waiting time".'),
        ("the agent should investigate why, not just report", "predict bottlenecks before they form", "implements", 0.74,
         'The "agent should investigate why" implements "predict bottlenecks" with an explanation.'),
        ("could live in slack where the team already talks", "the agent should investigate why, not just report", "can_combine_with", 0.72,
         '"Could live in Slack" gives the "agent" a place where the team already talks.'),
        ("could live in slack where the team already talks", "nurses will not open another dashboard", "solves", 0.78,
         '"Could live in Slack" avoids the "another dashboard" nurses will not open.'),
        ("start with one ward, not the whole hospital", "ai for hospitals", "specializes", 0.73,
         '"One ward" is a narrower first version of "AI for hospitals".'),
    )

    def normal(text):
        return re.sub(r"\s+", " ", _node_text({"text": text}).lower()).strip()

    def excerpt(node):
        words = _node_text(node).split()
        return " ".join(words[:7]) + ("…" if len(words) > 7 else "")

    out = []
    for i, p in enumerate(pairs):
        left, right = normal(p["a"]["text"]), normal(p["b"]["text"])
        scripted = next((edge for edge in demo_edges if {left, right} == {edge[0], edge[1]}), None)
        if scripted:
            source, _, relation, confidence, why = scripted
            out.append({"pair": i, "type": relation, "conf": confidence, "why": why,
                        "flip": left != source})
            continue

        shared = sorted(set(_terms(p["a"]["text"])) & set(_terms(p["b"]["text"])))[:3]
        if p.get("gravity", 0) <= 0.015 and p.get("sim", 0) < 0.08:
            continue
        confidence = min(0.68, 0.38 + max(0.0, p.get("sim", 0)) * 0.5 + p.get("gravity", 0) * 0.04)
        evidence = f"both use {', '.join(shared)}" if shared else "their embeddings occupy the same semantic neighbourhood"
        out.append({"pair": i, "type": "relates_to", "conf": round(confidence, 2),
                    "why": f'Offline match: "{excerpt(p["a"])}" and "{excerpt(p["b"])}" — {evidence}.'})
    return out


def connect(g):
    _prepare_graph(g)
    _drop_stale_relations(g)
    nodes = g["nodes"]
    if len(nodes) < 2:
        refresh_clusters(g)
        return g, {"new": 0, "mode": "none", "clusters": g["clusters"]}
    typed = {tuple(sorted((e.get("src"), e.get("dst")))) for e in g["edges"] if e.get("src") and e.get("dst")}
    rejected = _current_rejected_pairs(g)
    fresh = [p for p in candidates(nodes, edges=g["edges"])
             if tuple(sorted((p["a"]["id"], p["b"]["id"]))) not in typed | rejected]
    if not fresh:
        refresh_clusters(g)
        return g, {"new": 0, "mode": "cached", "clusters": g["clusters"]}
    found, mode = type_pairs_batched(fresh)
    added = 0
    for e in found:
        p = fresh[e["pair"]] if 0 <= e.get("pair", -1) < len(fresh) else None
        try:
            confidence = float(e.get("conf", 0))
        except (TypeError, ValueError):
            continue
        if not p or e.get("type") == "none" or e.get("type") not in VOCAB or confidence < 0.35:
            continue
        src, dst = (p["b"], p["a"]) if e.get("flip") else (p["a"], p["b"])
        why = str(e.get("why", "")).strip()
        if not why:
            continue
        g["edges"].append({
            "id": "e-" + uuid.uuid4().hex,
            "src": src["id"], "dst": dst["id"], "source_node": src["id"], "target_node": dst["id"],
            "type": e["type"], "relationship_type": e["type"], "why": why, "evidence": why,
            "conf": round(max(0.0, min(1.0, confidence)), 2), "confidence": round(max(0.0, min(1.0, confidence)), 2),
            "semantic_similarity": p["sim"], "gravitational_pull": p["gravity"],
            "source_content_hash": _text_digest(_node_text(src)),
            "target_content_hash": _text_digest(_node_text(dst)),
            "state": "pending", "mode": mode, "timestamp": _now(),
        })
        added += 1
    refresh_clusters(g)
    return g, {"new": added, "mode": mode, "considered": len(fresh), "clusters": g["clusters"],
               "embedding_model": _active_embedding_model()}


# ------------------------------------------------------------------ synthesis

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "core": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "refined": {"type": "object", "properties": {
            "problem": {"type": "string"}, "solution": {"type": "string"},
            "mvp": {"type": "string"}},
            "required": ["problem", "solution", "mvp"], "additionalProperties": False},
        "next_action": {"type": "string"},
    },
    "required": ["core", "gaps", "contradictions", "refined", "next_action"],
    "additionalProperties": False,
}


def render(g, ids=None):
    keep = [n for n in g["nodes"] if ids is None or n["id"] in ids]
    kept = {n["id"] for n in keep}
    lines = [f'- [{n["id"]}] {_node_text(n)}' for n in keep]
    lines += [f'- {e.get("src")} --{e.get("type")}--> {e.get("dst")} ({e.get("why", "")})'
              for e in g["edges"]
              if e.get("state") != "rejected" and e.get("src") in kept and e.get("dst") in kept]
    return "\n".join(lines) or "(empty canvas)"


def _cluster_anchor(g, ids):
    """A factual COG anchor for synthesis; the model still writes the meaning."""
    selected = set(ids or [node["id"] for node in g["nodes"]])
    clusters = g.get("clusters") or refresh_clusters(g)
    relevant = [cluster for cluster in clusters if selected & set(cluster["node_ids"])]
    if not relevant:
        return "No embedding cluster is available yet."
    cluster = max(relevant, key=lambda item: len(selected & set(item["node_ids"])))
    core_id = cluster["centre_of_gravity"]["core_node_id"]
    core = next((node for node in g["nodes"] if node["id"] == core_id), None)
    return (f'Embedding centre of gravity: "{_node_text(core) if core else cluster["label"]}" '
            f'(semantic cohesion {cluster["cohesion"]:.2f}; cluster mass {cluster["mass"]:.2f}).')


def _attach_synthesis(g, ids, result):
    selected = set(ids or [node["id"] for node in g["nodes"]])
    for cluster in g.get("clusters", []):
        if selected & set(cluster["node_ids"]):
            cluster["core_concept"] = result.get("core", "")
            cluster["unresolved_questions"] = result.get("gaps", [])
    return result


def synthesize(g, ids):
    """UC-03 core, UC-04 critique and UC-05 optimize are one call. Three
    questions about the same cluster do not need three round trips."""
    _prepare_graph(g)
    refresh_clusters(g)
    anchor = _cluster_anchor(g, ids)
    try:
        result = llm_json(
            f"""These ideas and typed connections came off one brainstorming canvas.

{render(g, ids)}

{anchor}

Identify the underlying concept, what is missing, what conflicts, and the strongest
version of this idea. Ground every claim in the ideas above — do not add new subject
matter. gaps and contradictions: at most 3 each, one sentence each, empty if none.""",
            SYNTH_SCHEMA)
        return _attach_synthesis(g, ids, result)
    except Exception as exc:
        print(f"[salaar] synth fallback: {exc!r}")
        texts = [_node_text(n) for n in g["nodes"] if ids is None or n["id"] in ids]
        top = [t for t, _ in Counter(t for x in texts for t in _terms(x)).most_common(4)]
        result = {"core": "Offline summary — recurring themes: " + ", ".join(top),
                  "gaps": [], "contradictions": [],
                  "refined": {"problem": texts[0] if texts else "", "solution": "", "mvp": ""},
                  "next_action": "Reconnect the model to synthesise this cluster."}
        return _attach_synthesis(g, ids, result)


NODES_SCHEMA = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["ideas"], "additionalProperties": False,
}


def extract(transcript):
    """UC-06. Chat is just another source feeding the same graph engine."""
    try:
        ideas = llm_json(
            "Pull the distinct ideas out of this team conversation. One atomic idea per "
            "item, under 12 words, in the speaker's own vocabulary. Skip greetings, "
            "logistics and agreement noise.\n\n" + transcript[:8000], NODES_SCHEMA)["ideas"]
        return [str(idea).strip() for idea in ideas if str(idea).strip()][:12]
    except Exception as exc:
        print(f"[salaar] extract fallback: {exc!r}")
        return [s.strip() for s in re.split(r"[.\n]", transcript) if len(s.strip()) > 25][:8]


def ask(g, question):
    _prepare_graph(g)
    try:
        return llm_text(
            f"""Current idea canvas:

{render(g)}

Question: {question}

Answer in under 80 words, referring to specific ideas on the canvas. Say plainly if
the canvas does not contain the answer.""")
    except Exception as exc:
        print(f"[salaar] chat fallback: {exc!r}")
        return "Model unreachable — canvas still saved. Try again in a moment."


# ---------------------------------------------------------------------- server

def new_nodes(g, texts, source="imported_context"):
    _prepare_graph(g)
    base = max((n.get("y", 0) for n in g["nodes"]), default=120)
    for i, t in enumerate(texts):
        text = str(t).strip()
        if not text:
            continue
        digest = hashlib.sha1(f"{text}|{len(g['nodes'])}".encode()).hexdigest()[:8]
        x, y = 80 + (i % 4) * 240, base + 140 + (i // 4) * 120
        g["nodes"].append({
            "id": f"n{len(g['nodes'])}_{digest}", "text": text, "content": text,
            "x": x, "y": y, "position": {"x": x, "y": y}, "source": source,
            "timestamp": _now(), "status": "raw",
        })
    return g


def _changed_node_ids(previous, incoming):
    before = {node["id"]: _node_text(node) for node in previous.get("nodes", [])}
    after = {node["id"]: _node_text(node) for node in incoming.get("nodes", [])}
    return {node_id for node_id, text in before.items() if node_id in after and after[node_id] != text}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        if self.path.startswith("/api/graph/"):
            with LOCK:
                graph = load(self.path.split("/")[-1])
                _prepare_graph(graph)
                return self._send(graph)
        f = STATIC / "index.html"
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        gid = self.path.split("/")[-1]
        graph = self._body()
        if not isinstance(graph, dict):
            return self._send({"error": "graph must be an object"}, 400)
        with LOCK:
            current = load(gid)
            _prepare_graph(current)
            # A delayed debounce must never overwrite a graph that an analysis
            # action has just committed. Returning current state lets the browser
            # advance its revision without a destructive retry.
            if _revision(graph) != _revision(current):
                return self._send(current)
            _prepare_graph(graph)
            _drop_incident_relations(graph, _changed_node_ids(current, graph))
            _drop_stale_relations(graph)
            graph["revision"] = _revision(current) + 1
            self._send(save(gid, graph))

    def do_POST(self):
        parts = self.path.strip("/").split("/")      # api/<action>/<gid>
        if len(parts) != 3 or parts[0] != "api":
            return self._send({"error": "not found"}, 404)
        _, action, gid = parts
        body = self._body()
        try:
            # Model and embedding calls deliberately happen outside LOCK. Each
            # committing path below compares revisions before writing, so a slow
            # model cannot block canvas saves or overwrite newer work.
            if action in ("connect", "synthesize", "chat", "extract", "clusters", "gravity"):
                with LOCK:
                    g = load(gid)
                    _prepare_graph(g)
                    _drop_stale_relations(g)
                    revision = _revision(g)

                if action == "connect":
                    g, meta = connect(g)
                    with LOCK:
                        current = load(gid)
                        _prepare_graph(current)
                        if _revision(current) != revision:
                            return self._send({"graph": current, "new": 0, "mode": "stale", "stale": True})
                        return self._send({"graph": commit(gid, g), **meta})

                if action == "synthesize":
                    result = synthesize(g, set(body["ids"]) if body.get("ids") else None)
                    with LOCK:
                        current = load(gid)
                        _prepare_graph(current)
                        result["graph"] = commit(gid, g) if _revision(current) == revision else current
                        result["stale"] = _revision(current) != revision
                        return self._send(result)

                if action == "chat":
                    return self._send({"answer": ask(g, str(body.get("q", "")))})

                if action == "extract":
                    ideas = extract(str(body.get("transcript", "")))
                    with LOCK:
                        # Importing ideas is additive, so it can safely land on
                        # the latest canvas even if someone moved a node meanwhile.
                        current = load(gid)
                        _prepare_graph(current)
                        graph = new_nodes(current, ideas, "team_chat")
                        return self._send(commit(gid, graph))

                if action == "clusters":
                    clusters = refresh_clusters(g)
                    with LOCK:
                        current = load(gid)
                        _prepare_graph(current)
                        if _revision(current) != revision:
                            return self._send({"clusters": current.get("clusters", []), "graph": current,
                                               "embedding_model": _active_embedding_model(), "stale": True})
                        return self._send({"clusters": clusters, "graph": commit(gid, g),
                                           "embedding_model": _active_embedding_model()})

                if action == "gravity":
                    layout = gravity_layout(g, body.get("ids") or [], body.get("iterations", 72))
                    refresh_clusters(g)
                    with LOCK:
                        current = load(gid)
                        _prepare_graph(current)
                        # Layout clusters can describe only a selection. Keep
                        # them separate from the full persisted graph clusters.
                        layout["layout_clusters"] = layout.pop("clusters", [])
                        layout["graph"] = commit(gid, g) if _revision(current) == revision else current
                        layout["stale"] = _revision(current) != revision
                        return self._send(layout)

            with LOCK:
                g = load(gid)
                _prepare_graph(g)
                if action == "edge":                 # FR-14: accept / reject
                    edge_id = body.get("edge_id")
                    if isinstance(edge_id, str) and edge_id:
                        index = next((i for i, item in enumerate(g["edges"])
                                      if item.get("id") == edge_id), -1)
                    else:  # Compatibility with canvases open in an older browser.
                        try:
                            index = int(body.get("index", -1))
                        except (TypeError, ValueError):
                            return self._send({"error": "edge index must be an integer"}, 400)
                    if not 0 <= index < len(g["edges"]):
                        return self._send({"error": "edge not found"}, 404)
                    state = body.get("state")
                    if state not in ("accepted", "rejected"):
                        return self._send({"error": "state must be accepted or rejected"}, 400)
                    edge = g["edges"][index]
                    if state == "rejected":
                        # tuple, not list: _current_rejected_pairs returns a set of
                        # tuples, and `list in set` raises unhashable type: 'list'.
                        pair = tuple(sorted((edge["src"], edge["dst"])))
                        by_id = {node["id"]: node for node in g["nodes"]}
                        source, target = by_id.get(edge["src"]), by_id.get(edge["dst"])
                        if pair not in _current_rejected_pairs(g):
                            g["rejected"].append({
                                "pair": list(pair), "src": edge["src"], "dst": edge["dst"],
                                "source_content_hash": _text_digest(_node_text(source)) if source else None,
                                "target_content_hash": _text_digest(_node_text(target)) if target else None,
                                "timestamp": _now(),
                            })
                        g["edges"].pop(index)
                    else:
                        if body.get("type") in VOCAB:
                            edge["type"] = edge["relationship_type"] = body["type"]
                        if isinstance(body.get("why"), str) and body["why"].strip():
                            edge["why"] = edge["evidence"] = body["why"].strip()[:500]
                        if body.get("conf") is not None:
                            try:
                                edge["conf"] = edge["confidence"] = round(max(0.0, min(1.0, float(body["conf"]))), 2)
                            except (TypeError, ValueError):
                                return self._send({"error": "confidence must be numeric"}, 400)
                        edge["state"] = "accepted"
                        edge["accepted_at"] = _now()
                        by_id = {node["id"]: node for node in g["nodes"]}
                        source, target = by_id.get(edge["src"]), by_id.get(edge["dst"])
                        if source:
                            edge["source_content_hash"] = _text_digest(_node_text(source))
                        if target:
                            edge["target_content_hash"] = _text_digest(_node_text(target))
                        edge.pop("needs_review", None)
                        edge.pop("review_reason", None)
                        edge.pop("review_requested_at", None)
                        for node in g["nodes"]:
                            if node["id"] in (edge["src"], edge["dst"]) and node.get("status") in ("raw", "understood"):
                                node["status"] = "connected"
                    clusters = refresh_clusters(g)
                    return self._send({"graph": commit(gid, g), "clusters": clusters})
            return self._send({"error": "not found"}, 404)
        except Exception as exc:
            print(f"[salaar] {action} failed: {exc!r}")
            return self._send({"error": str(exc)}, 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"SALAAR on http://localhost:{port}  (model {MODEL})")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
