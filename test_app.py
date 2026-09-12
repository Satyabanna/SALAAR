"""Self-check: no network or API key required. `python3 test_app.py`."""
import math
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp()
import app
app.API_KEY = ""
app.REMOTE_EMBEDDINGS_DISABLED = True

N = [{"id": "a", "text": "patient waiting time in hospitals"},
     {"id": "b", "text": "predict hospital bottlenecks early"},
     {"id": "c", "text": "agent should investigate why"},
     {"id": "d", "text": "sourdough starter needs feeding"}]

# Every node receives a reusable, unit semantic representation. Editing content
# invalidates its cached embedding without affecting the other nodes.
CACHE = [{"id": "cache", "text": "reduce hospital patient queues"}]
vector = app.ensure_embeddings(CACHE)["cache"]
semantic = CACHE[0]["semantic_representation"]
assert semantic["model"].startswith("local-semantic") and len(vector) == semantic["dimensions"]
assert abs(math.sqrt(sum(x * x for x in vector)) - 1) < 1e-7
old_hash = semantic["content_hash"]
CACHE[0]["content"] = CACHE[0]["text"] = "reduce hospital waiting queues with triage"
app.ensure_embeddings(CACHE)
assert CACHE[0]["semantic_representation"]["content_hash"] != old_hash

# The centre of gravity is a normalised weighted embedding centroid, and the
# softened force is finite even for identical ideas.
centre = app.centre_of_gravity([[1.0, 0.0], [0.0, 1.0]])
assert all(abs(value - math.sqrt(0.5)) < 1e-9 for value in centre), centre
assert app.gravitational_pull(0.0) == 0
assert 0 < app.gravitational_pull(1.0, 3, 3) <= app.GRAVITY_CAP

# With TOP_K covering every other node, the small canvas naturally includes every pair.
assert len(app.candidates(N)) == 6, app.candidates(N)

# large canvas: bounded at n*k, unrelated node dropped, related pair kept
app.MAX_PAIRS = 0
pairs = app.candidates(N, k=1)
assert len(pairs) <= len(N), pairs
assert any({p["a"]["id"], p["b"]["id"]} == {"a", "b"} for p in pairs), pairs
assert all("d" not in (p["a"]["id"], p["b"]["id"]) for p in pairs), pairs
app.MAX_PAIRS = 60

# Larger canvases are top-k bounded, and relationship typing remains exactly one
# request regardless of how many candidate pairs survive the union.
LARGE = [{"id": f"l{i}", "text": f"hospital patient workflow improvement {i}"} for i in range(7)]
large_pairs = app.candidates(LARGE, k=2)
assert 0 < len(large_pairs) <= len(LARGE) * 2 < len(LARGE) * (len(LARGE) - 1) // 2, large_pairs
called = []
original_type_pairs = app.type_pairs
app.type_pairs = lambda pairs: called.append(len(pairs)) or []
try:
    typed, typing_mode = app.type_pairs_batched(large_pairs)
finally:
    app.type_pairs = original_type_pairs
assert typed == [] and typing_mode == "llm" and called == [len(large_pairs)], called

# Prompted JSON is still untrusted: malformed pair indices cannot crash connect
# or reverse an edge through truthy non-boolean values.
original_type_pairs = app.type_pairs
app.type_pairs = lambda pairs: [{"pair": 0.5, "type": "solves", "why": "bad index", "conf": 1, "flip": "false"}]
try:
    invalid_typed, invalid_mode = app.type_pairs_batched(app.candidates(N)[:1])
finally:
    app.type_pairs = original_type_pairs
assert invalid_mode == "fallback" and all(isinstance(edge["pair"], int) for edge in invalid_typed)

# Semantic COG clusters keep distinct concepts apart. Pending edges are not
# evidence: only accepted positive edges can join graph components or add mass.
M = [
    {"id": "h1", "text": "reduce patient waiting time in hospital wards", "x": 0, "y": 0},
    {"id": "h2", "text": "predict hospital bottlenecks before patients wait", "x": 720, "y": 0},
    {"id": "bread", "text": "feed a sourdough starter every morning", "x": 330, "y": 500},
]
vectors = app.ensure_embeddings(M)
semantic_clusters = app.semantic_clusters(M, [], vectors)
assert any(set(cluster["node_ids"]) == {"h1", "h2"} for cluster in semantic_clusters), semantic_clusters
assert any(cluster["node_ids"] == ["bread"] for cluster in semantic_clusters), semantic_clusters
pending = app.union_find(M, [{"src": "h1", "dst": "h2", "type": "depends_on", "state": "pending"}])
assert all(len(group) == 1 for group in pending), pending
accepted = app.union_find(M, [{"src": "h1", "dst": "h2", "type": "depends_on", "state": "accepted"}])
assert any(set(group) == {"h1", "h2"} for group in accepted), accepted
base_mass = app._node_masses(M, vectors, [])
pending_mass = app._node_masses(M, vectors, [{"src": "h1", "dst": "h2", "type": "depends_on", "state": "pending", "conf": .9}])
accepted_mass = app._node_masses(M, vectors, [{"src": "h1", "dst": "h2", "type": "depends_on", "state": "accepted", "conf": .9}])
assert pending_mass == base_mass and accepted_mass["h1"] > base_mass["h1"]

# Gravity suggests a deterministic layout but never mutates the saved positions;
# the close embedding pair ends physically closer after the simulation.
layout_graph = {"nodes": M, "edges": [], "rejected": []}
before = math.dist((M[0]["x"], M[0]["y"]), (M[1]["x"], M[1]["y"]))
layout = app.gravity_layout(layout_graph)
after_by_id = {point["id"]: point for point in layout["positions"]}
after = math.dist((after_by_id["h1"]["x"], after_by_id["h1"]["y"]),
                  (after_by_id["h2"]["x"], after_by_id["h2"]["y"]))
assert after < before and M[0]["x"] == 0 and M[1]["x"] == 720, (before, after, M)
assert any("spatial_centre" in cluster for cluster in layout["clusters"])

# Rebuilding embedding metadata must retain a cluster's prior synthesis when its
# membership is unchanged, while malformed model-shaped JSON is rejected.
cluster_graph = {"nodes": M, "edges": [], "rejected": []}
app.refresh_clusters(cluster_graph)
app._attach_synthesis(cluster_graph, {"h1", "h2"}, {"core": "Hospital flow intelligence", "gaps": ["Validate triage data."]})
app.refresh_clusters(cluster_graph)
assert any(cluster["core_concept"] == "Hospital flow intelligence" for cluster in cluster_graph["clusters"])
try:
    app._validate_schema({"core": "x", "gaps": "not a list", "contradictions": [],
                          "refined": {"problem": "", "solution": "", "mvp": ""}, "next_action": "x"}, app.SYNTH_SCHEMA)
    raise AssertionError("accepted malformed synthesis")
except ValueError:
    pass

# connect with no API key must degrade, not raise
g, meta = app.connect({"nodes": N, "edges": [], "rejected": []})
assert meta["mode"] == "fallback" and g["edges"], meta
assert all(e["state"] == "pending" for e in g["edges"])          # nothing lands unapproved
assert all(e["conf"] >= 0.35 for e in g["edges"])
assert all(e.get("id", "").startswith("e-") for e in g["edges"])
assert all({"source_node", "target_node", "relationship_type", "evidence", "confidence"} <= set(e) for e in g["edges"])
assert all({"content", "position", "source", "timestamp", "status", "semantic_representation"} <= set(n) for n in g["nodes"])

# The rehearsed demo fallback contains several typed, evidenced suggestions,
# satisfying the core flow even when a live model is absent.
seed = [
    {"id": "s0", "text": "AI for hospitals"},
    {"id": "s1", "text": "Patient waiting time is the real pain"},
    {"id": "s2", "text": "Predict bottlenecks before they form"},
    {"id": "s3", "text": "The agent should investigate why, not just report"},
    {"id": "s4", "text": "Could live in Slack where the team already talks"},
    {"id": "s5", "text": "Nurses will not open another dashboard"},
    {"id": "s6", "text": "Start with one ward, not the whole hospital"},
]
seed_graph, seed_meta = app.connect({"nodes": seed, "edges": [], "rejected": []})
assert seed_meta["mode"] == "fallback" and len(seed_graph["edges"]) >= 3, seed_meta
assert all(edge["why"] and edge["gravitational_pull"] >= 0 for edge in seed_graph["edges"])

# Editing an idea invalidates its old AI proposal and denylist entry, allowing
# the new thought to be reinterpreted instead of displaying stale rationale.
stale_graph, _ = app.connect({"nodes": [
    {"id": "x", "text": "reduce patient waiting in hospitals"},
    {"id": "y", "text": "predict hospital bottlenecks"},
], "edges": [], "rejected": []})
assert stale_graph["edges"], stale_graph
stale_edge = stale_graph["edges"][0]
stale_graph["rejected"] = [{"pair": sorted((stale_edge["src"], stale_edge["dst"])),
    "src": stale_edge["src"], "dst": stale_edge["dst"],
    "source_content_hash": stale_edge["source_content_hash"], "target_content_hash": stale_edge["target_content_hash"]}]
changed = next(node for node in stale_graph["nodes"] if node["id"] == stale_edge["src"])
changed["text"] = changed["content"] = "feed a sourdough starter"
app._prepare_graph(stale_graph)
app._drop_stale_relations(stale_graph)
assert not stale_graph["edges"] and not stale_graph["rejected"], stale_graph

# A wording edit must never silently erase a human-approved edge. It remains
# visible as a review item, but cannot distort components or COG mass until the
# user explicitly reconfirms it.
approved_graph = {"nodes": [
    {"id": "p", "text": "reduce patient waiting in hospitals"},
    {"id": "q", "text": "predict hospital bottlenecks"},
], "edges": [{"src": "p", "dst": "q", "type": "solves", "conf": .9, "state": "accepted"}], "rejected": []}
app._prepare_graph(approved_graph)
approved_id = approved_graph["edges"][0]["id"]
approved_graph["nodes"][0]["text"] = approved_graph["nodes"][0]["content"] = "feed a sourdough starter"
app._drop_incident_relations(approved_graph, {"p"})
app._drop_stale_relations(approved_graph)
assert len(approved_graph["edges"]) == 1 and approved_graph["edges"][0]["needs_review"], approved_graph
assert approved_graph["edges"][0]["id"] == approved_id
approved_vectors = app.ensure_embeddings(approved_graph["nodes"])
assert app._accepted_strengths(approved_graph["nodes"], approved_graph["edges"])["p"] == 0
assert all(len(group) == 1 for group in app.union_find(approved_graph["nodes"], approved_graph["edges"]))

# a rejected pair is never proposed again, and already-typed pairs aren't re-sent
before = len(g["edges"])
g["rejected"] = [sorted((e["src"], e["dst"])) for e in g["edges"]]
g["edges"] = []
g2, meta2 = app.connect(g)
assert meta2["new"] == 0 and not g2["edges"], meta2
assert before > 0

# every LLM entry point must reach its fallback, not a TypeError the broad
# `except Exception` would silently swallow into permanent offline mode
s = app.synthesize({"nodes": N, "edges": []}, None)
assert s["core"] and "refined" in s and s["next_action"], s
assert app.ask({"nodes": N, "edges": []}, "what is this?"), "chat fallback empty"
assert app.extract("we should cut waiting time. nurses hate dashboards anyway."), "extract fallback empty"

# clusters = components of the accepted graph; contradicts must not merge groups
cl = app.union_find(N, [{"src": "a", "dst": "b", "type": "depends_on"},
                        {"src": "c", "dst": "d", "type": "contradicts"}])
groups = sorted(sorted(c) for c in cl)
assert ["a", "b"] in groups and ["c"] in groups and ["d"] in groups, groups

# canvas id is a filename: no traversal
for bad in ("../etc/passwd", "a/b", ""):
    try:
        app._file(bad); raise SystemExit(f"accepted bad id {bad!r}")
    except ValueError:
        pass

# round-trip persistence
app.save("t1", {"nodes": N, "edges": [], "rejected": []})
assert app.load("t1")["nodes"] == N
assert app.load("nope")["nodes"] == []

print("ok")
