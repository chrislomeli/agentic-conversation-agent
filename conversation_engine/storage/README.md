# Graph Storage Layer

World-class in-memory graph storage with proper graph semantics.

## Design Principles

This implementation follows production-grade graph database patterns:

1. **Nodes and edges are separate entities** - not embedded references
2. **Edges are first-class citizens** - queryable, indexable, independent
3. **Indexed operations** - O(1) or O(k) lookups where k = result size
4. **Type safety** - leverages Pydantic models throughout
5. **Transactional semantics** - operations succeed or fail atomically
6. **Graph paradigm** - same conceptual model as Neo4j/MemGraph

## Core Components

### `KnowledgeGraph` - Core Storage

The main graph data structure with proper graph semantics.

**Key operations:**

```python
from conversation_engine.storage import KnowledgeGraph
from conversation_engine.models import Goal, Requirement
from conversation_engine.models.base import BaseEdge

graph = KnowledgeGraph()

# Add nodes
goal = Goal(id="goal-1", name="My Goal", statement="Do something")
req = Requirement(id="req-1", name="My Requirement")

graph.add_node(goal)
graph.add_node(req)

# Add edges between existing nodes
edge = BaseEdge(
    edge_type="SATISFIED_BY",
    source_id="goal-1",
    target_id="req-1"
)
graph.add_edge(edge)

# Query edges
outgoing = graph.get_outgoing_edges("goal-1")
incoming = graph.get_incoming_edges("req-1")

# Graph metrics
out_degree = graph.get_out_degree("goal-1")
in_degree = graph.get_in_degree("req-1")
```

**Important characteristics:**

- **Edges are independent of nodes** - you can add/remove edges without touching nodes
- **Nodes can be updated** - replacing a node preserves all its edges
- **Multiple edge types** - same nodes can have multiple edge types between them
- **Safe deletion** - `remove_node()` fails if edges exist; use `remove_node_cascade()` to remove edges automatically

### `GraphQueries` - High-Level Operations

Graph analysis patterns built on top of core storage.

**Key operations:**

```python
from conversation_engine.storage import GraphQueries

queries = GraphQueries(graph)

# Find neighbors
requirements = queries.get_neighbors_out("goal-1", edge_type="SATISFIED_BY")
goals = queries.get_neighbors_in("req-1", edge_type="SATISFIED_BY")

# Find orphans (nodes with no edges)
orphan_goals = queries.find_orphans("goal", direction="out")

# Find gaps (nodes missing specific edge types)
goals_without_reqs = queries.find_nodes_missing_edge_type(
    "goal", 
    "SATISFIED_BY", 
    direction="out"
)

# Traverse paths (follow edge chains)
paths = queries.traverse_path(
    "goal-1",
    ["SATISFIED_BY", "REALIZED_BY", "REALIZED_BY"]
)
# Returns: [[goal, requirement, capability, component], ...]

# Find all reachable nodes
reachable = queries.find_reachable_nodes("goal-1", max_depth=10)

# Coverage analysis
ratio = queries.get_coverage_ratio("goal", "SATISFIED_BY", direction="out")
# Returns: 0.0 to 1.0 (fraction of goals with requirements)
```

## Why This Design?

### Proper Graph Semantics

**Wrong approach** (dictionary with embedded references):
```python
# Anti-pattern - don't do this
goal = {
    "id": "goal-1",
    "requirements": ["req-1", "req-2"]  # Embedded in node
}
```

**Right approach** (separate nodes and edges):
```python
# Proper graph - edges are separate entities
graph.add_node(goal)
graph.add_node(req1)
graph.add_node(req2)

graph.add_edge(BaseEdge("SATISFIED_BY", "goal-1", "req-1"))
graph.add_edge(BaseEdge("SATISFIED_BY", "goal-1", "req-2"))
```

**Why this matters:**

1. **Add edges to existing nodes** - no need to modify node content
2. **Query edges independently** - "show me all SATISFIED_BY edges"
3. **Multiple edge types** - same nodes can have different relationship types
4. **Bidirectional queries** - find incoming or outgoing edges efficiently
5. **Graph algorithms** - path traversal, reachability, orphan detection
6. **Database compatibility** - same model as Neo4j, MemGraph, etc.

### Indexed for Performance

Even though this is in-memory, we maintain proper indexes:

- **Outgoing edge index** - `source_id → [edges]`
- **Incoming edge index** - `target_id → [edges]`
- **Edge type index** - `edge_type → [edges]`
- **Node type index** - `node_type → {node_ids}`

This enables O(1) or O(k) queries instead of O(n) scans.

### Transactional Semantics

Operations either succeed completely or fail with clear errors:

```python
# This fails atomically - no partial state
try:
    graph.add_edge(edge_to_nonexistent_node)
except ValueError as e:
    # Graph state unchanged
    print(f"Failed: {e}")

# This succeeds atomically - both nodes removed
graph.remove_node_cascade("goal-1")
```

## Graph Paradigm Examples

### Example 1: Adding Edges to Existing Nodes

```python
# Create nodes
goal = Goal(id="goal-1", name="Goal", statement="Test")
req1 = Requirement(id="req-1", name="Req 1")
req2 = Requirement(id="req-2", name="Req 2")

graph.add_node(goal)
graph.add_node(req1)
graph.add_node(req2)

# Add first edge
graph.add_edge(BaseEdge("SATISFIED_BY", "goal-1", "req-1"))

# Later, add another edge to the same goal (no need to touch the goal node)
graph.add_edge(BaseEdge("SATISFIED_BY", "goal-1", "req-2"))

# Query all requirements for this goal
reqs = queries.get_neighbors_out("goal-1", edge_type="SATISFIED_BY")
# Returns: [req1, req2]
```

### Example 2: Multiple Edge Types

```python
comp1 = Component(id="comp-1", name="Component 1")
comp2 = Component(id="comp-2", name="Component 2")

graph.add_node(comp1)
graph.add_node(comp2)

# Same nodes, different relationships
graph.add_edge(BaseEdge("DEPENDS_ON", "comp-1", "comp-2"))
graph.add_edge(BaseEdge("INFORMS", "comp-1", "comp-2"))

# Query by edge type
dependencies = graph.get_outgoing_edges("comp-1", "DEPENDS_ON")
information_flows = graph.get_outgoing_edges("comp-1", "INFORMS")
```

### Example 3: Path Traversal

```python
# Build a chain: Goal → Requirement → Capability → Component
graph.add_edge(BaseEdge("SATISFIED_BY", "goal-1", "req-1"))
graph.add_edge(BaseEdge("REALIZED_BY", "req-1", "cap-1"))
graph.add_edge(BaseEdge("REALIZED_BY", "cap-1", "comp-1"))

# Traverse the full chain
paths = queries.traverse_path(
    "goal-1",
    ["SATISFIED_BY", "REALIZED_BY", "REALIZED_BY"]
)

# paths[0] = [goal, requirement, capability, component]
```

## Testing

Comprehensive test suite with 35 tests covering:

- Node CRUD operations
- Edge CRUD operations
- Graph metrics (degree, counts)
- Graph semantics (independence, multiple edge types)
- Neighbor queries
- Orphan detection
- Path traversal
- Coverage analysis

Run tests:
```bash
/opt/miniforge3/envs/langgraph-sandbox/bin/python -m pytest tests/conversation_engine/storage/ -v
```

## Migration Path to Real Graph Database

This implementation uses the same conceptual model as production graph databases.

**To migrate to Neo4j:**

```python
# Current (in-memory)
graph.add_node(goal)
graph.add_edge(edge)

# Neo4j (same pattern)
session.run("CREATE (g:Goal {id: $id, name: $name})", id=goal.session_id, name=goal.name)
session.run("MATCH (g:Goal {id: $src}), (r:Requirement {id: $tgt}) CREATE (g)-[:SATISFIED_BY]->(r)", ...)
```

**To migrate to MemGraph:**
```python
# Current (in-memory)
queries.get_neighbors_out("goal-1", "SATISFIED_BY")

# MemGraph (same pattern)
session.run("MATCH (g:Goal {id: 'goal-1'})-[:SATISFIED_BY]->(r) RETURN r")
```

The abstraction layer makes this swap straightforward when needed.

## Next Steps

This storage layer provides the foundation for:

1. **Integrity rule evaluation** - validate graph structure against rules
2. **AI gap detection** - find missing relationships
3. **AI reasoning** - explain gaps and suggest fixes
4. **Conversation orchestration** - build artifacts through dialogue

The graph is ready. Now we build the AI components that reason over it.
