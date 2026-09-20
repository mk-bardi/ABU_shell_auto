#!/usr/bin/env python3

"""
Minimal directed-graph and A* implementation used by GlobalRoutePlanner.

This replaces the two `networkx` calls the planner used to make
(`nx.DiGraph` and `nx.astar_path`). networkx is not part of the ROS base
image and is not installed by the competition's run job, so importing it
killed `planning_node` at start-up. Everything here is standard library
only, so the node can no longer fail to start for a missing dependency.

`DiGraph` implements exactly the surface the planner uses -- `add_node`,
`add_edge`, `nodes[n]`, `edges[u, v]` and `successors(n)` -- with the same
semantics as networkx: adding an existing node or edge merges attributes
rather than replacing them, and `add_edge` creates missing endpoints.

`astar_path` follows networkx's algorithm step for step, including its
handling of re-expanded nodes, so routes are identical to the ones the
planner produced before.
"""

from heapq import heappush, heappop
from itertools import count


class NoPathFound(RuntimeError):
    """Raised when no route connects source to target.

    Derives from RuntimeError so that callers already guarding CARLA calls
    against RuntimeError keep working unchanged.
    """


class DiGraph:
    """A directed graph with attribute dictionaries on nodes and edges."""

    def __init__(self):
        self._node = {}       # node -> attribute dict
        self._succ = {}       # node -> {successor -> edge attribute dict}
        self._pred_count = {}  # node -> number of incoming edges (unused, kept cheap)

    # -- construction ----------------------------------------------------

    def add_node(self, node, **attrs):
        """Add `node`, merging `attrs` into any attributes it already has."""
        if node not in self._node:
            self._node[node] = {}
            self._succ[node] = {}
            self._pred_count[node] = 0
        self._node[node].update(attrs)

    def add_edge(self, u, v, **attrs):
        """Add edge u -> v, creating endpoints and merging attributes."""
        self.add_node(u)
        self.add_node(v)
        if v not in self._succ[u]:
            self._succ[u][v] = {}
            self._pred_count[v] += 1
        self._succ[u][v].update(attrs)

    # -- access ----------------------------------------------------------

    @property
    def nodes(self):
        """Mapping of node -> attribute dict, e.g. ``graph.nodes[n]['vertex']``."""
        return self._node

    @property
    def edges(self):
        """Mapping of (u, v) -> attribute dict, e.g. ``graph.edges[u, v]['length']``."""
        return _EdgeView(self._succ)

    def successors(self, node):
        """Iterate over the nodes reachable from `node` by one edge."""
        return iter(self._succ[node])

    def __contains__(self, node):
        return node in self._node

    def __len__(self):
        return len(self._node)


class _EdgeView:
    """Provides the ``graph.edges[u, v]`` lookup used throughout the planner."""

    __slots__ = ('_succ',)

    def __init__(self, succ):
        self._succ = succ

    def __getitem__(self, key):
        u, v = key
        try:
            return self._succ[u][v]
        except KeyError:
            raise KeyError(f"No edge between {u!r} and {v!r}") from None

    def __contains__(self, key):
        u, v = key
        return u in self._succ and v in self._succ[u]


def astar_path(graph, source, target, heuristic=None, weight='weight'):
    """Return the lowest-cost path from `source` to `target` as a list of nodes.

    `heuristic(a, b)` estimates the remaining cost from `a` to `b`; when it is
    omitted the search degenerates to Dijkstra. `weight` names the edge
    attribute holding the cost of traversing that edge.

    Raises NoPathFound if either endpoint is missing or no route exists.
    """
    if source not in graph:
        raise NoPathFound(f"Source node {source!r} is not in the graph")
    if target not in graph:
        raise NoPathFound(f"Target node {target!r} is not in the graph")

    if heuristic is None:
        def heuristic(_a, _b):
            return 0

    tiebreaker = count()
    # Queue entries are (estimated total cost, tiebreaker, node, cost so far, parent).
    queue = [(0, next(tiebreaker), source, 0, None)]
    enqueued = {}   # node -> (best known cost to node, heuristic from node)
    explored = {}   # node -> parent on the best path found so far

    while queue:
        _, __, current, dist, parent = heappop(queue)

        if current == target:
            path = [current]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if current in explored:
            # A cheaper route to `current` was already finalised.
            if explored[current] is None:
                continue
            queued_cost, _h = enqueued[current]
            if queued_cost < dist:
                continue

        explored[current] = parent

        for neighbour in graph.successors(current):
            cost = graph.edges[current, neighbour].get(weight, 1)
            if cost is None:
                continue
            neighbour_cost = dist + cost
            if neighbour in enqueued:
                queued_cost, h = enqueued[neighbour]
                # Only continue if we found a strictly cheaper route.
                if queued_cost <= neighbour_cost:
                    continue
            else:
                h = heuristic(neighbour, target)
            enqueued[neighbour] = neighbour_cost, h
            heappush(
                queue,
                (neighbour_cost + h, next(tiebreaker), neighbour, neighbour_cost, current),
            )

    raise NoPathFound(f"No path between {source!r} and {target!r}")
