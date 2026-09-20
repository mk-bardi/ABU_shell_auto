#!/usr/bin/env python3

"""Tests for the stdlib graph/A* that replaced networkx in the route planner.

These deliberately use nothing but the standard library, so they run in the
same bare container the competition uses to launch the nodes.
"""

import math
import random
from itertools import permutations

import pytest

from shell_simulation.agents.navigation.route_graph import (
    DiGraph,
    NoPathFound,
    astar_path,
)


def _path_cost(graph, path, weight='length'):
    return sum(graph.edges[u, v][weight] for u, v in zip(path, path[1:]))


def _brute_force_cost(graph, nodes, source, target, weight='length'):
    """Cheapest source->target cost by enumerating every simple path."""
    if source == target:
        return 0.0
    best = math.inf
    others = [n for n in nodes if n not in (source, target)]
    for r in range(len(others) + 1):
        for middle in permutations(others, r):
            path = [source, *middle, target]
            if all((u, v) in graph.edges for u, v in zip(path, path[1:])):
                best = min(best, _path_cost(graph, path, weight))
    return best


# -- graph semantics ------------------------------------------------------


def test_add_node_merges_attributes():
    g = DiGraph()
    g.add_node(1, vertex=(0.0, 0.0, 0.0))
    g.add_node(1, extra='kept')
    assert g.nodes[1] == {'vertex': (0.0, 0.0, 0.0), 'extra': 'kept'}


def test_add_edge_merges_attributes_and_creates_endpoints():
    g = DiGraph()
    g.add_edge(1, 2, length=5, type='A')
    g.add_edge(1, 2, type='B')
    assert g.edges[1, 2] == {'length': 5, 'type': 'B'}
    # Endpoints are created implicitly, with empty attribute dicts.
    assert 1 in g and 2 in g
    assert g.nodes[2] == {}


def test_edges_are_directed():
    g = DiGraph()
    g.add_edge(1, 2, length=1)
    assert (1, 2) in g.edges
    assert (2, 1) not in g.edges
    assert list(g.successors(1)) == [2]
    assert list(g.successors(2)) == []


def test_edge_attributes_are_writable_in_place():
    g = DiGraph()
    g.add_edge(1, 2, type='LANEFOLLOW')
    g.edges[1, 2]['type'] = 'CHANGELANELEFT'
    assert g.edges[1, 2]['type'] == 'CHANGELANELEFT'


def test_missing_edge_raises_key_error():
    g = DiGraph()
    g.add_node(1)
    with pytest.raises(KeyError):
        g.edges[1, 2]


# -- A* ------------------------------------------------------------------


def test_finds_the_cheapest_route_not_the_shortest_hop_count():
    g = DiGraph()
    for node in ('a', 'b', 'c', 'd'):
        g.add_node(node)
    g.add_edge('a', 'd', length=100)       # one expensive hop
    g.add_edge('a', 'b', length=1)         # three cheap hops
    g.add_edge('b', 'c', length=1)
    g.add_edge('c', 'd', length=1)
    assert astar_path(g, 'a', 'd', weight='length') == ['a', 'b', 'c', 'd']


def test_trivial_path_to_self():
    g = DiGraph()
    g.add_node(7)
    assert astar_path(g, 7, 7, weight='length') == [7]


def test_raises_when_unreachable():
    g = DiGraph()
    g.add_edge(1, 2, length=1)
    g.add_node(3)
    with pytest.raises(NoPathFound):
        astar_path(g, 1, 3, weight='length')


def test_raises_when_endpoint_missing():
    g = DiGraph()
    g.add_edge(1, 2, length=1)
    with pytest.raises(NoPathFound):
        astar_path(g, 1, 99, weight='length')
    with pytest.raises(NoPathFound):
        astar_path(g, 99, 2, weight='length')


def test_no_path_found_is_catchable_as_runtime_error():
    """planning_node guards trace_route against RuntimeError; keep that working."""
    g = DiGraph()
    g.add_node(1)
    g.add_node(2)
    with pytest.raises(RuntimeError):
        astar_path(g, 1, 2, weight='length')


def test_one_way_edges_are_respected():
    g = DiGraph()
    g.add_edge(1, 2, length=1)
    g.add_edge(2, 3, length=1)
    assert astar_path(g, 1, 3, weight='length') == [1, 2, 3]
    with pytest.raises(NoPathFound):
        astar_path(g, 3, 1, weight='length')


@pytest.mark.parametrize('seed', range(25))
def test_matches_brute_force_optimum_on_random_graphs(seed):
    """With an admissible heuristic, A* must return a genuine optimum."""
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    positions = {i: (rng.uniform(-50, 50), rng.uniform(-50, 50)) for i in range(n)}

    g = DiGraph()
    for i in range(n):
        g.add_node(i, vertex=positions[i])
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < 0.45:
                # Weight is the true euclidean distance, so straight-line
                # distance is an admissible heuristic.
                g.add_edge(u, v, length=math.dist(positions[u], positions[v]))

    def heuristic(a, b):
        return math.dist(positions[a], positions[b])

    for source in range(n):
        for target in range(n):
            expected = _brute_force_cost(g, range(n), source, target)
            try:
                path = astar_path(g, source, target, heuristic, weight='length')
            except NoPathFound:
                assert expected == math.inf
                continue
            assert expected != math.inf
            assert path[0] == source and path[-1] == target
            assert _path_cost(g, path) == pytest.approx(expected)


def test_reaches_the_far_end_of_a_long_chain():
    """Guards against the queue bookkeeping dropping a route on deep graphs."""
    g = DiGraph()
    for i in range(500):
        g.add_edge(i, i + 1, length=1)
    path = astar_path(g, 0, 500, weight='length')
    assert path == list(range(501))
