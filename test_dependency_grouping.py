#!/usr/bin/env python3
"""
Simple unit test for the _group_by_dependency_level() method.
This can run with minimal dependencies to validate the core logic.
"""

import sys
from pathlib import Path

# Test the grouping logic with a mock node class
class MockNode:
    """Mock node for testing dependency grouping."""

    def __init__(self, name, parents=None):
        self.name = name
        self.parents = parents or []

    def _group_by_dependency_level(self, nodes):
        """
        Group nodes by dependency level (copied from implementation).
        """
        node_set = set(nodes)
        node_levels = {}

        def get_level(node):
            if node in node_levels:
                return node_levels[node]

            if not node.parents:
                node_levels[node] = 0
                return 0

            relevant_parents = [p for p in node.parents if p in node_set]

            if not relevant_parents:
                node_levels[node] = 0
                return 0

            parent_levels = [get_level(p) for p in relevant_parents]
            level = 1 + max(parent_levels)
            node_levels[node] = level
            return level

        for node in nodes:
            get_level(node)

        if not node_levels:
            return []

        max_level = max(node_levels.values())
        levels = [[] for _ in range(max_level + 1)]

        for node, level in node_levels.items():
            levels[level].append(node)

        return levels

    def __repr__(self):
        return f"Node({self.name})"


def test_diamond_graph():
    """Test dependency grouping for diamond graph: A -> B, C -> D"""
    print("Test 1: Diamond Graph")
    print("  Graph structure: A -> B, A -> C, B -> D, C -> D")

    # Create diamond graph
    node_a = MockNode("A")
    node_b = MockNode("B", [node_a])
    node_c = MockNode("C", [node_a])
    node_d = MockNode("D", [node_b, node_c])

    # Group nodes
    nodes = [node_a, node_b, node_c, node_d]
    levels = node_d._group_by_dependency_level(nodes)

    # Verify levels
    print(f"  Level 0 (roots): {[n.name for n in levels[0]]}")
    print(f"  Level 1 (B, C):  {[n.name for n in levels[1]]}")
    print(f"  Level 2 (D):     {[n.name for n in levels[2]]}")

    assert len(levels) == 3, f"Expected 3 levels, got {len(levels)}"
    assert len(levels[0]) == 1 and levels[0][0].name == "A", "Level 0 should be [A]"
    assert len(levels[1]) == 2, "Level 1 should have 2 nodes (B, C)"
    assert set(n.name for n in levels[1]) == {"B", "C"}, "Level 1 should be B and C"
    assert len(levels[2]) == 1 and levels[2][0].name == "D", "Level 2 should be [D]"

    print("  ✓ PASSED: B and C are at same level (can run in parallel)\n")


def test_linear_graph():
    """Test dependency grouping for linear graph: A -> B -> C -> D"""
    print("Test 2: Linear Graph")
    print("  Graph structure: A -> B -> C -> D")

    # Create linear graph
    node_a = MockNode("A")
    node_b = MockNode("B", [node_a])
    node_c = MockNode("C", [node_b])
    node_d = MockNode("D", [node_c])

    # Group nodes
    nodes = [node_a, node_b, node_c, node_d]
    levels = node_d._group_by_dependency_level(nodes)

    # Verify levels
    for i, level in enumerate(levels):
        print(f"  Level {i}: {[n.name for n in level]}")

    assert len(levels) == 4, f"Expected 4 levels, got {len(levels)}"
    assert all(len(level) == 1 for level in levels), "Each level should have 1 node"

    print("  ✓ PASSED: All nodes at different levels (sequential execution)\n")


def test_wide_graph():
    """Test dependency grouping for wide graph: A -> B, C, D, E -> F"""
    print("Test 3: Wide Graph")
    print("  Graph structure: A -> B, C, D, E -> F")

    # Create wide graph
    node_a = MockNode("A")
    node_b = MockNode("B", [node_a])
    node_c = MockNode("C", [node_a])
    node_d = MockNode("D", [node_a])
    node_e = MockNode("E", [node_a])
    node_f = MockNode("F", [node_b, node_c, node_d, node_e])

    # Group nodes
    nodes = [node_a, node_b, node_c, node_d, node_e, node_f]
    levels = node_f._group_by_dependency_level(nodes)

    # Verify levels
    print(f"  Level 0 (A):           {[n.name for n in levels[0]]}")
    print(f"  Level 1 (B, C, D, E):  {[n.name for n in levels[1]]}")
    print(f"  Level 2 (F):           {[n.name for n in levels[2]]}")

    assert len(levels) == 3, f"Expected 3 levels, got {len(levels)}"
    assert len(levels[0]) == 1 and levels[0][0].name == "A", "Level 0 should be [A]"
    assert len(levels[1]) == 4, "Level 1 should have 4 nodes (B, C, D, E)"
    assert set(n.name for n in levels[1]) == {"B", "C", "D", "E"}, "Level 1 should be B, C, D, E"
    assert len(levels[2]) == 1 and levels[2][0].name == "F", "Level 2 should be [F]"

    print("  ✓ PASSED: B, C, D, E are at same level (can run in parallel)\n")


def test_complex_graph():
    """Test dependency grouping for complex multi-level graph"""
    print("Test 4: Complex Multi-Level Graph")
    print("  Graph structure: A -> B, C -> D, E, F -> G")

    # Create complex graph
    node_a = MockNode("A")
    node_b = MockNode("B", [node_a])
    node_c = MockNode("C", [node_a])
    node_d = MockNode("D", [node_b])
    node_e = MockNode("E", [node_b, node_c])
    node_f = MockNode("F", [node_c])
    node_g = MockNode("G", [node_d, node_e, node_f])

    # Group nodes
    nodes = [node_a, node_b, node_c, node_d, node_e, node_f, node_g]
    levels = node_g._group_by_dependency_level(nodes)

    # Verify levels
    for i, level in enumerate(levels):
        print(f"  Level {i}: {[n.name for n in level]}")

    assert len(levels) == 4, f"Expected 4 levels, got {len(levels)}"
    assert levels[0][0].name == "A", "Level 0 should be A"
    assert set(n.name for n in levels[1]) == {"B", "C"}, "Level 1 should be B, C"
    assert set(n.name for n in levels[2]) == {"D", "E", "F"}, "Level 2 should be D, E, F"
    assert levels[3][0].name == "G", "Level 3 should be G"

    print("  ✓ PASSED: Multi-level parallelism correctly identified\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing _group_by_dependency_level() Logic")
    print("=" * 70)
    print()

    try:
        test_diamond_graph()
        test_linear_graph()
        test_wide_graph()
        test_complex_graph()

        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The dependency grouping logic correctly identifies which nodes")
        print("can execute in parallel at each level of the graph.")
        print()
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
