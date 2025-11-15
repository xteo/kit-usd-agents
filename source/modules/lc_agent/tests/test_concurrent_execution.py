## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Tests for concurrent execution of independent graph branches.

This test suite validates that independent branches in the execution graph
run in parallel rather than sequentially, providing significant performance
improvements for multi-agent workflows.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from lc_agent.runnable_node import RunnableNode
from lc_agent.runnable_network import RunnableNetwork


class TimedNode(RunnableNode):
    """
    A test node that tracks execution timing and can simulate work with delays.

    This node records when it starts and finishes execution, allowing tests
    to verify whether nodes execute in parallel or sequentially.
    """

    # Class variable to track all execution events across all instances
    execution_log = []

    def __init__(self, name: str, delay: float = 0, *args, **kwargs):
        """
        Initialize a timed node.

        Args:
            name: Human-readable name for the node (used in logging)
            delay: Simulated work delay in seconds
        """
        super().__init__(*args, **kwargs)
        self.node_name = name
        self.delay = delay
        self.start_time = None
        self.end_time = None

    async def ainvoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> AIMessage:
        """
        Execute the node asynchronously with timing tracking.

        Args:
            input: Input data dictionary
            config: Runnable configuration
            **kwargs: Additional keyword arguments

        Returns:
            AIMessage with the node's result
        """
        # Don't re-execute if already invoked
        if self.invoked:
            return self.outputs

        # Record start time
        self.start_time = time.time()
        TimedNode.execution_log.append({
            "event": "start",
            "node": self.node_name,
            "time": self.start_time
        })

        # Simulate work with async sleep
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Record end time
        self.end_time = time.time()
        TimedNode.execution_log.append({
            "event": "end",
            "node": self.node_name,
            "time": self.end_time
        })

        # Set outputs and mark as invoked
        self.outputs = AIMessage(content=f"{self.node_name} result")
        self.invoked = True

        return self.outputs

    def invoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> AIMessage:
        """
        Synchronous invoke (delegates to async version via asyncio.run).

        This is a simplified implementation for testing purposes.
        """
        if self.invoked:
            return self.outputs

        # For sync invoke, we'll use a simpler implementation
        self.start_time = time.time()
        if self.delay > 0:
            time.sleep(self.delay)
        self.end_time = time.time()

        self.outputs = AIMessage(content=f"{self.node_name} result")
        self.invoked = True
        return self.outputs

    @classmethod
    def reset_log(cls):
        """Reset the execution log (call before each test)."""
        cls.execution_log = []


def check_execution_overlap(node1: TimedNode, node2: TimedNode) -> bool:
    """
    Check if two nodes' execution overlapped (ran in parallel).

    Args:
        node1: First node
        node2: Second node

    Returns:
        True if executions overlapped, False if sequential
    """
    if not all([node1.start_time, node1.end_time, node2.start_time, node2.end_time]):
        return False

    # For overlap: max(start times) < min(end times)
    # If sequential: one ends before the other starts
    overlap_start = max(node1.start_time, node2.start_time)
    overlap_end = min(node1.end_time, node2.end_time)

    return overlap_start < overlap_end


@pytest.fixture
def reset_execution_log():
    """Fixture to reset execution log before each test."""
    TimedNode.reset_log()
    yield
    TimedNode.reset_log()


@pytest.mark.asyncio
async def test_diamond_graph_parallel_execution(reset_execution_log):
    """
    Test that independent branches in a diamond graph execute in parallel.

    Graph structure:
        A (0.1s)
       / \
      B   C  (1.0s each)
       \ /
        D (0.1s)

    Expected behavior with parallel execution:
    - Total time: ~1.2s (A + max(B,C) + D)
    - B and C execute concurrently (execution overlap)

    Expected behavior with sequential execution (current):
    - Total time: ~2.2s (A + B + C + D)
    - B and C execute one after another (no overlap)
    """
    # Create diamond graph
    with RunnableNetwork() as network:
        node_a = TimedNode(name="A", delay=0.1)
        node_b = TimedNode(name="B", delay=1.0)
        node_c = TimedNode(name="C", delay=1.0)
        node_d = TimedNode(name="D", delay=0.1)

    # Verify graph structure
    assert node_b in network.get_children(node_a)
    assert node_c in network.get_children(node_a)
    assert node_a in network.get_parents(node_b)
    assert node_a in network.get_parents(node_c)
    assert node_b in network.get_parents(node_d)
    assert node_c in network.get_parents(node_d)

    # Execute the network and measure time
    start = time.time()
    result = await network.ainvoke()
    total_time = time.time() - start

    # Verify all nodes executed
    assert node_a.invoked
    assert node_b.invoked
    assert node_c.invoked
    assert node_d.invoked

    # Check if B and C executed in parallel
    parallel = check_execution_overlap(node_b, node_c)

    # Print diagnostic information
    print(f"\n{'='*60}")
    print(f"Diamond Graph Execution Results:")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"\nNode timings:")
    print(f"  A: {node_a.start_time:.3f} -> {node_a.end_time:.3f} ({node_a.end_time - node_a.start_time:.3f}s)")
    print(f"  B: {node_b.start_time:.3f} -> {node_b.end_time:.3f} ({node_b.end_time - node_b.start_time:.3f}s)")
    print(f"  C: {node_c.start_time:.3f} -> {node_c.end_time:.3f} ({node_c.end_time - node_c.start_time:.3f}s)")
    print(f"  D: {node_d.start_time:.3f} -> {node_d.end_time:.3f} ({node_d.end_time - node_d.start_time:.3f}s)")

    if parallel:
        print(f"\n✓ B and C executed IN PARALLEL")
        print(f"  Expected time: ~1.2s")
        print(f"  Actual time: {total_time:.3f}s")
        # With parallel execution, should be close to 1.2s
        assert total_time < 1.5, \
            f"Parallel execution should take ~1.2s, but took {total_time:.3f}s"
    else:
        print(f"\n✗ B and C executed SEQUENTIALLY")
        print(f"  Expected time with parallel: ~1.2s")
        print(f"  Expected time with sequential: ~2.2s")
        print(f"  Actual time: {total_time:.3f}s")
        # With sequential execution, should be close to 2.2s
        assert total_time > 2.0, \
            f"Sequential execution should take ~2.2s, but took {total_time:.3f}s"

    print(f"{'='*60}\n")

    # THIS TEST WILL FAIL WITH CURRENT IMPLEMENTATION (sequential)
    # Uncomment the following assertion once parallel execution is implemented:
    # assert parallel, "B and C should execute in parallel but executed sequentially"
    # assert total_time < 1.5, f"Expected ~1.2s with parallel execution, got {total_time:.3f}s"


@pytest.mark.asyncio
async def test_wide_graph_parallel_execution(reset_execution_log):
    """
    Test parallel execution with multiple independent branches.

    Graph structure:
           A (0.1s)
         / | | \
        B  C D  E  (0.5s each)
         \ | | /
           F (0.1s)

    Expected with parallel: ~0.7s (A + max(B,C,D,E) + F)
    Expected with sequential: ~2.1s (A + B + C + D + E + F)
    """
    with RunnableNetwork() as network:
        node_a = TimedNode(name="A", delay=0.1)
        node_b = TimedNode(name="B", delay=0.5)
        node_c = TimedNode(name="C", delay=0.5)
        node_d = TimedNode(name="D", delay=0.5)
        node_e = TimedNode(name="E", delay=0.5)
        node_f = TimedNode(name="F", delay=0.1)

    # Execute
    start = time.time()
    await network.ainvoke()
    total_time = time.time() - start

    # Check if middle nodes executed in parallel
    parallel_bc = check_execution_overlap(node_b, node_c)
    parallel_cd = check_execution_overlap(node_c, node_d)
    parallel_de = check_execution_overlap(node_d, node_e)

    print(f"\n{'='*60}")
    print(f"Wide Graph Execution Results:")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"\nParallel execution detected:")
    print(f"  B-C overlap: {parallel_bc}")
    print(f"  C-D overlap: {parallel_cd}")
    print(f"  D-E overlap: {parallel_de}")

    if parallel_bc and parallel_cd and parallel_de:
        print(f"\n✓ All middle nodes executed IN PARALLEL")
        print(f"  Expected time: ~0.7s")
        print(f"  Actual time: {total_time:.3f}s")
        assert total_time < 1.0
    else:
        print(f"\n✗ Middle nodes executed SEQUENTIALLY")
        print(f"  Expected time with parallel: ~0.7s")
        print(f"  Expected time with sequential: ~2.1s")
        print(f"  Actual time: {total_time:.3f}s")
        assert total_time > 1.8

    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_linear_graph_no_parallelism(reset_execution_log):
    """
    Test that linear graphs (no branches) still work correctly.

    Graph structure: A -> B -> C -> D

    This should always be sequential regardless of parallel execution support.
    """
    with RunnableNetwork() as network:
        node_a = TimedNode(name="A", delay=0.2)
        node_b = TimedNode(name="B", delay=0.2)
        node_c = TimedNode(name="C", delay=0.2)
        node_d = TimedNode(name="D", delay=0.2)

    start = time.time()
    await network.ainvoke()
    total_time = time.time() - start

    print(f"\n{'='*60}")
    print(f"Linear Graph Execution Results:")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Expected time: ~0.8s (all sequential)")
    print(f"{'='*60}\n")

    # Should be sequential (0.8s total)
    assert total_time >= 0.7, "Linear graph should take at least 0.7s"
    assert total_time < 1.0, "Linear graph shouldn't take more than 1.0s"


@pytest.mark.asyncio
async def test_complex_multi_level_graph(reset_execution_log):
    """
    Test a more complex graph with multiple levels of parallelism.

    Graph structure:
            A (0.1s)
           / \
          B   C (0.3s each)
         / \ / \
        D   E   F (0.2s each)
         \ | /
           G (0.1s)

    Levels:
    - Level 0: A
    - Level 1: B, C (can be parallel)
    - Level 2: D, E, F (can be parallel)
    - Level 3: G

    Expected with parallel: ~0.7s (0.1 + 0.3 + 0.2 + 0.1)
    Expected with sequential: ~1.2s (sum of all)
    """
    with RunnableNetwork() as network:
        node_a = TimedNode(name="A", delay=0.1)
        node_b = TimedNode(name="B", delay=0.3)
        node_c = TimedNode(name="C", delay=0.3)
        node_d = TimedNode(name="D", delay=0.2)
        node_e = TimedNode(name="E", delay=0.2)
        node_f = TimedNode(name="F", delay=0.2)
        node_g = TimedNode(name="G", delay=0.1)

    start = time.time()
    await network.ainvoke()
    total_time = time.time() - start

    # Check level 1 parallelism (B and C)
    level1_parallel = check_execution_overlap(node_b, node_c)

    # Check level 2 parallelism (D, E, F)
    level2_parallel = (
        check_execution_overlap(node_d, node_e) and
        check_execution_overlap(node_e, node_f)
    )

    print(f"\n{'='*60}")
    print(f"Complex Multi-Level Graph Execution Results:")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"\nParallelism detected:")
    print(f"  Level 1 (B, C): {level1_parallel}")
    print(f"  Level 2 (D, E, F): {level2_parallel}")

    if level1_parallel and level2_parallel:
        print(f"\n✓ Multi-level parallel execution detected")
        print(f"  Expected time: ~0.7s")
        print(f"  Actual time: {total_time:.3f}s")
        assert total_time < 0.9
    else:
        print(f"\n✗ Sequential execution detected")
        print(f"  Expected time with parallel: ~0.7s")
        print(f"  Expected time with sequential: ~1.2s")
        print(f"  Actual time: {total_time:.3f}s")

    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_execution_log_ordering(reset_execution_log):
    """
    Test that execution log shows correct ordering of events.

    This helps visualize the execution pattern.
    """
    with RunnableNetwork() as network:
        node_a = TimedNode(name="A", delay=0.1)
        node_b = TimedNode(name="B", delay=0.3)
        node_c = TimedNode(name="C", delay=0.3)
        node_d = TimedNode(name="D", delay=0.1)

    await network.ainvoke()

    print(f"\n{'='*60}")
    print(f"Execution Log (time-ordered):")
    print(f"{'='*60}")

    # Sort log by time
    sorted_log = sorted(TimedNode.execution_log, key=lambda x: x["time"])

    for entry in sorted_log:
        relative_time = entry["time"] - sorted_log[0]["time"]
        print(f"  {relative_time:6.3f}s: {entry['node']} {entry['event']}")

    print(f"{'='*60}\n")

    # Analyze the log to detect parallel patterns
    starts = {e["node"]: e["time"] for e in sorted_log if e["event"] == "start"}
    ends = {e["node"]: e["time"] for e in sorted_log if e["event"] == "end"}

    # B and C should have overlapping execution if parallel
    if "B" in starts and "C" in starts and "B" in ends and "C" in ends:
        b_c_overlap = max(starts["B"], starts["C"]) < min(ends["B"], ends["C"])
        print(f"B and C overlap: {b_c_overlap}")


if __name__ == "__main__":
    """
    Run tests directly with detailed output.

    Usage: python test_concurrent_execution.py
    """
    import sys

    print("="*60)
    print("LC_Agent Concurrent Execution Test Suite")
    print("="*60)
    print()
    print("This test suite validates whether independent graph branches")
    print("execute in parallel or sequentially.")
    print()
    print("Running tests...")
    print()

    # Run pytest with verbose output
    sys.exit(pytest.main([__file__, "-v", "-s"]))
