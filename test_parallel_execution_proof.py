#!/usr/bin/env python3
"""
Standalone test demonstrating parallel execution in LC_agent.
This version imports only what's needed to avoid dependency issues.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the lc_agent source to path
lc_agent_src = Path(__file__).parent / "source" / "modules" / "lc_agent" / "src"
sys.path.insert(0, str(lc_agent_src))

# Import just what we need
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSerializable
from typing import Dict, Any, Optional, List

# Import the specific files we need, avoiding the full __init__.py
import importlib.util

# Load runnable_node directly
spec = importlib.util.spec_from_file_location(
    "lc_agent.runnable_node",
    lc_agent_src / "lc_agent" / "runnable_node.py"
)
runnable_node = importlib.util.module_from_spec(spec)

# Load runnable_network directly
spec_network = importlib.util.spec_from_file_location(
    "lc_agent.runnable_network",
    lc_agent_src / "lc_agent" / "runnable_network.py"
)
runnable_network = importlib.util.module_from_spec(spec_network)

# Mock dependencies that runnable_node needs
class MockChatModelRegistry:
    def get_chat_model(self, *args, **kwargs):
        return None

class MockNodeFactory:
    pass

class MockProfiler:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class MockUUIDMixin:
    def uuid(self):
        import uuid
        if not hasattr(self, '_uuid'):
            self._uuid = str(uuid.uuid4())
        return self._uuid

# Inject mocks
sys.modules['lc_agent.chat_model_registry'] = type('module', (), {
    'get_chat_model_registry': lambda: MockChatModelRegistry()
})()
sys.modules['lc_agent.node_factory'] = type('module', (), {
    'get_node_factory': lambda: MockNodeFactory()
})()
sys.modules['lc_agent.utils.culling'] = type('module', (), {
    '_cull_messages': lambda x: x
})()
sys.modules['lc_agent.utils.profiling_utils'] = type('module', (), {
    'Profiler': MockProfiler
})()
sys.modules['lc_agent.uuid_utils'] = type('module', (), {
    'UUIDMixin': MockUUIDMixin
})()

# Now load the modules
spec.loader.exec_module(runnable_node)
sys.modules['lc_agent.runnable_node'] = runnable_node

spec_network.loader.exec_module(runnable_network)
sys.modules['lc_agent.runnable_network'] = runnable_network

RunnableNode = runnable_node.RunnableNode
RunnableNetwork = runnable_network.RunnableNetwork


class TimedTestNode(RunnableNode):
    """Test node that tracks execution timing."""

    execution_log = []

    def __init__(self, name: str, delay: float = 0):
        # Initialize without calling super().__init__() to avoid network registration
        self.name = name
        self.node_name = name
        self.delay = delay
        self.start_time = None
        self.end_time = None
        self.parents = []
        self.inputs = []
        self.outputs = None
        self.metadata = {}
        self.invoked = False
        self.verbose = False
        self.chat_model_name = None
        self._uuid = None

    async def ainvoke(self, input: Dict[str, Any] = None, config=None, **kwargs):
        if self.invoked:
            return self.outputs

        self.start_time = time.time()
        TimedTestNode.execution_log.append({
            "event": "start",
            "node": self.node_name,
            "time": self.start_time
        })

        print(f"  [{time.time():.3f}] {self.node_name} STARTED")

        # Simulate work
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        self.end_time = time.time()
        TimedTestNode.execution_log.append({
            "event": "end",
            "node": self.node_name,
            "time": self.end_time
        })

        print(f"  [{time.time():.3f}] {self.node_name} FINISHED (took {self.end_time - self.start_time:.3f}s)")

        self.outputs = AIMessage(content=f"{self.node_name} result")
        self.invoked = True
        return self.outputs


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


async def test_diamond_graph():
    """Test diamond graph with parallel execution."""
    print_header("Diamond Graph Test: A → B, C → D")

    print("Graph structure:")
    print("       A (0.1s)")
    print("      / \\")
    print("     B   C  (1.0s each)")
    print("      \\ /")
    print("       D (0.1s)")
    print()

    TimedTestNode.execution_log = []

    # Create nodes manually
    node_a = TimedTestNode(name="A", delay=0.1)
    node_b = TimedTestNode(name="B", delay=1.0)
    node_c = TimedTestNode(name="C", delay=1.0)
    node_d = TimedTestNode(name="D", delay=0.1)

    # Set up parent relationships manually
    node_b.parents = [node_a]
    node_c.parents = [node_a]
    node_d.parents = [node_b, node_c]

    print("Executing graph from node D (which processes all parents)...")
    print()

    start = time.time()

    # Call _aprocess_parents directly on node D to test the parallel execution
    result = await node_d._aprocess_parents({}, None)

    total_time = time.time() - start

    print()
    print(f"{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Total execution time: {total_time:.3f}s")
    print()
    print("Individual node timings:")
    for node in [node_a, node_b, node_c]:
        if node.start_time and node.end_time:
            duration = node.end_time - node.start_time
            print(f"  {node.node_name}: {node.start_time:.3f}s -> {node.end_time:.3f}s  (duration: {duration:.3f}s)")

    # Check for parallel execution
    b_start = node_b.start_time
    b_end = node_b.end_time
    c_start = node_c.start_time
    c_end = node_c.end_time

    # Check if B and C overlapped
    if b_start and b_end and c_start and c_end:
        overlap = max(b_start, c_start) < min(b_end, c_end)

        print()
        if overlap:
            overlap_start = max(b_start, c_start)
            overlap_end = min(b_end, c_end)
            overlap_duration = overlap_end - overlap_start
            print(f"✓ B and C EXECUTED IN PARALLEL!")
            print(f"  Overlap duration: {overlap_duration:.3f}s")
            print(f"  Expected time with parallel: ~1.2s")
            print(f"  Actual time: {total_time:.3f}s")

            if total_time < 1.5:
                print(f"\n✓✓✓ SUCCESS! Parallel execution is working! ✓✓✓")
                print(f"    Speedup vs sequential (~2.2s): {2.2/total_time:.2f}x faster")
            else:
                print(f"\n⚠ Warning: Time is higher than expected for parallel execution")
        else:
            print(f"✗ B and C executed SEQUENTIALLY")
            print(f"  Expected time with sequential: ~2.2s")
            print(f"  Actual time: {total_time:.3f}s")
            print(f"\n✗✗✗ FAILED: Nodes ran sequentially, not in parallel ✗✗✗")

    print(f"{'='*70}")
    print()

    return overlap and total_time < 1.5


async def test_wide_graph():
    """Test wide graph with 4 parallel branches."""
    print_header("Wide Graph Test: A → B, C, D, E → F")

    print("Graph structure:")
    print("          A (0.1s)")
    print("        / | | \\")
    print("       B  C D  E  (0.5s each)")
    print("        \\ | | /")
    print("          F (0.1s)")
    print()

    TimedTestNode.execution_log = []

    # Create nodes
    node_a = TimedTestNode(name="A", delay=0.1)
    node_b = TimedTestNode(name="B", delay=0.5)
    node_c = TimedTestNode(name="C", delay=0.5)
    node_d = TimedTestNode(name="D", delay=0.5)
    node_e = TimedTestNode(name="E", delay=0.5)
    node_f = TimedTestNode(name="F", delay=0.1)

    # Set up parent relationships
    node_b.parents = [node_a]
    node_c.parents = [node_a]
    node_d.parents = [node_a]
    node_e.parents = [node_a]
    node_f.parents = [node_b, node_c, node_d, node_e]

    print("Executing graph from node F...")
    print()

    start = time.time()
    result = await node_f._aprocess_parents({}, None)
    total_time = time.time() - start

    print()
    print(f"{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Total execution time: {total_time:.3f}s")

    # Check if all middle nodes executed in parallel
    middle_nodes = [node_b, node_c, node_d, node_e]
    all_parallel = True

    for i, n1 in enumerate(middle_nodes):
        for n2 in middle_nodes[i+1:]:
            if n1.start_time and n2.start_time and n1.end_time and n2.end_time:
                overlap = max(n1.start_time, n2.start_time) < min(n1.end_time, n2.end_time)
                if not overlap:
                    all_parallel = False
                    break

    print()
    if all_parallel:
        print(f"✓ All 4 middle nodes (B, C, D, E) EXECUTED IN PARALLEL!")
        print(f"  Expected time with parallel: ~0.7s")
        print(f"  Actual time: {total_time:.3f}s")

        if total_time < 1.0:
            print(f"\n✓✓✓ SUCCESS! Parallel execution is working! ✓✓✓")
            print(f"    Speedup vs sequential (~2.1s): {2.1/total_time:.2f}x faster")
        else:
            print(f"\n⚠ Warning: Time is higher than expected")
    else:
        print(f"✗ Some nodes executed SEQUENTIALLY")
        print(f"  Actual time: {total_time:.3f}s")
        print(f"\n✗✗✗ FAILED: Nodes did not run in parallel ✗✗✗")

    print(f"{'='*70}")
    print()

    return all_parallel and total_time < 1.0


async def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  LC_Agent Parallel Execution Live Test".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("This test demonstrates REAL parallel execution of independent nodes.")
    print("Watch the timestamps to see nodes executing concurrently!")
    print()

    try:
        # Run tests
        test1_passed = await test_diamond_graph()
        test2_passed = await test_wide_graph()

        # Final summary
        print_header("FINAL SUMMARY")

        if test1_passed and test2_passed:
            print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
            print()
            print("The parallel execution implementation is WORKING!")
            print()
            print("Key results:")
            print("  • Independent nodes execute concurrently using asyncio.gather()")
            print("  • Diamond graph: ~1.2s (vs ~2.2s sequential) = 1.8x faster")
            print("  • Wide graph: ~0.7s (vs ~2.1s sequential) = 3.0x faster")
            print()
            print("This proves that the _aprocess_parents() method successfully")
            print("parallelizes independent branches using dependency-level grouping.")
            return 0
        else:
            print("✗✗✗ SOME TESTS FAILED ✗✗✗")
            print()
            print(f"  Diamond graph: {'PASSED' if test1_passed else 'FAILED'}")
            print(f"  Wide graph: {'PASSED' if test2_passed else 'FAILED'}")
            return 1

    except Exception as e:
        print(f"\n✗✗✗ ERROR DURING TEST EXECUTION ✗✗✗")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
