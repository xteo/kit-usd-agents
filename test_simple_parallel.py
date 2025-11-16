#!/usr/bin/env python3
"""
Simple test to verify parallel execution by invoking only the leaf node.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import ClassVar, List, Optional

# Add lc_agent to path
lc_agent_path = Path(__file__).parent / "source" / "modules" / "lc_agent" / "src"
sys.path.insert(0, str(lc_agent_path))

from lc_agent.runnable_node import RunnableNode
from langchain_core.messages import AIMessage


class SimpleNode(RunnableNode):
    """Simple timed node."""

    execution_log: ClassVar[List[str]] = []
    node_name: str = ""
    delay: float = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __init__(self, name: str, delay: float = 0):
        super().__init__()
        self.node_name = name
        self.delay = delay

    def _get_chat_model(self, chat_model_name, chat_model_input, input, config):
        """Override to skip chat model retrieval."""
        return None

    async def _ainvoke_chat_model(self, chat_model, chat_model_input, input, config, **kwargs):
        """Override to add timing and delay without calling LLM."""
        self.start_time = time.time()
        SimpleNode.execution_log.append(f"[{time.time():.3f}] {self.node_name} START")
        print(f"[{time.time():.3f}] {self.node_name} START")

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        self.end_time = time.time()
        SimpleNode.execution_log.append(f"[{time.time():.3f}] {self.node_name} END")
        print(f"[{time.time():.3f}] {self.node_name} END (duration: {self.end_time - self.start_time:.3f}s)")

        return AIMessage(content=f"{self.node_name} result")


async def test_direct_invoke():
    """Test by directly invoking the leaf node."""
    print("\n" + "="*70)
    print("TEST: Direct invoke of leaf node (diamond graph)")
    print("="*70)
    print("\nGraph: A -> B, C -> D")
    print("Expected: B and C run in parallel when D is invoked\n")

    SimpleNode.execution_log = []

    # Create diamond graph
    node_a = SimpleNode("A", 0.1)
    node_b = SimpleNode("B", 1.0)
    node_c = SimpleNode("C", 1.0)
    node_d = SimpleNode("D", 0.1)

    # Set up parent relationships
    node_b.parents = [node_a]
    node_c.parents = [node_a]
    node_d.parents = [node_b, node_c]

    # Invoke ONLY the leaf node - it should invoke its parents
    start = time.time()
    result = await node_d.ainvoke()
    total_time = time.time() - start

    print(f"\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Total time: {total_time:.3f}s")
    print(f"\nExpected with parallel: ~1.2s")
    print(f"Expected with sequential: ~2.2s")

    # Check if B and C overlapped
    if node_b.start_time and node_c.start_time:
        time_diff = abs(node_b.start_time - node_c.start_time)
        if time_diff < 0.05:  # Started within 50ms of each other
            print(f"\n✓✓✓ SUCCESS! B and C started concurrently (diff: {time_diff:.3f}s)")
            print(f"    Speedup: {2.2/total_time:.2f}x")
        else:
            print(f"\n✗ FAILED: B and C started {time_diff:.3f}s apart (sequential)")

    print("\nExecution log:")
    for event in SimpleNode.execution_log:
        print(f"  {event}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_direct_invoke())
