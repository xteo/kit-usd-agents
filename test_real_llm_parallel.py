#!/usr/bin/env python3
"""
REAL LLM PARALLEL EXECUTION TEST

This test uses ACTUAL NVIDIA NIM API calls to prove that parallel execution
works with real LLM inference, not just async sleeps.

Requirements:
1. Set NVIDIA_API_KEY environment variable
2. Valid NVIDIA NIM API key from build.nvidia.com

Usage:
    export NVIDIA_API_KEY="your_api_key_here"
    python test_real_llm_parallel.py

This will:
1. Make TWO concurrent LLM calls (different prompts)
2. Measure timing to prove they run in parallel
3. Make a THIRD LLM call to summarize the two results
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from typing import Optional

# Add lc_agent to path
lc_agent_src = Path(__file__).parent / "source" / "modules" / "lc_agent" / "src"
sys.path.insert(0, str(lc_agent_src))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from lc_agent.runnable_node import RunnableNode
from lc_agent.runnable_network import RunnableNetwork


class RealLLMNode(RunnableNode):
    """
    RunnableNode that makes REAL LLM API calls.
    """

    node_name: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    model_name: str = "meta/llama-3.1-8b-instruct"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def __init__(self, name: str, system_prompt: str, user_prompt: str, model: str = "meta/llama-3.1-8b-instruct"):
        super().__init__()
        self.node_name = name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_name = model
        self.start_time = None
        self.end_time = None
        self.duration = None

    def _get_chat_model(self, chat_model_input, invoke_input, config):
        """Override to provide our NVIDIA model."""
        api_key = os.environ.get("NVIDIA_API_KEY")
        chat_model = ChatNVIDIA(
            model=self.model_name,
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            max_tokens=100
        )
        return chat_model

    async def ainvoke(self, input=None, config=None, **kwargs):
        """Execute with timing."""
        if self.invoked:
            return self.outputs

        print(f"\n[{time.time():.3f}] {self.node_name} - Starting LLM call...")
        print(f"  Model: {self.model_name}")
        print(f"  Prompt: {self.user_prompt[:60]}...")

        self.start_time = time.time()

        try:
            # Get the chat model
            chat_model = self._get_chat_model(None, input, config)

            # Prepare messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.user_prompt)
            ]

            # Make the ACTUAL LLM API call
            result = await chat_model.ainvoke(messages)

            self.end_time = time.time()
            self.duration = self.end_time - self.start_time

            print(f"[{self.end_time:.3f}] {self.node_name} - FINISHED (took {self.duration:.2f}s)")
            print(f"  Response: {result.content[:100]}...")

            self.outputs = result
            self.invoked = True
            return self.outputs

        except Exception as e:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            print(f"[{self.end_time:.3f}] {self.node_name} - ERROR after {self.duration:.2f}s: {e}")
            raise


def check_api_key():
    """Check if NVIDIA_API_KEY is set."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("=" * 70)
        print("ERROR: NVIDIA_API_KEY environment variable not set!")
        print("=" * 70)
        print()
        print("To run this test, you need an NVIDIA NIM API key.")
        print()
        print("Steps to get an API key:")
        print("1. Go to https://build.nvidia.com/")
        print("2. Sign in with your NVIDIA account")
        print("3. Navigate to any model (e.g., meta/llama-3.1-8b-instruct)")
        print("4. Click 'Get API Key'")
        print("5. Copy the key")
        print()
        print("Then set it in your environment:")
        print("  export NVIDIA_API_KEY='your_key_here'")
        print()
        print("Or set it for this session only:")
        print("  NVIDIA_API_KEY='your_key_here' python test_real_llm_parallel.py")
        print()
        return False

    print(f"✓ NVIDIA_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    return True


async def test_diamond_with_real_llms():
    """
    Test diamond graph with REAL LLM calls:

         A (root)
        / \
       B   C  (TWO CONCURRENT LLM CALLS)
        \ /
         D  (THIRD LLM CALL - summarizes B and C)
    """
    print("=" * 70)
    print("TEST: Diamond Graph with REAL LLM API Calls")
    print("=" * 70)
    print()
    print("Graph structure:")
    print("         A (setup)")
    print("        / \\")
    print("       B   C  (TWO CONCURRENT LLM CALLS)")
    print("        \\ /")
    print("         D  (THIRD LLM CALL - summarizes)")
    print()
    print("This will make THREE actual API calls to NVIDIA NIM.")
    print("B and C should run CONCURRENTLY (same timestamp).")
    print("D should wait for both B and C to complete.")
    print()

    # Create nodes with REAL prompts
    with RunnableNetwork() as network:
        # Node A: Root (no LLM call, just setup)
        node_a = RunnableNode()
        node_a.name = "A-Setup"
        # Mark as already invoked so it doesn't try to call LLM
        node_a.invoked = True
        node_a.outputs = SystemMessage(content="Setup complete")

        # Node B: First LLM call
        node_b = RealLLMNode(
            name="B-HistoryOfAI",
            system_prompt="You are a helpful AI assistant. Be brief and concise.",
            user_prompt="In 2 sentences, explain the history of artificial intelligence."
        )
        node_b.parents = [node_a]

        # Node C: Second LLM call (CONCURRENT with B)
        node_c = RealLLMNode(
            name="C-FutureOfAI",
            system_prompt="You are a helpful AI assistant. Be brief and concise.",
            user_prompt="In 2 sentences, explain the future potential of artificial intelligence."
        )
        node_c.parents = [node_a]

        # Node D: Third LLM call (summarizes B and C)
        node_d = RealLLMNode(
            name="D-Summarize",
            system_prompt="You are a helpful AI assistant. Be brief and concise.",
            user_prompt="Combine these two perspectives about AI into one coherent sentence."
        )
        node_d.parents = [node_b, node_c]

    print("Starting execution...")
    print()

    overall_start = time.time()

    # Execute the network (D will trigger execution of A, B, C)
    try:
        result = await network.ainvoke()
        overall_end = time.time()
        overall_duration = overall_end - overall_start

        print()
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)
        print()

        # Check if B and C ran in parallel
        b_start = node_b.start_time
        b_end = node_b.end_time
        b_duration = node_b.duration

        c_start = node_c.start_time
        c_end = node_c.end_time
        c_duration = node_c.duration

        d_duration = node_d.duration

        print(f"Node B (History): {b_duration:.2f}s")
        print(f"Node C (Future):  {c_duration:.2f}s")
        print(f"Node D (Summary): {d_duration:.2f}s")
        print()

        # Check for parallel execution
        if b_start and c_start and b_end and c_end:
            # Calculate overlap
            overlap_start = max(b_start, c_start)
            overlap_end = min(b_end, c_end)
            overlap = overlap_start < overlap_end

            if overlap:
                overlap_duration = overlap_end - overlap_start
                print(f"✓✓✓ B and C EXECUTED IN PARALLEL! ✓✓✓")
                print(f"    Overlap: {overlap_duration:.2f}s")
                print()

                # Calculate expected vs actual time
                sequential_time = b_duration + c_duration + d_duration
                parallel_time = max(b_duration, c_duration) + d_duration

                print(f"Timing analysis:")
                print(f"  If SEQUENTIAL: ~{sequential_time:.2f}s (B + C + D)")
                print(f"  If PARALLEL:   ~{parallel_time:.2f}s (max(B,C) + D)")
                print(f"  ACTUAL:        {overall_duration:.2f}s")
                print()

                if overall_duration < sequential_time * 0.8:
                    speedup = sequential_time / overall_duration
                    print(f"✓✓✓ SUCCESS! REAL LLMs ran in PARALLEL! ✓✓✓")
                    print(f"    Speedup: {speedup:.2f}x faster than sequential")
                    print()
                    print("This proves that parallel execution works with REAL LLM API calls!")
                    return True
                else:
                    print(f"⚠ B and C overlapped but total time suggests some blocking")
                    return False
            else:
                print(f"✗ B and C executed SEQUENTIALLY")
                print(f"  B: {b_start:.2f}s -> {b_end:.2f}s")
                print(f"  C: {c_start:.2f}s -> {c_end:.2f}s")
                return False
        else:
            print("✗ Unable to measure timing")
            return False

    except Exception as e:
        print()
        print(f"✗✗✗ ERROR DURING EXECUTION ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the real LLM parallel execution test."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  REAL LLM Parallel Execution Test".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("This test makes ACTUAL API calls to NVIDIA NIM to prove")
    print("that parallel execution works with real LLM inference.")
    print()

    # Check for API key
    if not check_api_key():
        return 1

    print()
    print("WARNING: This test will make REAL API calls!")
    print("Expected cost: ~3 API calls to meta/llama-3.1-8b-instruct")
    print()

    try:
        success = await test_diamond_with_real_llms()

        print()
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print()

        if success:
            print("✓✓✓ TEST PASSED! ✓✓✓")
            print()
            print("PROVEN: Independent LLM API calls execute in parallel!")
            print("This demonstrates that the parallel execution implementation")
            print("works with REAL network calls to LLM APIs, not just async sleeps.")
            print()
            return 0
        else:
            print("✗ TEST FAILED")
            print("LLM calls did not execute in parallel as expected.")
            print()
            return 1

    except Exception as e:
        print(f"\n✗✗✗ ERROR ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
