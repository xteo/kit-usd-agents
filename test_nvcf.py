#!/usr/bin/env python3
"""
Quick test script for NVCF API connectivity.
This helps debug API issues independently of the full CLI.
"""

import os
import asyncio
import sys

async def test_nvcf():
    """Test NVCF API directly."""

    # Check API key
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[ERROR] NVIDIA_API_KEY environment variable not set")
        sys.exit(1)

    print(f"[INFO] API key present: {api_key[:10]}...")

    # Import after checking key
    try:
        from lc_agent.chat_models.chat_nvcf import ChatNVCF
    except ImportError as e:
        print(f"[ERROR] Failed to import ChatNVCF: {e}")
        sys.exit(1)

    # Create model instance
    print("[INFO] Creating ChatNVCF instance...")
    model = ChatNVCF(
        model="openai/gpt-oss-120b",
        max_tokens=100,
        temperature=0.1,
        api_token=api_key,
    )

    print(f"[INFO] Invoke URL: {model._invoke_url}")
    print(f"[INFO] Headers: {model._header}")

    # Test with a simple message
    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content="Say hello in French")]

    print("\n[INFO] Testing streaming...")
    print("[INFO] Response:")
    print("-" * 60)

    try:
        chunk_count = 0
        async for chunk in model._astream(messages):
            chunk_count += 1
            if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                print(chunk.message.content, end="", flush=True)
        print()
        print("-" * 60)
        print(f"[INFO] Received {chunk_count} chunks")
    except Exception as e:
        print(f"\n[ERROR] Exception during streaming: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_nvcf())
