#!/usr/bin/env python3
## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
LC Agent CLI - Interactive command-line interface for the LC Agent framework.

This script provides a simple CLI for interacting with the LC Agent system.
It demonstrates basic usage patterns and can be extended for specific use cases.
"""

import asyncio
import argparse
import sys
import os
from typing import Optional


def setup_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LC Agent CLI - Interactive AI agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default settings
  %(prog)s

  # Use specific chat model
  %(prog)s --model gpt-4

  # Run in verbose mode
  %(prog)s --verbose

  # Execute a single query
  %(prog)s --query "Explain USD prims"

  # Use USD assistant mode
  %(prog)s --assistant usd

For more information, see claude.md in the repository root.
        """
    )

    parser.add_argument(
        "--model",
        default=os.environ.get("LC_AGENT_MODEL", "openai/gpt-oss-120b"),
        help="Chat model to use (default: openai/gpt-oss-120b or LC_AGENT_MODEL env var)"
    )

    parser.add_argument(
        "--query",
        help="Single query to execute (non-interactive mode)"
    )

    parser.add_argument(
        "--assistant",
        choices=["default", "usd"],
        default="default",
        help="Assistant type to use (default: default)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream responses (default: True)"
    )

    return parser.parse_args()


async def run_interactive(model: str, assistant_type: str, verbose: bool, stream: bool):
    """Run interactive mode."""
    try:
        from lc_agent import RunnableNetwork, RunnableNode, get_node_factory

        # Check for NVIDIA API key
        if not os.environ.get("NVIDIA_API_KEY"):
            print("[WARN] NVIDIA_API_KEY environment variable not set.")
            print("       Set it to use NVIDIA models:")
            print("       export NVIDIA_API_KEY=your_key_here  # Linux/Mac")
            print("       set NVIDIA_API_KEY=your_key_here     # Windows")
            print()

        # Try to import chat models if available
        try:
            from lc_agent.chat_models import register_all as register_chat_models
            register_chat_models()
            if verbose:
                print("[INFO] Chat models registered")
        except ImportError as e:
            if verbose:
                print(f"[WARN] Chat models not available: {e}")
                print("[WARN] Continuing without them")

        # Register node types
        get_node_factory().register(RunnableNode)

        # Use USD assistant if requested
        if assistant_type == "usd":
            try:
                from lc_agent import USDAssistantNode
                get_node_factory().register(USDAssistantNode)
                default_node = "USDAssistantNode"
                if verbose:
                    print("[INFO] Using USD Assistant mode")
            except ImportError:
                print("[WARN] USDAssistantNode not available, using default")
                default_node = "RunnableNode"
        else:
            default_node = "RunnableNode"

        print("=" * 60)
        print("LC Agent Interactive CLI")
        print("=" * 60)
        print(f"Model: {model}")
        print(f"Assistant: {assistant_type}")
        print(f"Stream mode: {stream}")
        print("\nType 'quit' or 'exit' to end the session")
        print("Type 'help' for usage information")
        print("=" * 60)
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  quit, exit, q  - Exit the CLI")
                    print("  help           - Show this help message")
                    print("  <your query>   - Ask the agent anything")
                    print()
                    continue

                # Create network and process query
                from lc_agent import RunnableHumanNode
                get_node_factory().register(RunnableHumanNode)

                with RunnableNetwork(
                    default_node=default_node,
                    chat_model_name=model
                ) as network:
                    RunnableHumanNode(user_input)

                print("\nAgent: ", end="", flush=True)

                if stream:
                    async for chunk in network.astream():
                        if hasattr(chunk, 'content'):
                            print(chunk.content, end="", flush=True)
                    print("\n")
                else:
                    result = await network.ainvoke()
                    if hasattr(result, 'content'):
                        print(result.content)
                    print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\n[ERROR] {type(e).__name__}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                print()

    except ImportError as e:
        print(f"[ERROR] Failed to import LC Agent: {e}")
        print("\nPlease ensure lc_agent is installed:")
        print("  ./dev-install.sh")
        print("\nOr:")
        print("  cd source/modules/lc_agent && pip install -e .")
        sys.exit(1)


async def run_single_query(query: str, model: str, assistant_type: str, verbose: bool, stream: bool):
    """Run a single query and exit."""
    try:
        from lc_agent import RunnableNetwork, RunnableNode, get_node_factory, RunnableHumanNode

        # Check for NVIDIA API key
        if not os.environ.get("NVIDIA_API_KEY"):
            print("[ERROR] NVIDIA_API_KEY environment variable not set.")
            print("        Set it to use NVIDIA models:")
            print("        export NVIDIA_API_KEY=your_key_here  # Linux/Mac")
            print("        set NVIDIA_API_KEY=your_key_here     # Windows")
            print()
            sys.exit(1)

        # Try to import chat models if available
        try:
            from lc_agent.chat_models import register_all as register_chat_models
            register_chat_models()
        except ImportError as e:
            if verbose:
                print(f"[WARN] Failed to register chat models: {e}")
            pass

        # Register node types
        get_node_factory().register(RunnableNode)
        get_node_factory().register(RunnableHumanNode)

        # Use USD assistant if requested
        if assistant_type == "usd":
            try:
                from lc_agent import USDAssistantNode
                get_node_factory().register(USDAssistantNode)
                default_node = "USDAssistantNode"
            except ImportError:
                default_node = "RunnableNode"
        else:
            default_node = "RunnableNode"

        # Create network and process query
        with RunnableNetwork(
            default_node=default_node,
            chat_model_name=model
        ) as network:
            RunnableHumanNode(query)

        if stream:
            async for chunk in network.astream():
                if hasattr(chunk, 'content'):
                    print(chunk.content, end="", flush=True)
            print()
        else:
            result = await network.ainvoke()
            if hasattr(result, 'content'):
                print(result.content)

    except ImportError as e:
        print(f"[ERROR] Failed to import LC Agent: {e}")
        print("\nPlease ensure lc_agent is installed:")
        print("  ./dev-install.sh")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    args = setup_args()

    if args.query:
        # Single query mode
        asyncio.run(run_single_query(
            args.query,
            args.model,
            args.assistant,
            args.verbose,
            args.stream
        ))
    else:
        # Interactive mode
        asyncio.run(run_interactive(
            args.model,
            args.assistant,
            args.verbose,
            args.stream
        ))


if __name__ == "__main__":
    main()
