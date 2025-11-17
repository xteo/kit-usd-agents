#!/usr/bin/env python3
## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
LC Agent CLI with Planning - Demonstrates parallel task execution with dependency graphs.

This enhanced CLI adds a "planning" mode that shows how the planning agent creates
plans with dependencies and executes them with parallel task execution.

Usage:
    # Interactive planning mode
    lc-agent --assistant planning

    # Single query with planning
    lc-agent --assistant planning --query "Deploy a microservices application"

    # Use different model
    lc-agent --assistant planning --model llama-maverick
"""

import asyncio
import argparse
import sys
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path


# Add planning module to path
planning_module_path = Path(__file__).parent.parent.parent / "agents" / "planning" / "src"
if planning_module_path.exists():
    sys.path.insert(0, str(planning_module_path))


def display_plan(plan: Dict[str, Any]):
    """Display a plan with dependencies in a visual format."""
    print("\n" + "=" * 70)
    print(f"📋 PLAN: {plan['title']}")
    print("=" * 70)

    # Group steps by their dependencies to show phases
    steps = plan["steps"]

    # Find parallel opportunities
    parallel_groups = {}
    for step in steps:
        dep_key = tuple(sorted(step.get("dependencies", [])))
        if dep_key not in parallel_groups:
            parallel_groups[dep_key] = []
        parallel_groups[dep_key].append(step)

    # Display steps
    for step in steps:
        step_num = step["step_number"]
        title = step["title"]
        dependencies = step.get("dependencies", [])
        step_type = step.get("step_type", "action")

        # Determine icon based on type
        if step_type == "planning_review":
            icon = "🔍"
        else:
            icon = "⚙️"

        # Check if this step can run in parallel
        parallel_marker = ""
        dep_key = tuple(sorted(dependencies))
        if len(parallel_groups[dep_key]) > 1:
            parallel_marker = " [PARALLEL]"

        # Format dependencies
        if dependencies:
            dep_str = f" (depends on: {', '.join(map(str, dependencies))})"
        else:
            dep_str = " (independent - can start immediately)"

        print(f"\n{icon} Step {step_num}: {title}{parallel_marker}")
        print(f"   Dependencies:{dep_str}")

        # Show details if any
        details = step.get("details", [])
        if details:
            for detail in details[:3]:  # Show first 3 details
                print(f"   - {detail}")
            if len(details) > 3:
                print(f"   ... and {len(details) - 3} more")

    # Show execution phases
    print("\n" + "-" * 70)
    print("📊 EXECUTION PHASES:")
    print("-" * 70)

    # Simulate execution order based on dependencies
    from omni_aiq_planning.modifiers.planning_modifier import DependencyGraph

    try:
        graph = DependencyGraph(steps)
        phase = 1
        while not graph.is_complete():
            ready = graph.get_ready_steps()
            if not ready:
                break

            if len(ready) > 1:
                print(f"Phase {phase} (PARALLEL): Launch steps {ready}")
            else:
                print(f"Phase {phase}: Launch step {ready[0]}")

            for step_num in ready:
                graph.mark_completed(step_num)

            phase += 1

        parallel_phases = sum(1 for _ in range(phase) if len(graph.get_ready_steps()) > 1)
        print(f"\nTotal phases: {phase - 1}")
        print(f"Parallel phases: {parallel_phases}")
        print(f"Potential speedup: {len(steps) / (phase - 1):.1f}x vs sequential")

    except Exception as e:
        print(f"Could not simulate execution: {e}")

    print("=" * 70)


class ExecutionTracker:
    """Track and display execution progress in real-time."""

    def __init__(self):
        self.steps_status = {}
        self.start_times = {}
        self.end_times = {}
        self.currently_running = []

    def mark_started(self, step_num: int, step_title: str):
        """Mark a step as started."""
        self.steps_status[step_num] = "in_progress"
        self.start_times[step_num] = time.time()
        self.currently_running.append(step_num)

        parallel_marker = ""
        if len(self.currently_running) > 1:
            parallel_marker = f" [Running in parallel with: {', '.join(map(str, [s for s in self.currently_running if s != step_num]))}]"

        print(f"\n🚀 Started Step {step_num}: {step_title}{parallel_marker}")

    def mark_completed(self, step_num: int, step_title: str):
        """Mark a step as completed."""
        self.steps_status[step_num] = "completed"
        self.end_times[step_num] = time.time()
        if step_num in self.currently_running:
            self.currently_running.remove(step_num)

        duration = self.end_times[step_num] - self.start_times[step_num]
        print(f"✅ Completed Step {step_num}: {step_title} ({duration:.2f}s)")

    def display_summary(self):
        """Display execution summary."""
        if not self.start_times or not self.end_times:
            return

        total_time = max(self.end_times.values()) - min(self.start_times.values())
        sequential_time = sum(
            self.end_times[s] - self.start_times[s]
            for s in self.end_times.keys()
        )
        speedup = sequential_time / total_time if total_time > 0 else 1

        print("\n" + "=" * 70)
        print("📈 EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Sequential time would be: {sequential_time:.2f}s")
        print(f"Actual speedup: {speedup:.2f}x")
        print("=" * 70)


async def run_planning_mode(query: str, model: str, verbose: bool, stream: bool, interactive: bool = False):
    """Run in planning mode with multi-agent execution."""
    try:
        from lc_agent import (
            RunnableNetwork,
            RunnableNode,
            get_node_factory,
            RunnableHumanNode,
            MultiAgentNetworkNode
        )
        from omni_aiq_planning.nodes.planning_node import PlanningGenNode, PlanningNetworkNode
        from omni_aiq_planning.modifiers.planning_modifier import PlanningModifier

        # Check for NVIDIA API key
        if not os.environ.get("NVIDIA_API_KEY"):
            print("[ERROR] NVIDIA_API_KEY environment variable not set.")
            print("        Set it to use NVIDIA models:")
            print("        export NVIDIA_API_KEY=your_key_here  # Linux/Mac")
            print("        set NVIDIA_API_KEY=your_key_here     # Windows")
            print()
            return

        # Register NVIDIA chat models
        try:
            from lc_agent_cli import register_all
            register_all(verbose=verbose)
            if verbose:
                print("[INFO] Chat models registered successfully")
        except ImportError as e:
            print(f"[ERROR] Failed to import lc_agent_cli: {e}")
            return
        except Exception as e:
            print(f"[ERROR] Failed to register chat models: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return

        # Register node types
        get_node_factory().register(RunnableNode)
        get_node_factory().register(RunnableHumanNode)
        get_node_factory().register(PlanningGenNode)
        get_node_factory().register(PlanningNetworkNode)

        print("\n" + "=" * 70)
        print("🤖 LC AGENT - PLANNING MODE")
        print("=" * 70)
        print(f"Model: {model}")
        print("Mode: Planning with Multi-Agent Execution")
        print("=" * 70)

        print(f"\n📝 Query: {query}\n")
        print("⏳ Generating plan with dependencies...\n")

        # Create planning network
        with PlanningNetworkNode(
            default_node="PlanningGenNode",
            chat_model_name=model
        ) as planning_network:
            # Add planning modifier
            planning_network.add_modifier(PlanningModifier())

            # Create human input
            RunnableHumanNode(query)

            # Generate plan
            result = await planning_network.ainvoke()

            # Extract plan from network metadata
            plan = planning_network.metadata.get("current_plan")

            if plan:
                # Display the generated plan
                display_plan(plan)

                # Ask user if they want to execute (in interactive mode)
                if interactive:
                    user_response = input("\nExecute this plan? (yes/no): ").strip().lower()
                    if user_response not in ['yes', 'y']:
                        print("Plan execution cancelled.")
                        return

                # Execute the plan (for now, we'll simulate)
                print("\n" + "=" * 70)
                print("🚀 EXECUTING PLAN...")
                print("=" * 70)
                print("Note: Full multi-agent execution would happen here.")
                print("For now, showing how parallel execution would work:\n")

                tracker = ExecutionTracker()

                # Simulate execution based on dependency graph
                from omni_aiq_planning.modifiers.planning_modifier import DependencyGraph

                try:
                    graph = DependencyGraph(plan["steps"])

                    while not graph.is_complete():
                        ready = graph.get_ready_steps()
                        if not ready:
                            break

                        # Start all ready steps (parallel execution!)
                        for step_num in ready:
                            step = next(s for s in plan["steps"] if s["step_number"] == step_num)
                            tracker.mark_started(step_num, step["title"])

                        # Simulate execution time
                        await asyncio.sleep(1.0)

                        # Complete all started steps
                        for step_num in ready:
                            step = next(s for s in plan["steps"] if s["step_number"] == step_num)
                            tracker.mark_completed(step_num, step["title"])
                            graph.mark_completed(step_num)

                    tracker.display_summary()

                except Exception as e:
                    print(f"Error during execution simulation: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()

            else:
                print("⚠️  No valid plan was generated.")
                print("The planning agent's response:")
                if hasattr(result, 'content'):
                    print(result.content)

    except ImportError as e:
        print(f"[ERROR] Failed to import required modules: {e}")
        print("\nPlease ensure the planning module is installed:")
        print("  cd source/modules/agents/planning && pip install -e .")
        if verbose:
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def setup_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LC Agent CLI with Planning - Interactive AI agent framework with parallel execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive planning mode
  %(prog)s --assistant planning

  # Single query with planning
  %(prog)s --assistant planning --query "Deploy a microservices app with auth, user, and API services"

  # Use different model
  %(prog)s --assistant planning --model llama-maverick

  # Verbose mode to see detailed execution
  %(prog)s --assistant planning --verbose --query "Build and test a Python web application"

For more information, see claude.md in the repository root.
        """
    )

    parser.add_argument(
        "--model",
        default=os.environ.get("LC_AGENT_MODEL", "gpt-120b"),
        help="Chat model to use (default: gpt-120b or LC_AGENT_MODEL env var)"
    )

    parser.add_argument(
        "--query",
        help="Single query to execute (non-interactive mode)"
    )

    parser.add_argument(
        "--assistant",
        choices=["default", "usd", "planning"],
        default="planning",
        help="Assistant type to use (default: planning)"
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


async def run_interactive_planning():
    """Run interactive planning mode."""
    args = setup_args()

    print("=" * 70)
    print("🤖 LC AGENT - INTERACTIVE PLANNING MODE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print("\nType your planning request, or:")
    print("  'quit' or 'exit' to end the session")
    print("  'help' for examples")
    print("=" * 70)
    print()

    while True:
        try:
            # Get user input
            user_input = input("\nYour planning request: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'help':
                print("\nExample planning requests:")
                print("  - Deploy a microservices application with auth, user, and API services")
                print("  - Build and test a Python web application")
                print("  - Set up a CI/CD pipeline for a React app")
                print("  - Create a data processing pipeline with validation")
                print()
                continue

            # Process planning request
            await run_planning_mode(
                user_input,
                args.model,
                args.verbose,
                args.stream,
                interactive=True
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            print()


def main():
    """Main entry point."""
    args = setup_args()

    if args.assistant != "planning":
        print("[ERROR] This CLI only supports planning mode.")
        print("        Use: --assistant planning")
        sys.exit(1)

    if args.query:
        # Single query mode
        asyncio.run(run_planning_mode(
            args.query,
            args.model,
            args.verbose,
            args.stream,
            interactive=False
        ))
    else:
        # Interactive mode
        asyncio.run(run_interactive_planning())


if __name__ == "__main__":
    main()
