#!/usr/bin/env python3
## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
LC Agent Planning CLI - DEMO VERSION

This demo version shows how the planning system works with parallel execution
using pre-generated plans. Use this to see the parallel execution in action
without requiring network access or API keys.

Usage:
    python demo-planning-cli.py
    python demo-planning-cli.py --scenario microservices
    python demo-planning-cli.py --scenario data-pipeline
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List


# Add planning module to path
sys.path.insert(0, str(Path(__file__).parent / "source" / "modules" / "lc_agent" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "source" / "modules" / "agents" / "planning" / "src"))


# Pre-generated sample plans
SAMPLE_PLANS = {
    "webapp": {
        "title": "Deploy Web Application",
        "steps": [
            {
                "step_number": 1,
                "title": "Build frontend application",
                "step_type": "action",
                "dependencies": [],
                "details": [
                    "Install React dependencies with npm",
                    "Run webpack build for production",
                    "Optimize and minify assets"
                ]
            },
            {
                "step_number": 2,
                "title": "Build backend API",
                "step_type": "action",
                "dependencies": [],
                "details": [
                    "Install Python dependencies",
                    "Run pytest test suite",
                    "Build Docker image for backend"
                ]
            },
            {
                "step_number": 3,
                "title": "Setup database",
                "step_type": "action",
                "dependencies": [],
                "details": [
                    "Provision PostgreSQL instance",
                    "Create database schemas",
                    "Set up initial tables"
                ]
            },
            {
                "step_number": 4,
                "title": "Planning Review - Build Verification",
                "step_type": "planning_review",
                "dependencies": [1, 2, 3],
                "details": [
                    "Review focus: Verify all builds succeeded",
                    "Decision points: All builds passed? Tests green?",
                    "Potential outcomes: CONTINUE or REPLAN"
                ]
            },
            {
                "step_number": 5,
                "title": "Deploy backend to production",
                "step_type": "action",
                "dependencies": [4],
                "details": [
                    "Deploy Docker container to Kubernetes",
                    "Configure environment variables",
                    "Expose service via load balancer"
                ]
            },
            {
                "step_number": 6,
                "title": "Deploy frontend to CDN",
                "step_type": "action",
                "dependencies": [4],
                "details": [
                    "Upload static assets to S3",
                    "Invalidate CloudFront cache",
                    "Configure DNS routing"
                ]
            },
            {
                "step_number": 7,
                "title": "Run integration tests",
                "step_type": "action",
                "dependencies": [5, 6],
                "details": [
                    "Execute end-to-end test suite",
                    "Verify frontend-backend integration",
                    "Check database connectivity"
                ]
            }
        ]
    },
    "microservices": {
        "title": "Deploy Microservices Application",
        "steps": [
            {
                "step_number": 1,
                "title": "Build authentication service",
                "step_type": "action",
                "dependencies": [],
                "details": ["Compile Go code", "Run unit tests", "Build Docker image"]
            },
            {
                "step_number": 2,
                "title": "Build user service",
                "step_type": "action",
                "dependencies": [],
                "details": ["Compile Python code", "Run pytest", "Build Docker image"]
            },
            {
                "step_number": 3,
                "title": "Build API gateway",
                "step_type": "action",
                "dependencies": [],
                "details": ["Compile Node.js", "Run Jest tests", "Build Docker image"]
            },
            {
                "step_number": 4,
                "title": "Build database migrations",
                "step_type": "action",
                "dependencies": [],
                "details": ["Generate SQL scripts", "Validate syntax", "Package migrations"]
            },
            {
                "step_number": 5,
                "title": "Planning Review - Build Verification",
                "step_type": "planning_review",
                "dependencies": [1, 2, 3, 4],
                "details": ["Review focus: Verify all builds succeeded"]
            },
            {
                "step_number": 6,
                "title": "Deploy database and run migrations",
                "step_type": "action",
                "dependencies": [5],
                "details": ["Provision PostgreSQL", "Apply migrations", "Verify schema"]
            },
            {
                "step_number": 7,
                "title": "Deploy auth service",
                "step_type": "action",
                "dependencies": [6],
                "details": ["Deploy to k8s", "Configure secrets", "Expose service"]
            },
            {
                "step_number": 8,
                "title": "Deploy user service",
                "step_type": "action",
                "dependencies": [6],
                "details": ["Deploy to k8s", "Configure secrets", "Expose service"]
            },
            {
                "step_number": 9,
                "title": "Deploy API gateway",
                "step_type": "action",
                "dependencies": [6],
                "details": ["Deploy to k8s", "Configure routing", "Expose service"]
            },
            {
                "step_number": 10,
                "title": "Run integration tests",
                "step_type": "action",
                "dependencies": [7, 8, 9],
                "details": ["Execute test suite", "Verify API endpoints", "Check auth flows"]
            },
            {
                "step_number": 11,
                "title": "Planning Review - Deployment Verification",
                "step_type": "planning_review",
                "dependencies": [10],
                "details": ["Review focus: Assess deployment health and test results"]
            }
        ]
    },
    "data-pipeline": {
        "title": "Build Data Processing Pipeline",
        "steps": [
            {
                "step_number": 1,
                "title": "Extract data from source A",
                "step_type": "action",
                "dependencies": [],
                "details": ["Connect to MySQL database", "Query customer data", "Export to CSV"]
            },
            {
                "step_number": 2,
                "title": "Extract data from source B",
                "step_type": "action",
                "dependencies": [],
                "details": ["Connect to REST API", "Fetch order data", "Export to JSON"]
            },
            {
                "step_number": 3,
                "title": "Extract data from source C",
                "step_type": "action",
                "dependencies": [],
                "details": ["Read from S3 bucket", "Parse logs", "Extract metrics"]
            },
            {
                "step_number": 4,
                "title": "Validate extracted data",
                "step_type": "action",
                "dependencies": [1, 2, 3],
                "details": ["Check data completeness", "Verify schemas", "Flag anomalies"]
            },
            {
                "step_number": 5,
                "title": "Transform customer data",
                "step_type": "action",
                "dependencies": [4],
                "details": ["Clean and normalize", "Apply business rules", "Enrich with metadata"]
            },
            {
                "step_number": 6,
                "title": "Transform order data",
                "step_type": "action",
                "dependencies": [4],
                "details": ["Clean and normalize", "Calculate aggregates", "Join with customer data"]
            },
            {
                "step_number": 7,
                "title": "Transform log data",
                "step_type": "action",
                "dependencies": [4],
                "details": ["Parse timestamps", "Extract events", "Aggregate metrics"]
            },
            {
                "step_number": 8,
                "title": "Load to data warehouse",
                "step_type": "action",
                "dependencies": [5, 6, 7],
                "details": ["Create staging tables", "Bulk load data", "Merge into production"]
            },
            {
                "step_number": 9,
                "title": "Verify data quality",
                "step_type": "action",
                "dependencies": [8],
                "details": ["Run quality checks", "Compare row counts", "Validate integrity"]
            }
        ]
    }
}


def display_plan(plan: Dict[str, Any]):
    """Display a plan with dependencies in a visual format."""
    print("\n" + "=" * 70)
    print(f"📋 PLAN: {plan['title']}")
    print("=" * 70)

    steps = plan["steps"]

    # Display steps with dependency information
    for step in steps:
        step_num = step["step_number"]
        title = step["title"]
        dependencies = step.get("dependencies", [])
        step_type = step.get("step_type", "action")

        # Icon based on type
        icon = "🔍" if step_type == "planning_review" else "⚙️"

        # Format dependencies
        if dependencies:
            dep_str = f" (depends on: {', '.join(map(str, dependencies))})"
        else:
            dep_str = " ⭐ INDEPENDENT - Can start immediately"

        print(f"\n{icon} Step {step_num}: {title}")
        print(f"   {dep_str}")

        # Show details
        details = step.get("details", [])
        for detail in details:
            print(f"   - {detail}")

    # Analyze and show execution phases
    print("\n" + "-" * 70)
    print("📊 EXECUTION ANALYSIS:")
    print("-" * 70)

    try:
        from omni_aiq_planning.modifiers.planning_modifier import DependencyGraph

        graph = DependencyGraph(steps)

        # Validate graph
        is_valid, error = graph.validate_dependencies()
        if not is_valid:
            print(f"⚠️  Invalid dependency graph: {error}")
            return

        print("✅ Dependency graph is valid\n")

        # Simulate execution to show phases
        phases = []
        phase = 1
        while not graph.is_complete():
            ready = graph.get_ready_steps()
            if not ready:
                break

            if len(ready) > 1:
                phases.append((f"Phase {phase} (PARALLEL)", ready.copy()))
                step_titles = [next(s["title"] for s in steps if s["step_number"] == n)[:30] for n in ready]
                print(f"Phase {phase} (🚀 PARALLEL): Steps {ready}")
                for i, (num, title) in enumerate(zip(ready, step_titles)):
                    print(f"   └─ Step {num}: {title}...")
            else:
                phases.append((f"Phase {phase}", ready.copy()))
                step_title = next(s["title"] for s in steps if s["step_number"] == ready[0])[:40]
                print(f"Phase {phase}: Step {ready[0]} - {step_title}")

            for step_num in ready:
                graph.mark_completed(step_num)

            phase += 1

        parallel_phases = sum(1 for name, steps_list in phases if "PARALLEL" in name)
        total_steps = len(plan["steps"])
        total_phases = len(phases)

        print(f"\n📈 Summary:")
        print(f"   Total steps: {total_steps}")
        print(f"   Total phases: {total_phases}")
        print(f"   Parallel phases: {parallel_phases}")
        print(f"   Potential speedup: {total_steps / total_phases:.1f}x vs sequential")

    except ImportError as e:
        print(f"⚠️  Could not import DependencyGraph: {e}")
        print("   Install planning module: cd source/modules/agents/planning && pip install -e .")

    print("=" * 70)


class ExecutionSimulator:
    """Simulate parallel execution with timing."""

    def __init__(self, plan: Dict[str, Any]):
        self.plan = plan
        self.steps_status = {}
        self.start_times = {}
        self.end_times = {}
        self.currently_running = []

    async def simulate(self):
        """Simulate execution with visual feedback."""
        try:
            from omni_aiq_planning.modifiers.planning_modifier import DependencyGraph

            graph = DependencyGraph(self.plan["steps"])

            print("\n" + "=" * 70)
            print("🚀 SIMULATING PARALLEL EXECUTION...")
            print("=" * 70)
            print()

            start_time = time.time()

            while not graph.is_complete():
                ready = graph.get_ready_steps()
                if not ready:
                    break

                # Start all ready steps (parallel execution!)
                for step_num in ready:
                    step = next(s for s in self.plan["steps"] if s["step_number"] == step_num)
                    self._mark_started(step_num, step["title"])

                # Simulate execution time (0.5-1.5 seconds per step)
                import random
                execution_time = random.uniform(0.5, 1.5)
                await asyncio.sleep(execution_time)

                # Complete all started steps
                for step_num in ready:
                    step = next(s for s in self.plan["steps"] if s["step_number"] == step_num)
                    self._mark_completed(step_num, step["title"])
                    graph.mark_completed(step_num)

            total_time = time.time() - start_time
            self._display_summary(total_time)

        except ImportError as e:
            print(f"⚠️  Could not import DependencyGraph: {e}")

    def _mark_started(self, step_num: int, step_title: str):
        """Mark step as started."""
        self.steps_status[step_num] = "in_progress"
        self.start_times[step_num] = time.time()
        self.currently_running.append(step_num)

        parallel_marker = ""
        if len(self.currently_running) > 1:
            others = [str(s) for s in self.currently_running if s != step_num]
            parallel_marker = f" [🔀 Running in parallel with: {', '.join(others)}]"

        print(f"🚀 Started Step {step_num}: {step_title[:50]}{parallel_marker}")

    def _mark_completed(self, step_num: int, step_title: str):
        """Mark step as completed."""
        self.steps_status[step_num] = "completed"
        self.end_times[step_num] = time.time()
        if step_num in self.currently_running:
            self.currently_running.remove(step_num)

        duration = self.end_times[step_num] - self.start_times[step_num]
        print(f"✅ Completed Step {step_num}: {step_title[:50]} ({duration:.2f}s)")

    def _display_summary(self, total_time: float):
        """Display execution summary."""
        sequential_time = sum(
            self.end_times[s] - self.start_times[s]
            for s in self.end_times.keys()
        )
        speedup = sequential_time / total_time if total_time > 0 else 1

        print("\n" + "=" * 70)
        print("📈 EXECUTION SUMMARY")
        print("=" * 70)
        print(f"⏱️  Total execution time: {total_time:.2f}s")
        print(f"⏱️  Sequential time would be: {sequential_time:.2f}s")
        print(f"🚀 Actual speedup: {speedup:.2f}x")
        print(f"⚡ Time saved: {sequential_time - total_time:.2f}s ({((speedup - 1) * 100):.0f}% faster)")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LC Agent Planning CLI - Demo Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run webapp deployment scenario
  python demo-planning-cli.py --scenario webapp

  # Run microservices deployment scenario
  python demo-planning-cli.py --scenario microservices

  # Run data pipeline scenario
  python demo-planning-cli.py --scenario data-pipeline

  # Execute the plan simulation
  python demo-planning-cli.py --scenario microservices --execute
"""
    )

    parser.add_argument(
        "--scenario",
        choices=["webapp", "microservices", "data-pipeline"],
        default="microservices",
        help="Scenario to demonstrate (default: microservices)"
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the plan simulation to see parallel execution in action"
    )

    args = parser.parse_args()

    # Get the plan
    plan = SAMPLE_PLANS[args.scenario]

    print("\n" + "=" * 70)
    print("🤖 LC AGENT - PLANNING DEMO")
    print("=" * 70)
    print(f"Scenario: {args.scenario}")
    print("Mode: Parallel Planning with Dependency Graphs")
    print("=" * 70)

    # Display the plan
    display_plan(plan)

    # Execute if requested
    if args.execute:
        print("\n")
        user_response = input("Execute this plan simulation? (yes/no): ").strip().lower()
        if user_response in ['yes', 'y']:
            simulator = ExecutionSimulator(plan)
            asyncio.run(simulator.simulate())
        else:
            print("Execution cancelled.")
    else:
        print("\n💡 Tip: Use --execute flag to see parallel execution in action!")
        print("   python demo-planning-cli.py --scenario {} --execute".format(args.scenario))


if __name__ == "__main__":
    main()
