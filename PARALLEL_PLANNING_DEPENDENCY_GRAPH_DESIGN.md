# Parallel Planning with Dependency Graphs: Design Document

**Author**: AI Research & Development Team
**Date**: 2025-01-16
**Status**: Design Phase - Ready for Implementation
**Related Module**: `source/modules/agents/planning`
**Integration Point**: `source/modules/aiq/lc_agent_aiq` (MultiAgent Supervisor)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Findings](#research-findings)
3. [Core Design Principles](#core-design-principles)
4. [Enhanced Plan Structure with Dependencies](#enhanced-plan-structure-with-dependencies)
5. [Dependency Graph Implementation](#dependency-graph-implementation)
6. [Parallel Execution Strategy](#parallel-execution-strategy)
7. [Integration with Planning Review Steps](#integration-with-planning-review-steps)
8. [Re-planning and Validation Points](#re-planning-and-validation-points)
9. [Implementation Specification](#implementation-specification)
10. [Testing Strategy](#testing-strategy)
11. [Future Enhancements](#future-enhancements)

---

## Executive Summary

### Problem Statement

The current planning system has several limitations:
1. **Sequential-only execution**: Steps execute one at a time, even when independent tasks could run in parallel
2. **No dependency tracking**: Cannot express which steps depend on others
3. **Inefficient resource use**: Wastes time executing tasks that could run concurrently
4. **Limited adaptability**: While Planning Review Steps enable re-planning, they lack dependency-aware replanning

### Solution: Dependency Graph-Based Parallel Planning

We propose enhancing the planning system with:

1. **Dependency Graphs**: Explicit dependency tracking between steps (inspired by claude-task-master)
2. **Parallel Execution**: Automatically execute independent tasks concurrently
3. **Planning Review Integration**: Combine with existing Planning Review Steps for adaptive replanning
4. **Dependency-Aware Replanning**: When replanning, maintain and update dependency relationships

### Key Benefits

- ✅ **Faster execution**: Independent tasks run in parallel
- ✅ **Explicit dependencies**: Clear prerequisite relationships
- ✅ **Automatic optimization**: Planner identifies parallelizable tasks
- ✅ **Adaptive replanning**: Review steps can replan with dependency awareness
- ✅ **Validation points**: Checkpoints verify dependencies are satisfied
- ✅ **Clean integration**: Works with existing MultiAgent supervisor and Task tool

---

## Research Findings

### 1. Claude-Task-Master Insights

**Dependency Structure:**
```json
{
  "id": 5,
  "title": "Deploy to production",
  "dependencies": [2, 3, 4],  // Must complete steps 2, 3, 4 first
  "status": "pending"
}
```

**Key Learnings:**
- **Simple array format**: Dependencies are just an array of prerequisite task IDs
- **Validation**: System prevents circular dependencies
- **Status tracking**: Dependencies show visual status (✅ completed, ⏱️ pending)
- **Next task selection**: Identifies tasks with all dependencies satisfied
- **Sequential execution**: Despite dependency tracking, claude-task-master executes sequentially

**What We'll Improve:**
- Add parallel execution when dependencies allow
- Integrate with Planning Review Steps for adaptive behavior
- Enable concurrent Task tool invocations

### 2. Planning Review Steps (from Research Branch)

**Key Insights:**
- **Explicit review steps**: Planning Agent inserts review steps in the plan
- **Strategic decisions**: CONTINUE, REPLAN, COMPLETE, ABORT
- **Replanning capability**: Can update remaining steps based on results
- **Self-correction**: Adapts when execution deviates from plan

**Integration Opportunity:**
- Planning Review Steps can trigger dependency graph updates
- Replanned steps can include new dependencies
- Review decisions can consider dependency satisfaction

### 3. Current Kit USD Agents System

**Parallel Execution Capability:**
- User mentioned: "Now that I can run concurrent Task"
- System supports multiple concurrent Task tool invocations
- MultiAgent supervisor can route to multiple agents simultaneously

**Current Planning System:**
- Sequential step execution via `plan_status` tracking
- Simple status: "pending" → "in_progress" → "completed"
- No dependency awareness
- No parallel execution support

---

## Core Design Principles

### Principle 1: Dependency-First Planning

**Planning Agent decides dependencies during plan creation**
- Not hard-coded rules or post-processing
- Dependencies are part of the plan structure
- Planning Agent considers:
  - Which steps require outputs from other steps
  - Which steps can run independently
  - Resource constraints (e.g., can't deploy before building)

### Principle 2: Automatic Parallelization

**System automatically identifies parallelizable steps**
- Steps with satisfied dependencies execute immediately
- Multiple independent steps launch concurrently
- No manual "parallel" flags needed
- Respects max concurrency limits

### Principle 3: Dependency-Aware Replanning

**Planning Review Steps update dependency graph**
- When replanning, new steps include dependencies
- Existing completed steps remain in the graph
- Dependency chains remain valid (no circular deps)
- Review steps can check dependency satisfaction

### Principle 4: Minimal Architectural Changes

**Build on existing infrastructure**
- Extend current plan structure (add "dependencies" field)
- Enhance PlanningModifier for dependency tracking
- Leverage existing Task tool parallel execution
- Integrate with Planning Review Steps from research branch

---

## Enhanced Plan Structure with Dependencies

### Current Structure (Sequential)

```python
{
    "title": "Deploy Application",
    "steps": [
        {
            "step_number": 1,
            "title": "Build application",
            "details": ["Compile source", "Run tests"]
        },
        {
            "step_number": 2,
            "title": "Deploy to staging",
            "details": ["Upload artifacts", "Configure"]
        }
    ]
}
```

### Enhanced Structure (With Dependencies)

```python
{
    "title": "Deploy Application",
    "steps": [
        {
            "step_number": 1,
            "title": "Build application",
            "step_type": "action",
            "dependencies": [],  # NEW: No prerequisites
            "details": ["Compile source", "Run tests"]
        },
        {
            "step_number": 2,
            "title": "Build documentation",
            "step_type": "action",
            "dependencies": [],  # NEW: Independent, can run parallel with step 1
            "details": ["Generate API docs", "Build user manual"]
        },
        {
            "step_number": 3,
            "title": "Planning Review - Build Verification",
            "step_type": "planning_review",
            "dependencies": [1, 2],  # NEW: Waits for both builds
            "review_context": {
                "review_focus": "Verify both builds completed successfully",
                "previous_steps": "Steps 1, 2",
                "decision_points": [
                    "Did application build succeed?",
                    "Is documentation generated correctly?"
                ]
            }
        },
        {
            "step_number": 4,
            "title": "Deploy to staging",
            "step_type": "action",
            "dependencies": [3],  # NEW: Requires review approval
            "details": ["Upload artifacts", "Configure environment"]
        },
        {
            "step_number": 5,
            "title": "Run integration tests",
            "step_type": "action",
            "dependencies": [4],  # NEW: Requires deployment
            "details": ["Execute test suite", "Verify functionality"]
        }
    ]
}
```

### Execution Flow with Parallelism

```
Timeline:
t=0:   Start step 1 (build app) and step 2 (build docs) in PARALLEL
       └─ Both have dependencies=[], so both start immediately

t=60s: Step 1 completes (build app)
t=75s: Step 2 completes (build docs)

t=76s: Start step 3 (Planning Review)
       └─ dependencies=[1,2] satisfied, review can run

t=80s: Step 3 completes with DECISION: CONTINUE

t=81s: Start step 4 (deploy to staging)
       └─ dependencies=[3] satisfied

t=120s: Step 4 completes

t=121s: Start step 5 (integration tests)
        └─ dependencies=[4] satisfied
```

**Time saved**: Steps 1 and 2 run in parallel, saving ~60-75 seconds compared to sequential execution.

### New Fields

**`dependencies`**: `List[int]`
- Array of step numbers that must complete before this step
- Empty array `[]` means no dependencies (can start immediately)
- Example: `"dependencies": [1, 3]` means "wait for steps 1 and 3"

**`step_type`**: `"action" | "planning_review"`
- Already introduced in Planning Review Steps design
- Determines whether step executes a tool or invokes Planning Agent for review

### Dependency Validation Rules

1. **No self-references**: Step cannot depend on itself
2. **No forward-only references**: Can only depend on earlier step numbers
3. **No circular dependencies**: A→B→C→A is invalid
4. **Dependencies must exist**: Referenced step numbers must exist in plan
5. **Review steps can depend on any prior steps**: Flexible checkpoint placement

---

## Dependency Graph Implementation

### Graph Structure

```python
class DependencyGraph:
    """
    Manages task dependencies and determines execution readiness.

    Attributes:
        steps: List of all plan steps
        adjacency_list: Map from step_number to list of dependent steps
        in_degree: Map from step_number to count of unsatisfied dependencies
        completed: Set of completed step numbers
    """

    def __init__(self, plan_steps: List[Dict]):
        """Build dependency graph from plan steps."""
        self.steps = {step["step_number"]: step for step in plan_steps}
        self.adjacency_list = {}  # step_num → [steps that depend on it]
        self.in_degree = {}  # step_num → count of unsatisfied dependencies
        self.completed = set()

        # Build graph
        for step in plan_steps:
            step_num = step["step_number"]
            dependencies = step.get("dependencies", [])

            # Initialize adjacency list
            if step_num not in self.adjacency_list:
                self.adjacency_list[step_num] = []

            # Set in-degree (number of dependencies)
            self.in_degree[step_num] = len(dependencies)

            # Add edges from dependencies to this step
            for dep in dependencies:
                if dep not in self.adjacency_list:
                    self.adjacency_list[dep] = []
                self.adjacency_list[dep].append(step_num)

    def get_ready_steps(self) -> List[int]:
        """
        Get all steps ready to execute (dependencies satisfied, not completed).

        Returns:
            List of step numbers ready for execution
        """
        ready = []
        for step_num, degree in self.in_degree.items():
            if degree == 0 and step_num not in self.completed:
                ready.append(step_num)
        return sorted(ready)

    def mark_completed(self, step_num: int):
        """
        Mark a step as completed and update dependent steps.

        Args:
            step_num: The step number that completed
        """
        if step_num in self.completed:
            return

        self.completed.add(step_num)

        # Reduce in-degree for all dependent steps
        for dependent in self.adjacency_list.get(step_num, []):
            self.in_degree[dependent] -= 1

    def validate_dependencies(self) -> tuple[bool, Optional[str]]:
        """
        Validate dependency graph for correctness.

        Returns:
            (is_valid, error_message)
        """
        # Check for non-existent dependencies
        for step in self.steps.values():
            for dep in step.get("dependencies", []):
                if dep not in self.steps:
                    return False, f"Step {step['step_number']} depends on non-existent step {dep}"
                if dep >= step["step_number"]:
                    return False, f"Step {step['step_number']} has forward/self dependency on step {dep}"

        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for dependent in self.adjacency_list.get(node, []):
                if dependent not in visited:
                    if has_cycle(dependent):
                        return True
                elif dependent in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_num in self.steps:
            if step_num not in visited:
                if has_cycle(step_num):
                    return False, "Circular dependency detected"

        return True, None

    def get_dependency_status(self, step_num: int) -> Dict[str, Any]:
        """
        Get detailed dependency status for a step.

        Returns:
            {
                "ready": bool,
                "dependencies": List[int],
                "unsatisfied": List[int],
                "satisfied": List[int]
            }
        """
        step = self.steps.get(step_num)
        if not step:
            return {"ready": False, "dependencies": [], "unsatisfied": [], "satisfied": []}

        dependencies = step.get("dependencies", [])
        satisfied = [d for d in dependencies if d in self.completed]
        unsatisfied = [d for d in dependencies if d not in self.completed]

        return {
            "ready": len(unsatisfied) == 0,
            "dependencies": dependencies,
            "satisfied": satisfied,
            "unsatisfied": unsatisfied
        }
```

### Integration with PlanningModifier

```python
class PlanningModifier(NetworkModifier):
    """Enhanced with dependency graph support."""

    def __init__(self):
        super().__init__()
        self.current_plan = None
        self.plan_status = {}
        self.dependency_graph = None  # NEW
        self.max_parallel_steps = 5  # NEW: Configurable concurrency limit

    def _extract_plan(self, content: str) -> Dict[str, Any]:
        """Enhanced to extract dependencies from plan."""
        # ... existing extraction logic ...

        # Extract dependencies for each step
        for step in steps:
            step_number = step["step_number"]
            step_content = step["_raw_content"]  # Keep raw content for dependency extraction

            # Look for dependency declarations
            # Format: "Dependencies: 1, 2, 3" or "Depends on: Steps 1, 2"
            dep_match = re.search(
                r"(?:Dependencies|Depends on):?\s*(?:Steps?\s*)?(\d+(?:\s*,\s*\d+)*)",
                step_content,
                re.IGNORECASE
            )

            if dep_match:
                dep_str = dep_match.group(1)
                dependencies = [int(d.strip()) for d in dep_str.split(",")]
            else:
                dependencies = []

            step["dependencies"] = dependencies

        return {"title": plan_title, "steps": steps}

    async def on_post_invoke_async(self, network, node):
        """Enhanced to initialize dependency graph."""
        # ... existing plan generation logic ...

        if self._is_valid_plan(plan_content):
            self.current_plan = self._extract_plan(plan_content)

            # NEW: Initialize dependency graph
            self.dependency_graph = DependencyGraph(self.current_plan["steps"])

            # Validate dependency graph
            is_valid, error = self.dependency_graph.validate_dependencies()
            if not is_valid:
                logger.error(f"Invalid dependency graph: {error}")
                # Plan has invalid dependencies, notify Planning Agent to fix
                # Could trigger automatic re-planning here
                return

            # ... share plan with multi-agent network ...

            # Initialize plan status
            self.plan_status = {
                step["step_number"]: "pending"
                for step in self.current_plan["steps"]
            }

    async def on_pre_invoke_async(self, network, node):
        """Enhanced to launch multiple ready steps in parallel."""
        from ..nodes.planning_node import PlanningGenNode, PlanningNetworkNode

        if (
            isinstance(network, MultiAgentNetworkNode)
            and type(node) == get_node_factory().get_registered_node_type(network.default_node)
            and not network.get_children(node)
        ):
            # Get all ready steps from dependency graph
            ready_steps = self.dependency_graph.get_ready_steps()

            if not ready_steps:
                # No steps ready, check if plan is complete
                if len(self.dependency_graph.completed) == len(self.current_plan["steps"]):
                    # All steps completed
                    logger.info("All plan steps completed")
                return

            # Determine how many steps to launch in parallel
            currently_in_progress = sum(
                1 for status in self.plan_status.values()
                if status == "in_progress"
            )
            available_slots = self.max_parallel_steps - currently_in_progress

            if available_slots <= 0:
                # Already at max parallelism, wait for some to complete
                return

            # Launch up to available_slots ready steps
            steps_to_launch = ready_steps[:available_slots]

            # NEW: Launch multiple steps concurrently using Task tool
            for step_num in steps_to_launch:
                step = next(
                    s for s in self.current_plan["steps"]
                    if s["step_number"] == step_num
                )

                if step.get("step_type") == "planning_review":
                    # Planning Review step
                    await self._inject_planning_review_step(
                        network, node, step_num, step
                    )
                else:
                    # Regular action step
                    # Mark as in_progress before launching
                    self.plan_status[step_num] = "in_progress"

                    # Inject step instruction
                    # The MultiAgent will route to appropriate tool
                    # Multiple steps can be in_progress simultaneously
                    follow_the_plan_message = self._build_step_instruction_message(
                        step_num, step
                    )

                    with network:
                        follow_the_plan_node = RunnableHumanNode(
                            human_message=follow_the_plan_message
                        )
                        follow_the_plan_node.parents.clear()
                        follow_the_plan_node.metadata["plan_step_number"] = step_num
                        node._add_parent(follow_the_plan_node)
```

---

## Parallel Execution Strategy

### Execution Model

```
+------------------+
| Dependency Graph |
+------------------+
        |
        | get_ready_steps()
        v
+------------------+
| Ready Queue      |  ← Steps with dependencies satisfied
| [1, 2, 3]        |
+------------------+
        |
        | Launch up to max_parallel_steps
        v
+------------------+
| In-Progress Pool |  ← Currently executing steps
| {1: Task, 2: Task}|
+------------------+
        |
        | on_complete → mark_completed()
        v
+------------------+
| Dependency Graph |  ← Update in-degrees
+------------------+
        |
        | get_ready_steps() → new ready steps
        v
   ... cycle continues ...
```

### Concurrency Control

**Max Parallel Steps**: `max_parallel_steps = 5` (configurable)
- Prevents overwhelming the system with too many concurrent tasks
- Can be adjusted based on:
  - Available resources (CPU, memory)
  - API rate limits (for LLM calls)
  - User preferences

**Task Launching**:
- When `in_progress_count < max_parallel_steps`, launch more tasks
- Always respect dependency constraints
- Prioritize by step number when multiple steps are ready

### Example: Complex Parallel Plan

```python
{
    "title": "Build and Deploy Microservices Application",
    "steps": [
        # Phase 1: Independent builds (parallel)
        {
            "step_number": 1,
            "title": "Build auth-service",
            "dependencies": [],
            "details": ["Compile Go code", "Run unit tests"]
        },
        {
            "step_number": 2,
            "title": "Build user-service",
            "dependencies": [],
            "details": ["Compile Python code", "Run pytest"]
        },
        {
            "step_number": 3,
            "title": "Build api-gateway",
            "dependencies": [],
            "details": ["Compile Node.js", "Run Jest tests"]
        },
        {
            "step_number": 4,
            "title": "Build database migrations",
            "dependencies": [],
            "details": ["Generate migration scripts", "Validate SQL"]
        },

        # Phase 2: Review builds
        {
            "step_number": 5,
            "title": "Planning Review - Build Verification",
            "step_type": "planning_review",
            "dependencies": [1, 2, 3, 4],  # Wait for all builds
            "review_context": {
                "review_focus": "Verify all services built successfully",
                "decision_points": [
                    "Did all builds pass?",
                    "Are all tests green?",
                    "Ready for deployment?"
                ]
            }
        },

        # Phase 3: Deploy database first
        {
            "step_number": 6,
            "title": "Deploy database and run migrations",
            "dependencies": [5],  # After review approval
            "details": ["Provision PostgreSQL", "Apply migrations"]
        },

        # Phase 4: Deploy services in parallel (after DB ready)
        {
            "step_number": 7,
            "title": "Deploy auth-service to staging",
            "dependencies": [6],  # Needs database
            "details": ["Create k8s deployment", "Configure secrets"]
        },
        {
            "step_number": 8,
            "title": "Deploy user-service to staging",
            "dependencies": [6],  # Needs database
            "details": ["Create k8s deployment", "Configure secrets"]
        },
        {
            "step_number": 9,
            "title": "Deploy api-gateway to staging",
            "dependencies": [6],  # Needs database
            "details": ["Create k8s deployment", "Configure routes"]
        },

        # Phase 5: Integration tests (after all services deployed)
        {
            "step_number": 10,
            "title": "Run integration tests",
            "dependencies": [7, 8, 9],  # All services must be up
            "details": ["Execute test suite", "Verify API endpoints"]
        },

        # Phase 6: Final review
        {
            "step_number": 11,
            "title": "Planning Review - Deployment Validation",
            "step_type": "planning_review",
            "dependencies": [10],
            "review_context": {
                "review_focus": "Assess deployment health",
                "decision_points": [
                    "Are all services healthy?",
                    "Did integration tests pass?",
                    "Ready for production?"
                ]
            }
        }
    ]
}
```

**Execution Timeline**:

```
t=0s:    Launch steps 1, 2, 3, 4 in PARALLEL (all have dependencies=[])
         [Build auth, user, api-gateway, db-migrations concurrently]

t=45s:   Step 1 completes (auth-service built)
t=50s:   Step 2 completes (user-service built)
t=52s:   Step 3 completes (api-gateway built)
t=55s:   Step 4 completes (db-migrations ready)

t=56s:   Launch step 5 (Planning Review - Build Verification)
         dependencies=[1,2,3,4] all satisfied

t=60s:   Step 5 completes → DECISION: CONTINUE

t=61s:   Launch step 6 (Deploy database)
         dependencies=[5] satisfied

t=90s:   Step 6 completes (database deployed)

t=91s:   Launch steps 7, 8, 9 in PARALLEL
         [Deploy auth, user, api-gateway services concurrently]
         dependencies=[6] satisfied for all

t=120s:  Step 7 completes (auth-service deployed)
t=125s:  Step 8 completes (user-service deployed)
t=130s:  Step 9 completes (api-gateway deployed)

t=131s:  Launch step 10 (Integration tests)
         dependencies=[7,8,9] all satisfied

t=160s:  Step 10 completes (tests passed)

t=161s:  Launch step 11 (Planning Review - Final)
         dependencies=[10] satisfied

t=165s:  Step 11 completes → DECISION: COMPLETE

Total time: 165 seconds

Sequential execution would have taken:
(45+50+52+55) + 60 + (90-61) + (120+125+130) + (160-131) + (165-161)
= 202s + 60s + 29s + 375s + 29s + 4s = 699s

Time saved: 534 seconds (76% faster!)
```

---

## Integration with Planning Review Steps

### Combined Architecture

Planning Review Steps + Dependency Graphs = Adaptive Parallel Planning

**Synergies:**

1. **Review steps validate dependency satisfaction**
   - Before proceeding, review checks if prerequisites truly succeeded
   - Example: "Did the database deploy correctly before deploying services?"

2. **Replanning updates dependency graph**
   - When review decides to REPLAN, new steps include dependencies
   - Dependency graph is rebuilt with updated steps
   - Parallel execution continues with new graph

3. **Review steps act as synchronization barriers**
   - Multiple parallel steps → Review → More parallel steps
   - Enables "phases" in execution: Build → Review → Deploy → Review

### Enhanced Review Context

```python
{
    "step_number": 5,
    "title": "Planning Review - Build Verification",
    "step_type": "planning_review",
    "dependencies": [1, 2, 3, 4],  # NEW: Explicit dependencies
    "review_context": {
        "review_focus": "Verify all parallel builds succeeded",
        "previous_steps": "Steps 1-4",  # Human-readable
        "dependency_status": {  # NEW: Detailed dependency info
            "1": {"satisfied": True, "result": "✅ auth-service built"},
            "2": {"satisfied": True, "result": "✅ user-service built"},
            "3": {"satisfied": True, "result": "✅ api-gateway built"},
            "4": {"satisfied": True, "result": "✅ db-migrations ready"}
        },
        "decision_points": [
            "Did all 4 builds complete successfully?",
            "Are all tests passing?",
            "Any build errors to address?"
        ],
        "potential_outcomes": [
            "CONTINUE - Proceed to database deployment",
            "REPLAN - Fix failed builds and rebuild",
            "ABORT - Critical build failures"
        ]
    }
}
```

### Replanning with Dependency Updates

When Planning Review decides to REPLAN, it can:

1. **Add new steps with dependencies**
2. **Modify remaining step dependencies**
3. **Insert new parallel branches**

**Example Replanning Scenario:**

Original plan (steps 6-10 remaining):
```
Step 6: Deploy all services  [dependencies: [5]]
Step 7: Run tests  [dependencies: [6]]
```

Review at step 5 discovers: "Service A has compatibility issue with new library"

Replanned (new steps 6-12):
```
Step 6: Fix compatibility in Service A  [dependencies: [5]]
Step 7: Rebuild Service A  [dependencies: [6]]
Step 8: Deploy Service B (independent)  [dependencies: [5]]  ← Parallel!
Step 9: Deploy Service C (independent)  [dependencies: [5]]  ← Parallel!
Step 10: Planning Review - Partial Deployment Check  [dependencies: [8, 9]]
Step 11: Deploy Service A  [dependencies: [7]]
Step 12: Run integration tests  [dependencies: [10, 11]]
```

**Benefits:**
- Services B and C deploy in parallel while A is being fixed
- Review step 10 checks B and C before deploying A
- Dependency graph automatically manages execution order

---

## Re-planning and Validation Points

### Strategic Validation Points

**Validation points** are Planning Review steps strategically placed to:
1. **Verify dependency satisfaction**: Check prerequisites truly succeeded
2. **Assess parallel execution results**: Ensure all concurrent tasks completed correctly
3. **Trigger re-planning if needed**: Adapt when things don't go as planned
4. **Provide synchronization barriers**: Coordinate between parallel phases

### When to Insert Validation Points

The Planning Agent should insert Planning Review steps:

1. **After parallel phases**
   - Example: After 4 parallel builds, review before deploying
   - Ensures all parallel tasks succeeded before proceeding

2. **Before critical operations**
   - Example: Review before production deployment
   - Verify all prerequisites (builds, tests, staging validation)

3. **After dependency-heavy operations**
   - Example: After deploying database, review before deploying services
   - Confirm foundational dependencies are stable

4. **At phase transitions**
   - Example: Development → Staging → Production
   - Validate state before moving to next phase

### Enhanced System Prompt for Dependency-Aware Planning

```markdown
## Planning with Dependencies and Parallelism

You are now planning with support for **parallel execution** and **dependency graphs**.

### Expressing Dependencies

For each step, specify which previous steps must complete first:

```
Step N: <title>
Dependencies: <comma-separated step numbers> or "None" if no dependencies
- <detail 1>
- <detail 2>
```

**Examples:**

```
Step 1: Build authentication service
Dependencies: None
- Compile Go source code
- Run unit tests
- Build Docker image

Step 2: Build user service
Dependencies: None
- Compile Python source code
- Run pytest suite
- Build Docker image

Step 3: Planning Review - Build Verification
Dependencies: 1, 2
- Review focus: Verify both services built successfully
- Decision points:
  * Did both builds pass all tests?
  * Are Docker images created correctly?
- Potential outcomes:
  * CONTINUE - Proceed to deployment
  * REPLAN - Fix build issues

Step 4: Deploy both services to staging
Dependencies: 3
- Deploy auth-service to k8s
- Deploy user-service to k8s
- Configure load balancer
```

### Guidelines for Dependencies

1. **Independent tasks**: Set `Dependencies: None` to enable parallel execution
   - Example: Building multiple services can happen in parallel

2. **Sequential dependencies**: Reference prerequisite steps
   - Example: Deployment depends on build completion

3. **Multiple dependencies**: List all required prerequisite steps
   - Example: Integration tests depend on all services being deployed

4. **Review step dependencies**: Reviews should depend on steps they're assessing
   - Example: Build verification review depends on all build steps

### Planning for Parallelism

**Identify opportunities for parallelism:**
- Multiple independent builds → Run in parallel
- Multiple service deployments (after shared prerequisite) → Run in parallel
- Multiple test suites (if independent) → Run in parallel

**Structure with phases:**
```
Phase 1: Parallel builds (Steps 1, 2, 3 - Dependencies: None)
Phase 2: Build review (Step 4 - Dependencies: 1, 2, 3)
Phase 3: Shared setup (Step 5 - Dependencies: 4)
Phase 4: Parallel deploys (Steps 6, 7, 8 - Dependencies: 5)
Phase 5: Deployment review (Step 9 - Dependencies: 6, 7, 8)
```

### Validation Points (Planning Review Steps)

Insert Planning Review steps to:
- **Synchronize after parallel operations**: Wait for all parallel tasks to complete
- **Validate before critical operations**: Review state before risky operations
- **Enable adaptive replanning**: Adjust plan based on results

**Best practices:**
- Review after completing parallel builds/deploys
- Review before production deployments
- Review after tests to decide next steps

### Example: Complete Plan with Dependencies

```
PLAN: Deploy Microservices Application

Step 1: Build service A
Dependencies: None
- Compile code
- Run tests

Step 2: Build service B
Dependencies: None
- Compile code
- Run tests

Step 3: Build service C
Dependencies: None
- Compile code
- Run tests

Step 4: Planning Review - Build Verification
Dependencies: 1, 2, 3
- Review focus: Verify all builds succeeded
- Decision points:
  * All builds passed?
  * All tests green?
- Potential outcomes:
  * CONTINUE - Proceed to deployment
  * REPLAN - Fix failing builds

Step 5: Setup database
Dependencies: 4
- Provision PostgreSQL
- Run migrations

Step 6: Deploy service A
Dependencies: 5
- Deploy to k8s

Step 7: Deploy service B
Dependencies: 5
- Deploy to k8s

Step 8: Deploy service C
Dependencies: 5
- Deploy to k8s

Step 9: Planning Review - Deployment Verification
Dependencies: 6, 7, 8
- Review focus: Verify all services deployed
- Decision points:
  * All pods healthy?
  * Services accessible?
- Potential outcomes:
  * CONTINUE - Run integration tests
  * REPLAN - Fix deployment issues

Step 10: Run integration tests
Dependencies: 9
- Execute test suite
- Verify API endpoints
```

**Execution pattern:**
- Steps 1, 2, 3 run in PARALLEL
- Step 4 waits for all three, then reviews
- Step 5 runs after review approval
- Steps 6, 7, 8 run in PARALLEL (after step 5)
- Step 9 waits for all three deployments
- Step 10 runs after final review
```

---

## Implementation Specification

### File Changes Required

#### 1. `planning_gen_system.md`
**Add section on dependencies and parallel execution**
- Teach Planning Agent how to express dependencies
- Examples of parallel-friendly plans
- Guidelines for phase-based planning with reviews

#### 2. `planning_modifier.py`

**A. Add DependencyGraph class** (~150 lines)
- `__init__`: Build graph from plan steps
- `get_ready_steps()`: Return steps with satisfied dependencies
- `mark_completed()`: Update graph when step completes
- `validate_dependencies()`: Check for circular deps, invalid refs
- `get_dependency_status()`: Detailed status for a step

**B. Enhance PlanningModifier class**
- Add `self.dependency_graph: DependencyGraph`
- Add `self.max_parallel_steps: int`
- Update `_extract_plan()`: Parse dependencies from step content
- Update `on_post_invoke_async()`: Initialize dependency graph, validate
- Update `on_pre_invoke_async()`: Launch multiple ready steps in parallel
- Add `_mark_step_completed()`: Update dependency graph on completion
- Update `_replace_remaining_steps()`: Rebuild dependency graph after replan

**C. Integration with Planning Review Steps**
- Enhance `_build_review_prompt()`: Include dependency status
- Update `_process_planning_review_decision()`: Handle dependency updates in REPLAN

**Total additions**: ~400 lines

#### 3. `planning_node.py`
**No changes required** - Works as-is

#### 4. Integration with Task Tool

**Prerequisite**: User mentioned "Now that I can run concurrent Task"
- Verify Task tool supports concurrent invocations
- Ensure MultiAgent supervisor can launch multiple tasks simultaneously
- Test that results from parallel tasks are properly captured

### Configuration Options

```python
# In PlanningModifier or config
PLANNING_CONFIG = {
    "max_parallel_steps": 5,  # Max concurrent step execution
    "enable_dependency_validation": True,  # Validate dependency graph
    "auto_parallelize": True,  # Automatically execute independent steps in parallel
    "dependency_timeout_seconds": 3600,  # Timeout for waiting on dependencies
}
```

---

## Testing Strategy

### Unit Tests

**Test File**: `test_dependency_graph.py`

1. **test_dependency_graph_construction**
   - Build graph from plan with dependencies
   - Verify adjacency list and in-degrees

2. **test_get_ready_steps**
   - Graph with 5 steps, various dependencies
   - Initially: steps with dependencies=[] are ready
   - After marking step 1 complete: steps depending only on 1 become ready

3. **test_mark_completed**
   - Mark step complete, verify dependent steps' in-degrees decrease

4. **test_validate_dependencies_valid**
   - Valid graph passes validation

5. **test_validate_dependencies_circular**
   - Detect circular dependency: 1→2→3→1

6. **test_validate_dependencies_forward_ref**
   - Detect invalid forward reference: step 2 depends on step 5

7. **test_validate_dependencies_nonexistent**
   - Detect dependency on non-existent step

8. **test_get_dependency_status**
   - Get detailed status for a step
   - Verify satisfied/unsatisfied lists

### Integration Tests

**Test File**: `test_parallel_planning.py`

1. **test_parallel_execution_basic**
   - Plan with 2 independent steps
   - Verify both launch simultaneously
   - Verify both complete before dependent step 3 starts

2. **test_parallel_execution_phases**
   - Plan with build phase (3 parallel) → review → deploy phase (3 parallel)
   - Verify correct execution order
   - Verify review waits for all builds

3. **test_replan_with_dependencies**
   - Execute plan, review triggers REPLAN
   - Verify new steps include dependencies
   - Verify dependency graph rebuilt correctly
   - Verify parallel execution continues

4. **test_max_parallel_limit**
   - Plan with 10 independent steps, max_parallel=3
   - Verify only 3 execute at a time
   - Verify next steps launch as others complete

5. **test_dependency_failure_handling**
   - Step 2 depends on step 1
   - Step 1 fails
   - Verify step 2 never starts (dependency not satisfied)

### Manual Testing Scenarios

**Scenario 1: Parallel Microservice Deployment**
- User: "Deploy auth, user, and api services to staging"
- Expected: Plan with 3 parallel builds, review, 3 parallel deploys
- Verify: Builds run in parallel, deploys run in parallel

**Scenario 2: Complex Data Pipeline**
- User: "Process customer data: extract, transform, load"
- Expected: Multiple extractors run in parallel, transform waits for all, load runs after
- Verify: Dependency graph correctly sequences operations

**Scenario 3: Replanning After Parallel Failure**
- Plan: 3 parallel builds → review
- One build fails during execution
- Review: REPLAN to fix failed build
- Verify: New plan includes fixing step, other builds not re-run

---

## Future Enhancements

### 1. Dynamic Dependency Injection

**Concept**: Steps can declare runtime dependencies based on execution results

```python
{
    "step_number": 5,
    "title": "Run tests",
    "dependencies": [4],
    "dynamic_dependencies": {
        "if_condition": "step_4_result.contains('database')",
        "add_dependency": 3  # Also depend on database setup if detected
    }
}
```

### 2. Priority-Based Scheduling

**Concept**: Steps have priority levels for execution ordering

```python
{
    "step_number": 2,
    "title": "Deploy critical service",
    "dependencies": [1],
    "priority": "high",  # Execute before other ready steps
}
```

### 3. Resource-Aware Scheduling

**Concept**: Steps declare resource requirements

```python
{
    "step_number": 3,
    "title": "Train ML model",
    "dependencies": [1, 2],
    "resources": {
        "gpu": 1,
        "memory_gb": 16,
        "estimated_duration_minutes": 120
    }
}
```

Scheduler considers:
- Available resources
- Resource contention
- Optimal packing

### 4. Conditional Steps

**Concept**: Steps that only execute if conditions are met

```python
{
    "step_number": 8,
    "title": "Rollback deployment",
    "dependencies": [7],
    "condition": "step_7_result.success == False",  # Only run if step 7 failed
}
```

### 5. Dependency Visualization

**Concept**: Generate visual dependency graph for user

```
Planning Agent generates Mermaid diagram:

graph TD
    1[Build Auth] --> 4[Review Builds]
    2[Build User] --> 4
    3[Build API] --> 4
    4 --> 5[Deploy DB]
    5 --> 6[Deploy Auth]
    5 --> 7[Deploy User]
    5 --> 8[Deploy API]
    6 --> 9[Integration Tests]
    7 --> 9
    8 --> 9
```

### 6. Learning from Execution Patterns

**Concept**: Analyze execution history to improve dependency predictions

- Track actual execution times
- Identify frequently parallel steps
- Suggest better dependency structures
- Optimize max_parallel based on resource usage

---

## Conclusion

**Parallel Planning with Dependency Graphs** transforms the planning system into a high-performance, adaptive execution engine that:

- **Maximizes parallelism**: Executes independent tasks concurrently
- **Manages complexity**: Dependency graphs handle intricate task relationships
- **Adapts dynamically**: Planning Review Steps enable intelligent replanning
- **Validates execution**: Strategic checkpoints ensure correctness
- **Integrates cleanly**: Builds on existing infrastructure

### Implementation Summary

| Component | Lines Added | Complexity | Priority |
|-----------|-------------|------------|----------|
| `DependencyGraph` class | ~150 | Medium | CRITICAL |
| `planning_modifier.py` enhancements | ~400 | High | CRITICAL |
| `planning_gen_system.md` updates | ~200 | Low | HIGH |
| Integration tests | ~300 | Medium | HIGH |
| **TOTAL** | **~1050** | - | - |

### Expected Impact

**Performance Gains:**
- 50-80% faster execution for plans with parallel-friendly steps
- Example: 4 parallel builds save 75% of build time

**Quality Improvements:**
- Explicit dependencies reduce errors
- Validation points catch issues early
- Adaptive replanning recovers from failures

**Developer Experience:**
- Clear dependency visualization
- Automatic parallelization (no manual tuning)
- Transparent execution progress

---

**Ready for implementation! Let's build the future of intelligent, parallel planning.**
