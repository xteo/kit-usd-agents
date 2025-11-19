# Planning Review Steps: Design Document

**Author**: AI Research & Development Team
**Date**: 2025-01-16
**Status**: Design Complete - Ready for Implementation
**Related Module**: `source/modules/agents/planning`
**Integration Point**: `source/modules/aiq/lc_agent_aiq` (MultiAgent Supervisor)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Initial Research Objectives](#initial-research-objectives)
3. [Industry Research Findings](#industry-research-findings)
4. [Architecture Analysis](#architecture-analysis)
5. [Design Decision: Planning Review Steps](#design-decision-planning-review-steps)
6. [Detailed Implementation Specification](#detailed-implementation-specification)
7. [File-by-File Changes](#file-by-file-changes)
8. [Integration with MultiAgent Supervisor](#integration-with-multiagent-supervisor)
9. [Testing Strategy](#testing-strategy)
10. [Future Enhancements](#future-enhancements)

---

## Executive Summary

### Problem Statement
The current planning system (`source/modules/agents/planning`) generates sequential plans but lacks:
- **Adaptive replanning** when execution deviates from plan
- **Strategic checkpoints** for reflection and decision-making
- **Self-correction** mechanisms when steps fail or produce unexpected results

### Solution: Planning Review Steps
Instead of implicit checkpoints or metadata flags, we implement **explicit Planning Review Steps** where the Planning Agent:
1. **Decides during planning** where review/reflection is needed
2. **Inserts review steps** directly into the plan as first-class steps
3. **Evaluates progress** at these checkpoints
4. **Makes strategic decisions**: CONTINUE, REPLAN, COMPLETE, or ABORT
5. **Adjusts the plan dynamically** based on execution results

### Key Benefits
- ✅ **Programmatic**: Reviews are explicit steps, not hidden metadata
- ✅ **Intelligent**: Planning Agent decides when reviews are needed
- ✅ **Self-correcting**: Can replan when things go wrong
- ✅ **Transparent**: Review steps visible in execution logs
- ✅ **Clean integration**: Works seamlessly with existing MultiAgent supervisor

---

## Initial Research Objectives

### Original Questions

**Primary Question:**
> "Do a deep research on agent planning. How do other frameworks like LangChain, LangGraph, CrewAI, and AutoGPT handle planning? Focus on:
> 1. How to express tasks in plans that can run in parallel
> 2. Checkpoint mechanisms for replanning and reevaluation
> 3. Force checkpoints for thinking and reflection"

### Research Scope

We conducted comprehensive research on:
1. **LangGraph**: State-of-the-art graph-based agent orchestration
2. **LangChain Plan-and-Execute**: Explicit planning and replanning patterns
3. **CrewAI**: Multi-agent task execution and planning
4. **AutoGPT**: Autonomous planning with self-refinement
5. **Industry patterns**: DAG dependencies, scatter-gather, checkpointing strategies

---

## Industry Research Findings

### 1. LangGraph (State-of-the-Art)

**Architecture:**
- **DAG-based orchestration**: Nodes = agents/functions, Edges = data flow
- **StateGraph**: Centralized state management for parallel execution
- **Conditional branching**: Dynamic routing based on state

**Parallel Execution:**
- Tasks with satisfied dependencies execute concurrently
- Scatter-gather pattern: Distribute → Execute in parallel → Aggregate results
- Pipeline parallelism: Sequential stages run concurrently

**Checkpointing:**
- **PostgresSaver**: Persistent checkpoint storage
- **Time-travel**: Restore to any previous checkpoint
- **Pause/resume**: Can resume on any machine after arbitrary time
- **Human-in-the-loop**: Checkpoints before critical actions

**Key Insight:**
> "Structured execution makes checkpointing feasible, enabling snapshots that can be resumed anywhere, anytime"

### 2. LangChain Plan-and-Execute

**Architecture:**
- **Planner LLM**: Generates initial plan
- **Executor**: Executes sub-tasks
- **Replanner**: Dedicated node for dynamic replanning
- **Joiner**: Decides whether to finish or replan

**Replanning Strategy:**
```
Execute Step → Evaluate Result → Decision:
  ├─ Success & Complete → Return final answer
  ├─ Success & More Work → Continue with plan
  └─ Failure or Deviation → Invoke Replanner
                           └─ Replanner receives:
                              - Original objective
                              - Original plan
                              - Completed steps + results
                              - Failed step + error
                           └─ Outputs: Updated plan
```

**Replanning Inputs:**
1. Original objective (the goal)
2. Original plan (what was intended)
3. Completed steps (what succeeded)
4. Execution results (what actually happened)
5. Current context (current state)

**Key Insight:**
> "Replanning is not just error recovery—it's continuous plan validation against reality"

### 3. CrewAI

**Architecture:**
- **Task Execution Planner**: Dedicated agent for step-by-step planning
- **Async execution**: `async_execution=True` on tasks
- **Process modes**: Sequential, Parallel, Hierarchical

**Parallel Execution:**
- `asyncio.gather()` for concurrent task execution
- Tasks execute in parallel when dependencies allow
- Results aggregated before next phase

**Planning Features:**
- Planning agent creates execution plan
- Tasks distributed to specialized agents
- Hierarchical process for complex workflows

**Key Insight:**
> "Planning should be a dedicated responsibility, not mixed with execution"

### 4. AutoGPT

**Architecture:**
- **Planning loop**: Think → Plan → Execute → Reflect → Replan
- **Self-refinement**: Agent evaluates its own performance
- **Feedback integration**: Execution results inform next planning cycle

**Checkpoint & Reflection:**
- **Semantic checkpoints**: Agent reflects before executing
- **Internal reflection**: Diagnoses failures and updates strategy
- **Self-continuous thinking**: Cycle of planning, execution, feedback

**Replanning Mechanism:**
```
If stalls or unexpected results:
  1. Pause execution
  2. Run internal reflection process
  3. Diagnose failure points
  4. Update strategy
  5. Resume with adjusted plan
```

**Key Insight:**
> "Effective agents don't just execute plans—they continuously reflect and adapt"

### 5. Common Industry Patterns

**Pattern 1: DAG (Directed Acyclic Graph)**
- **Purpose**: Model task dependencies
- **Benefit**: Identify parallelizable tasks automatically
- **Usage**: Tasks with zero dependency count execute concurrently
- **Acyclic requirement**: Prevents deadlocks and infinite loops

**Pattern 2: Scatter-Gather**
```
Root receives request
  ↓
Scatter: Divide into independent tasks
  ↓
Parallel: Each worker processes its task
  ↓
Gather: Collect and aggregate results
  ↓
Return: Synthesized response
```

**Pattern 3: Checkpoint Strategies**
- **Before high-impact actions**: Fund transfers, production deployments
- **After expensive operations**: Model training, data processing
- **At dependency boundaries**: Between major phases
- **Periodic**: Every N steps in long-running tasks

**Pattern 4: Trajectory Evaluation**
- **Track decision sequences**: Record each step and its outcome
- **Detect deviations**: Compare actual vs. expected path
- **Trigger corrections**: Replan when trajectory diverges
- **Learn from patterns**: Improve future planning

---

## Architecture Analysis

### Current System Overview

**Component Hierarchy:**
```
AIQ Request
  ↓
LCAgentFunction (Bridge: AIQ ↔ LC Agent)
  ↓
MultiAgentNetworkNode (Supervisor)
  ├─ Registers route_nodes as tools
  ├─ Binds tools to chat model
  ├─ ToolModifier: Handles routing and tool calls
  └─ Creates RunnableSupervisorNode for decisions
  ↓
When "planning" is a route_node:
  ↓
PlanningNetworkNode (Tool)
  ├─ PlanningGenNode: Generates plan
  └─ PlanningModifier: Manages execution
      ├─ Captures generated plan
      ├─ Injects into MultiAgentNetworkNode
      ├─ Guides step-by-step execution
      └─ Tracks plan_status
```

### Key Files and Responsibilities

**1. `source/modules/aiq/lc_agent_aiq/lc_agent_aiq/`**

**`multi_agent_register.py`**
- Registers `MultiAgentNetworkNode` with AIQ
- Configuration: `MultiAgentConfig`
  - `tool_names`: List of agents/tools to route to
  - `system_message`: Supervisor instructions
  - `multishot`: Multi-step conversation support
  - `function_calling`: Use LLM tool calling vs. classification

**`utils/multi_agent_network_function.py`**
- `MultiAgentNetworkFunction`: Specialized LCAgentFunction
- Pre-invoke: Registers each tool with node factory
- Post-invoke: Unregisters tools (cleanup)
- Routes: `route_nodes` parameter specifies available tools

**`utils/lc_agent_function.py`**
- `LCAgentFunction`: Base class for all LC Agent integrations
- Bridges AIQ (Function) ↔ LC Agent (NetworkNode/RunnableNode)
- Lifecycle: `pre_invoke()` → `ainvoke()` / `astream()` → `post_invoke()`
- Message conversion: AIQ ↔ LangChain formats
- Network setup: Creates RunnableNetwork with message history

**2. `source/modules/lc_agent/src/lc_agent/`**

**`multi_agent_network_node.py`**
- `MultiAgentNetworkNode`: The supervisor that routes to agents/tools
- `ToolModifier`: Core routing logic
  - `on_begin_invoke_async()`: Creates initial supervisor node
  - `on_post_invoke_async()`: Processes tool calls, routes to next node
  - `_process_tool_calls()`: Handles LLM tool calling
  - `_create_next_action_node()`: Classification-based routing
- `RunnableToolNode`: Represents tool execution results
- `RunnableSupervisorNode`: Wrapper for supervisor decisions

**3. `source/modules/agents/planning/src/omni_aiq_planning/`**

**`nodes/planning_node.py`**
- `PlanningGenNode`: Generates plans using LLM
  - Loads system prompt from `systems/planning_gen_system.md`
  - Supports `short_plan` and `add_details` modes
  - Outputs: Structured plan in markdown format

- `PlanningNetworkNode`: NetworkNode wrapper for planning
  - Registers as a tool with description
  - Adds `PlanningModifier` to manage execution

**`modifiers/planning_modifier.py`**
- `PlanningModifier`: **Core planning execution logic**

  **Key Hooks:**
  - `on_post_invoke_async()`:
    - **Branch 1**: Captures plan from PlanningGenNode output
    - **Branch 2**: Generates details for steps on-demand
    - Shares plan with MultiAgentNetworkNode
    - Initializes `plan_status` tracking

  - `on_pre_invoke_async()`:
    - **Branch 1**: Injects step instructions to guide supervisor
    - **Branch 2**: Adds tool information to planning node

  **Key Methods:**
  - `_find_planning_and_multi_agent_networks()`: Locates both networks in active stack
  - `_extract_plan()`: Parses plan from markdown into structured dict
  - `_get_next_pending_step()`: Finds next step to execute
  - `_build_step_instruction_message()`: Formats step guidance for supervisor

**`nodes/systems/planning_gen_system.md`**
- System prompt for plan generation
- Defines plan format and structure
- Instructions for detailed vs. short plans
- Examples of well-formed plans

**`nodes/systems/planning_tools_system.md`**
- Tool-aware planning instructions
- Inserted when planning needs to know available tools
- Guides planning to create executable plans

### Current Execution Flow

```
1. User Request → MultiAgent Supervisor
   ↓
2. Supervisor decides: "I need a plan"
   ↓
3. Supervisor calls "planning" tool (function calling or classification)
   ↓
4. PlanningNetworkNode activated
   ↓
5. PlanningGenNode generates plan (with tools context)
   ↓
6. PlanningModifier.on_post_invoke_async():
   - Detects plan in AIMessage output
   - Extracts plan structure
   - Finds MultiAgentNetworkNode in active networks
   - Injects self into MultiAgent with priority -1000
   - Shares plan via metadata
   - Initializes plan_status: {1: "pending", 2: "pending", ...}
   ↓
7. MultiAgent continues execution
   ↓
8. PlanningModifier.on_pre_invoke_async():
   - Before supervisor invokes next node
   - Gets next pending step
   - Injects RunnableHumanNode with step instructions
   - Updates plan_status[step_number] = "in_progress"
   ↓
9. Supervisor sees step instruction, decides which tool to call
   ↓
10. Tool executes, returns result
    ↓
11. PlanningModifier.on_post_invoke_async():
    - Marks step as completed (current implementation)
    - (No replanning logic yet)
    ↓
12. Repeat 8-11 for each step until plan complete
```

### Critical Integration Points

**1. Network Discovery**
```python
# In PlanningModifier
planning_network_node, multi_agent_network_node = \
    self._find_planning_and_multi_agent_networks(network)
```
- Searches `RunnableNetwork.get_active_networks()`
- Planning network = current network (PlanningNetworkNode)
- MultiAgent network = next network in stack (MultiAgentNetworkNode)

**2. Modifier Injection**
```python
# In PlanningModifier.on_post_invoke_async()
multi_agent_network_node.add_modifier(self, once=True, priority=-1000)
```
- Injects PlanningModifier into MultiAgent supervisor
- `once=True`: Modifier not re-added on subsequent invocations
- `priority=-1000`: Runs before other modifiers (early in pipeline)

**3. Step Guidance**
```python
# In PlanningModifier.on_pre_invoke_async()
follow_the_plan_message = self._build_step_instruction_message(
    current_step_number, current_step
)
with network:
    follow_the_plan_node = RunnableHumanNode(human_message=follow_the_plan_message)
    follow_the_plan_node.parents.clear()
    node._add_parent(follow_the_plan_node)
```
- Creates HumanMessage before supervisor invokes
- Supervisor sees step instruction as part of conversation
- Guides tool selection and question formation

**4. Metadata Sharing**
```python
# Plan shared via metadata
network.metadata["current_plan"] = self.current_plan
network.metadata["plan_status"] = self.plan_status

# Plan structure:
{
    "title": "Plan Title",
    "steps": [
        {
            "step_number": 1,
            "title": "Step title",
            "details": ["detail1", "detail2"]
        },
        ...
    ]
}
```

---

## Design Decision: Planning Review Steps

### Why Not Traditional Checkpointing?

After analyzing LangGraph's checkpoint mechanisms and AutoGPT's semantic checkpoints, we decided **against traditional checkpointing** for these reasons:

1. **User doesn't need it**: Explicitly stated "not interested in checkpoint actually"
2. **Parallel execution exists**: User has parallel execution in another MR
3. **Storage complexity**: Persistent checkpointing requires DB/storage infrastructure
4. **State serialization**: Complex to serialize entire network state
5. **Over-engineering**: Most value comes from **adaptive replanning**, not state snapshots

### Why Not Implicit Checkpoint Metadata?

Initial consideration was to add checkpoint markers via metadata:
```python
# REJECTED APPROACH
{
    "step_number": 3,
    "checkpoint_type": "reflection",  # Hidden metadata
    "reflection_prompt": "..."
}
```

**Problems with this approach:**
- ❌ **Not explicit**: Checkpoints hidden in metadata
- ❌ **Not discoverable**: Can't see checkpoints in plan output
- ❌ **Not programmatic**: Hard-coded checkpoint types
- ❌ **Planning Agent blind**: Agent doesn't decide where checkpoints go
- ❌ **Not debuggable**: Checkpoints don't show in execution logs

### Solution: Explicit Planning Review Steps

**Core Idea:**
> The Planning Agent explicitly generates "Planning Review" steps as part of the plan itself, making reviews a first-class citizen.

**Example:**
```
PLAN: Deploy Web Application

Step 1: Set up infrastructure
- Create cloud instances
- Configure networking

Step 2: Planning Review - Infrastructure Assessment    ← EXPLICIT REVIEW STEP
- Review focus: Verify infrastructure is ready
- Previous steps: Step 1
- Decision points:
  * Is infrastructure configured correctly?
  * Are all required resources available?
- Potential outcomes:
  * CONTINUE - Proceed to database setup
  * REPLAN - Fix infrastructure issues
  * ABORT - Infrastructure setup failed

Step 3: Deploy database
- Install PostgreSQL
- Configure security

Step 4: Deploy application server
- Deploy app container
- Configure load balancer

Step 5: Planning Review - Deployment Verification    ← EXPLICIT REVIEW STEP
- Review focus: Verify both services running
- Previous steps: Steps 3, 4
- Decision points:
  * Are both services healthy?
  * Can they communicate?

Step 6: Run integration tests
- Execute test suite
- Validate functionality
```

### Advantages of This Approach

1. **✅ Explicit and Visible**: Review steps appear in the plan
2. **✅ Planning Agent Decides**: Agent determines where reviews are needed
3. **✅ Natural Routing**: Reviews route to planning tool like any other step
4. **✅ Self-Correcting**: Agent can replan dynamically based on results
5. **✅ No Special Infrastructure**: Uses existing step execution mechanism
6. **✅ Clean Data Model**: No hidden metadata flags
7. **✅ Debuggable**: Review steps show in logs and UI
8. **✅ Flexible**: Agent can add reviews wherever appropriate
9. **✅ Programmatic**: Reviews are code, not configuration

### Design Principles

**Principle 1: Planning Agent is Intelligent**
- Agent decides during planning where reviews are needed
- Not hard-coded rules or heuristics
- Agent considers:
  - Complexity of upcoming steps
  - Risk level of operations
  - Dependencies between steps
  - Criticality of decisions

**Principle 2: Reviews are First-Class Steps**
- Review steps are indistinguishable from action steps in structure
- They execute through the same routing mechanism
- They have step numbers, titles, and details
- They participate in plan_status tracking

**Principle 3: Self-Correction through Reflection**
- Reviews invoke the Planning Agent in "review mode"
- Agent evaluates progress against objective
- Agent makes strategic decisions (CONTINUE/REPLAN/COMPLETE/ABORT)
- Agent can generate updated plans on the fly

**Principle 4: Minimal Architectural Changes**
- Builds on existing PlanningModifier infrastructure
- Uses existing network discovery and modifier injection
- Leverages existing step routing in MultiAgent
- No new components or services required

---

## Detailed Implementation Specification

### 1. Enhanced Plan Structure

**Current Structure:**
```python
{
    "title": "Plan Title",
    "steps": [
        {
            "step_number": 1,
            "title": "Step title",
            "details": ["detail1", "detail2"]
        }
    ]
}
```

**Enhanced Structure:**
```python
{
    "title": "Plan Title",
    "steps": [
        {
            "step_number": 1,
            "title": "Set up infrastructure",
            "step_type": "action",              # NEW: Type of step
            "details": [
                "Create cloud instances",
                "Configure networking"
            ]
        },
        {
            "step_number": 2,
            "title": "Planning Review - Infrastructure Assessment",
            "step_type": "planning_review",     # NEW: Review step type
            "details": [],
            "review_context": {                 # NEW: Review metadata
                "review_focus": "Verify infrastructure is ready for deployment",
                "previous_steps": "Step 1",
                "decision_points": [
                    "Is infrastructure configured correctly?",
                    "Are all required resources available?",
                    "Should we proceed with database setup?"
                ],
                "potential_outcomes": [
                    "CONTINUE - Proceed to database setup",
                    "REPLAN - Fix infrastructure issues first",
                    "ABORT - Infrastructure setup failed"
                ]
            }
        },
        {
            "step_number": 3,
            "title": "Deploy database",
            "step_type": "action",
            "details": [...]
        }
    ]
}
```

**New Fields:**
- `step_type`: "action" | "planning_review"
- `review_context`: Object containing review metadata (only for planning_review steps)
  - `review_focus`: What to assess in this review
  - `previous_steps`: Which steps to evaluate
  - `decision_points`: Key questions to answer
  - `potential_outcomes`: Possible decisions (CONTINUE/REPLAN/ABORT/COMPLETE)

### 2. Planning Agent System Prompt Enhancement

**File:** `source/modules/agents/planning/src/omni_aiq_planning/nodes/systems/planning_gen_system.md`

**Add Section: Planning Review Steps**

```markdown
## Planning Review Steps

In addition to action steps, you should insert **Planning Review** steps at strategic points:

### When to Insert Planning Reviews

1. **After critical actions** - Review outcomes before proceeding to next phase
2. **Before major transitions** - Assess readiness (e.g., dev → staging → production)
3. **After parallel/complex operations** - Verify all parts completed successfully
4. **Before high-risk operations** - Confirm prerequisites and safety measures
5. **Periodically in long plans** - Reassess every 3-5 action steps
6. **After failure-prone steps** - Evaluate results and adjust if needed

### Planning Review Format

For Planning Review steps, use this exact format:

```
Step N: Planning Review - <brief review focus>
- Review focus: <what to assess>
- Previous steps: <which steps to review>
- Decision points:
  * <key question 1>
  * <key question 2>
  * <key question 3>
- Potential outcomes:
  * CONTINUE - <when to proceed with existing plan>
  * REPLAN - <when to adjust remaining steps>
  * ABORT - <when to stop execution>
  * COMPLETE - <when objective already achieved>
```

### Example Plan with Reviews

```
PLAN: Deploy Machine Learning Model to Production

Step 1: Prepare model artifacts
- Export trained model to ONNX format
- Package dependencies and requirements
- Validate model file integrity

Step 2: Planning Review - Model Preparation Assessment
- Review focus: Verify model artifacts are ready for deployment
- Previous steps: Step 1
- Decision points:
  * Is the model exported correctly without errors?
  * Are all dependencies packaged and compatible?
  * Is the model file size within limits?
- Potential outcomes:
  * CONTINUE - Proceed to staging deployment
  * REPLAN - Fix export issues and regenerate artifacts
  * ABORT - Model export failed critically

Step 3: Deploy to staging environment
- Create staging infrastructure on cloud
- Deploy model container to staging
- Configure health check endpoints

Step 4: Run validation tests on staging
- Execute comprehensive test suite
- Verify prediction accuracy against benchmarks
- Load test with sample traffic

Step 5: Planning Review - Staging Validation
- Review focus: Assess staging deployment and test results
- Previous steps: Steps 3, 4
- Decision points:
  * Did staging deployment succeed without errors?
  * Are all tests passing?
  * Is prediction accuracy above 95% threshold?
  * Can staging handle expected load?
- Potential outcomes:
  * CONTINUE - Ready for production deployment
  * REPLAN - Fix failing tests and redeploy to staging
  * ABORT - Model performance unacceptable for production

Step 6: Production deployment
- Deploy model to production cluster
- Configure production load balancer
- Enable monitoring and alerting

Step 7: Run smoke tests on production
- Execute minimal test suite on production
- Verify basic functionality
- Check error rates and latency

Step 8: Planning Review - Production Verification
- Review focus: Confirm production deployment is successful and stable
- Previous steps: Steps 6, 7
- Decision points:
  * Is production deployment healthy?
  * Are smoke tests passing?
  * Are error rates and latency within acceptable ranges?
  * Any critical errors in logs?
- Potential outcomes:
  * CONTINUE - Mark deployment complete
  * REPLAN - Rollback and fix issues
  * COMPLETE - Deployment successful, objective achieved
```

### Guidelines for Planning Reviews

1. **Naming Convention**: Always start with "Planning Review - " followed by brief focus
2. **Frequency**: Insert reviews every 2-4 action steps in complex plans
3. **Critical Points**: Always add reviews before risky operations (deployments, migrations, deletions)
4. **Transition Points**: Add reviews between major phases
5. **Decision Points**: List 2-5 specific questions that need answering
6. **Outcomes**: Always include CONTINUE, REPLAN, and at least one termination option (ABORT or COMPLETE)
7. **Clarity**: Make review focus and decision points concrete and answerable

### What NOT to Do

- ❌ Don't add review steps after every single action (too frequent)
- ❌ Don't make decision points vague ("Is everything okay?")
- ❌ Don't forget to reference which previous steps to evaluate
- ❌ Don't add reviews where no meaningful decision can be made
- ❌ Don't use review steps as regular actions with a different name

Remember: Planning Review steps are executed by you (the Planning Agent) in review mode, so make them actionable, specific, and strategically placed.
```

### 3. Plan Extraction Enhancement

**File:** `source/modules/agents/planning/src/omni_aiq_planning/modifiers/planning_modifier.py`

**Update `_extract_plan()` method:**

```python
def _extract_plan(self, content: str) -> Dict[str, Any]:
    """Extract and structure the plan from the content.

    Enhanced to detect and parse Planning Review steps.

    Args:
        content: The content containing the plan

    Returns:
        dict: Structured plan with format:
            {
                "title": "Plan title",
                "steps": [
                    {
                        "step_number": 1,
                        "title": "Step title",
                        "step_type": "action" | "planning_review",
                        "details": ["detail1", "detail2", ...],
                        "review_context": {...}  # Only for planning_review steps
                    },
                    ...
                ]
            }
    """
    # Extract plan title
    plan_title_match = re.search(r"PLAN:\s*(.+?)(?:\n|$)", content)
    plan_title = plan_title_match.group(1).strip() if plan_title_match else "Untitled Plan"

    # Extract steps
    steps = []
    step_matches = re.finditer(r"Step (\d+):\s*(.+?)(?=\nStep \d+:|$)", content, re.DOTALL)

    for match in step_matches:
        step_number = int(match.group(1))
        step_content = match.group(2).strip()

        # Extract step title (first line)
        lines = step_content.split("\n")
        title = lines[0].strip()

        # Detect Planning Review steps by title pattern
        step_type = "action"  # default
        review_context = None

        if "Planning Review" in title or "planning review" in title.lower():
            step_type = "planning_review"
            review_context = self._extract_review_context(step_content)

        # Extract details from bullet points (lines starting with -)
        details = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("-") and not stripped.startswith("- Review focus:") \
               and not stripped.startswith("- Previous steps:") \
               and not stripped.startswith("- Decision points:") \
               and not stripped.startswith("- Potential outcomes:"):
                # Regular detail line
                details.append(stripped[1:].strip())

        steps.append({
            "step_number": step_number,
            "title": title,
            "step_type": step_type,
            "details": details,
            "review_context": review_context
        })

    # Sort steps by step number to ensure proper ordering
    steps.sort(key=lambda x: x["step_number"])

    return {"title": plan_title, "steps": steps}


def _extract_review_context(self, step_content: str) -> Dict[str, Any]:
    """Extract review context from Planning Review step content.

    Parses the structured review information including focus, decision points,
    and potential outcomes.

    Args:
        step_content: The full content of the Planning Review step

    Returns:
        dict: Review context with keys:
            - review_focus: str
            - previous_steps: str
            - decision_points: List[str]
            - potential_outcomes: List[str]
    """
    context = {
        "review_focus": "",
        "previous_steps": "",
        "decision_points": [],
        "potential_outcomes": []
    }

    lines = step_content.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("- Review focus:"):
            context["review_focus"] = line.replace("- Review focus:", "").strip()
            current_section = None

        elif line.startswith("- Previous steps:"):
            context["previous_steps"] = line.replace("- Previous steps:", "").strip()
            current_section = None

        elif line.startswith("- Decision points:"):
            current_section = "decision_points"

        elif line.startswith("- Potential outcomes:"):
            current_section = "potential_outcomes"

        elif line.startswith("*") and current_section:
            # Bullet point under current section
            text = line.strip("* ").strip()
            context[current_section].append(text)

    return context
```

### 4. Planning Review Execution Logic

**File:** `source/modules/agents/planning/src/omni_aiq_planning/modifiers/planning_modifier.py`

**Add new system prompt constant at top of file:**

```python
# Planning Review System Prompt
# Used when Planning Agent evaluates execution progress
PLANNING_REVIEW_SYSTEM_PROMPT = """You are the Planning Agent in REVIEW mode.

Your role is to assess execution progress and make strategic decisions about the plan.

## Your Responsibilities

1. **Evaluate Completed Steps**: Analyze what has been done and the results achieved
2. **Assess Current State**: Determine if we're on track to achieve the objective
3. **Make Strategic Decisions**: Decide whether to continue, replan, complete early, or abort
4. **Provide Clear Reasoning**: Explain your decision based on evidence from execution

## Decision Types

**CONTINUE**: Use when:
- Completed steps achieved their intended outcomes
- No significant issues or blockers detected
- Remaining plan steps are still appropriate
- On track to achieve the original objective with current plan

**REPLAN**: Use when:
- Completed steps revealed new information requiring plan adjustment
- Issues or failures detected that need addressing before continuing
- Original plan steps no longer make sense given current state
- A better approach has been identified based on results so far
- Assumptions made during planning were incorrect

**COMPLETE**: Use when:
- Original objective has already been achieved
- Remaining steps are no longer necessary
- Early success makes further steps redundant
- Goal accomplished more efficiently than planned

**ABORT**: Use when:
- Fundamental blocker prevents objective completion
- Critical failures that cannot be recovered from
- Original objective is no longer achievable or viable
- Continuing execution would be wasteful or harmful
- Safety or quality constraints cannot be met

## Output Format

Always respond with this exact format:

```
DECISION: [CONTINUE | REPLAN | COMPLETE | ABORT]

REASONING: <2-3 sentences explaining your decision based on evidence>

[If REPLAN, include:]
UPDATED_PLAN:
Step N: <first remaining step>
- <detail>
Step N+1: <next step>
- <detail>
...

[If COMPLETE, include:]
FINAL_RESULT: <summary of what was accomplished>

[If ABORT, include:]
ABORT_REASON: <specific reason why we cannot proceed>
```

## Guidelines for Decision-Making

1. **Be Evidence-Based**: Base decisions on actual step results, not assumptions
2. **Be Decisive**: Make clear choices, don't hedge or be ambiguous
3. **Be Concise**: Keep reasoning brief but substantive
4. **Be Strategic**: Think about the big picture and ultimate objective
5. **Be Realistic**: Don't continue if fundamental issues exist
6. **Be Adaptive**: Be willing to change approach when needed

## Examples

### Example 1: CONTINUE
```
DECISION: CONTINUE

REASONING: Infrastructure setup completed successfully with all resources available and properly configured. Database deployment is ready to proceed as planned. No issues detected.
```

### Example 2: REPLAN
```
DECISION: REPLAN

REASONING: Staging tests revealed model accuracy of 89%, below the 95% threshold. We need to improve the model before production deployment. Original plan assumed tests would pass.

UPDATED_PLAN:
Step 6: Analyze test failures and low accuracy causes
- Review prediction errors on test set
- Identify data quality issues
- Determine if model architecture needs adjustment

Step 7: Improve model performance
- Add more training data if needed
- Tune hyperparameters
- Consider ensemble approach

Step 8: Planning Review - Model Improvement Assessment
- Review focus: Verify model improvements
- Decision points: Is accuracy now >95%?

Step 9: Redeploy to staging and retest
- Deploy improved model to staging
- Run full test suite again

Step 10: Planning Review - Final Staging Validation
- Review focus: Confirm readiness for production
- Decision points: All tests passing? Accuracy acceptable?

Step 11: Deploy to production
- Proceed with production deployment
```

### Example 3: COMPLETE
```
DECISION: COMPLETE

REASONING: Integration tests in step 6 revealed all functionality working perfectly. The deployment is stable and all success criteria met. No need for additional verification steps.

FINAL_RESULT: Application successfully deployed to production. All services healthy, tests passing, and monitoring confirms stable operation.
```

### Example 4: ABORT
```
DECISION: ABORT

REASONING: Infrastructure setup failed due to insufficient cloud quota. Attempts to increase quota were denied. Cannot proceed without required compute resources.

ABORT_REASON: Cloud quota limit prevents infrastructure provisioning. Manual intervention required to resolve quota with cloud provider before retrying deployment.
```

Remember: You are the Planning Agent evaluating your own plan. Be honest about what's working and what isn't. Your decision guides the entire execution—make it count.
"""
```

**Update `on_pre_invoke_async()` method:**

```python
async def on_pre_invoke_async(self, network, node):
    """Pre-invoke hook that guides execution and handles Planning Review steps.

    This method:
    1. For Planning Review steps: Invokes Planning Agent in review mode
    2. For regular action steps: Injects step instructions to guide supervisor
    3. For PlanningGenNode: Adds tools information

    Args:
        network: The current network being executed
        node: The node about to be invoked
    """
    from ..nodes.planning_node import PlanningGenNode, PlanningNetworkNode

    # ============================================
    # BRANCH 1: Inject plan step instructions for supervisor OR handle Planning Review
    # ============================================
    if (
        isinstance(network, MultiAgentNetworkNode)
        and type(node) == get_node_factory().get_registered_node_type(network.default_node)
        and not network.get_children(node)
    ):
        # Get the current plan and its execution status
        plan_metadata = self._get_plan_metadata(network)
        if not plan_metadata:
            return

        plan_status, current_plan = plan_metadata

        # Find the next pending step to execute
        next_step_info = self._get_next_pending_step(plan_status, current_plan)
        if not next_step_info:
            return

        current_step_number, current_step = next_step_info

        # Check if this is a Planning Review step
        if current_step.get("step_type") == "planning_review":
            # Special handling for Planning Review steps
            await self._inject_planning_review_step(
                network, node, current_step_number, current_step
            )
        else:
            # Regular action step - inject step instruction as before
            follow_the_plan_message = self._build_step_instruction_message(
                current_step_number, current_step
            )

            # Inject the plan instruction as a parent node
            with network:
                follow_the_plan_node = RunnableHumanNode(human_message=follow_the_plan_message)
                follow_the_plan_node.parents.clear()
                node._add_parent(follow_the_plan_node)

        # Update the step status to in_progress
        plan_status[current_step_number] = "in_progress"

    # ============================================
    # BRANCH 2: Add tools information to planning node (UNCHANGED)
    # ============================================
    elif (
        not node.invoked
        and isinstance(node, PlanningGenNode)
        and isinstance(network, PlanningNetworkNode)
        and node.outputs is None
        and not network.get_children(node)
    ):
        # Existing logic for adding tools information
        planning_network_node, multi_agent_network_node = self._find_planning_and_multi_agent_networks(network)

        if multi_agent_network_node is None:
            return

        system_message = self._get_tools_system_message(multi_agent_network_node)
        node.inputs.append(RunnableSystemAppend(system_message=system_message))
```

**Add new method `_inject_planning_review_step()`:**

```python
async def _inject_planning_review_step(self, network, node, step_number, step):
    """Handle Planning Review step by invoking Planning Agent in review mode.

    Creates a planning node with review context and connects it to the supervisor.
    The Planning Agent will evaluate progress and make a strategic decision.

    Args:
        network: The MultiAgentNetworkNode
        node: The supervisor node about to invoke
        step_number: The step number of the review step
        step: The review step data with review_context
    """
    # Build review prompt with execution context
    review_prompt = self._build_review_prompt(network, step_number, step)

    with network:
        # Get the planning node name (should be registered as a tool)
        planning_node_name = network.metadata.get("plan_gen_node")
        if not planning_node_name:
            # Fallback: If no planning node available, just continue
            # This shouldn't happen in normal execution
            logger.warning(f"Planning Review step {step_number} executed but no planning node available")
            return

        # Create a Planning Agent node for review
        planning_review_node = get_node_factory().create_node(planning_node_name)

        # Add system prompt for review mode
        planning_review_node.inputs.append(RunnableSystemAppend(
            system_message=PLANNING_REVIEW_SYSTEM_PROMPT
        ))

        # Add the review prompt
        planning_review_node.inputs.append(RunnableAppend(
            message=HumanMessage(content=review_prompt)
        ))

        # Mark this node as a planning review for special handling in post_invoke
        planning_review_node.metadata["planning_review"] = True
        planning_review_node.metadata["review_step_number"] = step_number

        # Clear parents and connect to supervisor
        planning_review_node.parents.clear()
        node._add_parent(planning_review_node)


def _build_review_prompt(self, network, step_number, step) -> str:
    """Build the review prompt for Planning Agent.

    Constructs a detailed prompt with:
    - Original objective and plan title
    - Completed steps with results
    - Review focus and decision points
    - Remaining steps

    Args:
        network: The MultiAgentNetworkNode
        step_number: Current review step number
        step: The review step data

    Returns:
        str: Formatted review prompt
    """
    review_context = step.get("review_context", {})

    # Get completed steps and their results
    completed_steps_info = self._get_completed_steps_summary(network, step_number)

    # Get remaining steps (excluding other review steps in summary)
    remaining_steps_info = self._get_remaining_steps_summary(network, step_number)

    # Build the prompt
    prompt = f"""PLANNING REVIEW

You are reviewing the execution progress of the plan: {self.current_plan['title']}

Original Objective: {network.metadata.get('original_objective', 'Not specified')}

## Review Focus
{review_context.get('review_focus', 'Assess progress and determine next steps')}

## Completed Steps
{completed_steps_info}

## Current Situation
You are at Step {step_number}: {step['title']}

## Decision Points

The key questions you need to answer:
"""

    decision_points = review_context.get('decision_points', [])
    if decision_points:
        for i, point in enumerate(decision_points, 1):
            prompt += f"{i}. {point}\n"
    else:
        prompt += "1. Are we on track to achieve the objective?\n"
        prompt += "2. Do the remaining steps still make sense?\n"

    prompt += f"""
## Remaining Planned Steps
{remaining_steps_info}

## Potential Outcomes

Based on the plan, these outcomes were anticipated:
"""

    potential_outcomes = review_context.get('potential_outcomes', [])
    if potential_outcomes:
        for outcome in potential_outcomes:
            prompt += f"- {outcome}\n"

    prompt += """
## Your Task

Based on the completed steps and their results, make a strategic decision:

**Required Response Format:**
```
DECISION: [CONTINUE | REPLAN | COMPLETE | ABORT]

REASONING: <brief explanation of your decision based on evidence>

[If REPLAN, provide updated plan:]
UPDATED_PLAN:
Step {next_num}: <new step>
- <details>
Step {next_num+1}: <next step>
- <details>
...

[If COMPLETE:]
FINAL_RESULT: <summary of what was accomplished>

[If ABORT:]
ABORT_REASON: <specific reason why we cannot proceed>
```

Make your decision now.
"""

    return prompt


def _get_completed_steps_summary(self, network, current_step_number) -> str:
    """Generate summary of completed steps with results.

    Args:
        network: The network containing execution state
        current_step_number: The current review step number

    Returns:
        str: Formatted summary of completed steps
    """
    summary = []

    for step in self.current_plan['steps']:
        if step['step_number'] >= current_step_number:
            break

        status = self.plan_status.get(step['step_number'], 'pending')

        if status in ['completed', 'in_progress']:
            # Get result if available
            result = step.get('result', {})
            output = result.get('output', 'No output captured')

            # Truncate long outputs
            if len(output) > 300:
                output = output[:300] + "... (truncated)"

            summary.append(f"""
Step {step['step_number']}: {step['title']} [{status.upper()}]
Result: {output}
""")

    return "\n".join(summary) if summary else "No steps completed yet."


def _get_remaining_steps_summary(self, network, current_step_number) -> str:
    """Generate summary of remaining steps (excluding review steps).

    Args:
        network: The network
        current_step_number: Current review step number

    Returns:
        str: Formatted summary of remaining action steps
    """
    remaining = []

    for step in self.current_plan['steps']:
        if step['step_number'] <= current_step_number:
            continue

        # Include all remaining steps in summary (including review steps for context)
        step_type_marker = "[REVIEW]" if step.get('step_type') == 'planning_review' else ""
        remaining.append(f"Step {step['step_number']}: {step['title']} {step_type_marker}")

    return "\n".join(remaining) if remaining else "No remaining steps."
```

**Update `on_post_invoke_async()` method:**

```python
async def on_post_invoke_async(self, network, node):
    """Post-invoke hook that processes plans and handles Planning Review decisions.

    This handles multiple workflows:
    1. Plan generation from PlanningGenNode
    2. Dynamic detail generation for steps
    3. Planning Review decision processing (NEW)

    Args:
        network: The current network being executed
        node: The node that was just invoked
    """
    from ..nodes.planning_node import PlanningGenNode, PlanningNetworkNode

    # ============================================
    # BRANCH 1: Process plan generation from PlanningGenNode (UNCHANGED)
    # ============================================
    if (
        node.invoked
        and isinstance(node, PlanningGenNode)
        and isinstance(network, PlanningNetworkNode)
        and isinstance(node.outputs, AIMessage)
        and node.outputs.content
        and not network.get_children(node)
    ):
        # Existing plan generation logic
        planning_network_node, multi_agent_network_node = self._find_planning_and_multi_agent_networks(network)

        # Validation
        if planning_network_node:
            from ..nodes.planning_node import PlanningNetworkNode
            if not isinstance(planning_network_node, PlanningNetworkNode):
                planning_network_node = None

        if multi_agent_network_node and not isinstance(multi_agent_network_node, MultiAgentNetworkNode):
            multi_agent_network_node = None

        # Extract and validate the plan
        plan_content = node.outputs.content.strip()
        if self._is_valid_plan(plan_content):
            self.current_plan = self._extract_plan(plan_content)
            node.metadata["current_plan"] = self.current_plan
            network.metadata["current_plan"] = self.current_plan

            # Share the plan with multi-agent network if available
            if multi_agent_network_node:
                add_details = node.metadata.get("add_details", False)
                if add_details:
                    multi_agent_network_node.metadata["plan_gen_node"] = network.default_node

                multi_agent_network_node.metadata["current_plan"] = self.current_plan
                multi_agent_network_node.add_modifier(self, once=True, priority=-1000)

                # Initialize plan execution status tracking
                self.plan_status = {step["step_number"]: "pending" for step in self.current_plan["steps"]}
                multi_agent_network_node.metadata["plan_status"] = self.plan_status

    # ============================================
    # BRANCH 2: Add details to plan steps on request (UNCHANGED)
    # ============================================
    elif (
        node.invoked
        and isinstance(node, RunnableHumanNode)
        and isinstance(network, MultiAgentNetworkNode)
        and isinstance(node.outputs, HumanMessage)
        and node.metadata.get("multi_agent_classification", None) == True
        and network.metadata.get("plan_gen_node", None)
        and node.outputs.content
    ):
        # Existing detail generation logic (unchanged)
        children = network.get_children(node)
        if len(children) == 1 and children[0].outputs is None:
            planning_node_name = network.metadata.get("plan_gen_node", None)

            plan_metadata = self._get_plan_metadata(network)
            if not plan_metadata:
                return

            plan_status, current_plan = plan_metadata
            next_step_info = self._get_next_pending_step(plan_status, current_plan)
            if not next_step_info:
                return

            current_step_number, current_step = next_step_info

            if current_step.get("details", None):
                return

            with network:
                current_step_title = current_step["title"]
                formatted_plan = self._format_plan_for_context(current_plan)
                provide_details_message = HumanMessage(
                    content=f"Here is the full plan:\n\n{formatted_plan}\n\n"
                           f"Please provide specific implementation details for Step {current_step_number}: {current_step_title}"
                )

                tools_system_message = self._get_tools_system_message(network)

                planning_node = get_node_factory().create_node(planning_node_name)
                planning_node.inputs.append(RunnableSystemAppend(system_message=tools_system_message))
                planning_node.inputs.append(RunnableSystemAppend(system_message=PLAN_DETAILS_SYSTEM))
                planning_node.inputs.append(RunnableAppend(message=provide_details_message))
                planning_node.parents.clear()
                planning_node.metadata["plan_details"] = True

                children[0]._add_parent(planning_node)

    # ============================================
    # BRANCH 3: Handle Planning Review decision results (NEW)
    # ============================================
    elif (
        node.invoked
        and node.metadata.get("planning_review") == True
        and isinstance(node.outputs, AIMessage)
        and node.outputs.content
    ):
        review_step_number = node.metadata.get("review_step_number")
        if review_step_number:
            await self._process_planning_review_decision(
                network, node, review_step_number
            )

    # ============================================
    # BRANCH 4: Track completed action steps (NEW)
    # ============================================
    elif (
        isinstance(network, MultiAgentNetworkNode)
        and node.invoked
        and node.metadata.get("tool_call_name") in network.route_nodes
        and self.current_plan  # We have an active plan
    ):
        # A tool/agent step just completed, store its result
        step_number = self._get_current_step_number(node)
        if step_number:
            current_step = self._get_step_by_number(step_number)
            if current_step and current_step.get('step_type') == 'action':
                # Store result for later review
                current_step['result'] = {
                    'output': node.outputs.content if hasattr(node.outputs, 'content') else str(node.outputs),
                    'metadata': dict(node.metadata)
                }
                # Mark as completed
                self.plan_status[step_number] = "completed"
```

**Add new methods for Planning Review processing:**

```python
async def _process_planning_review_decision(self, network, node, step_number):
    """Process the Planning Agent's review decision.

    Parses the review output and takes action based on decision:
    - CONTINUE: Mark step complete, continue with plan
    - REPLAN: Update remaining steps with new plan
    - COMPLETE: Mark remaining steps as skipped, set final output
    - ABORT: Mark remaining steps as aborted, set abort message

    Args:
        network: The MultiAgentNetworkNode
        node: The planning review node with decision output
        step_number: The review step number
    """
    review_content = node.outputs.content
    decision = self._parse_review_decision(review_content)

    # Mark review step as completed
    self.plan_status[step_number] = "completed"

    # Log the decision
    logger.info(f"Planning Review Step {step_number}: {decision['action']} - {decision['reasoning']}")

    if decision['action'] == 'CONTINUE':
        # Continue with existing plan - no changes needed
        # Next step will be picked up normally in on_pre_invoke_async
        logger.info(f"Planning Review {step_number}: CONTINUE - proceeding with existing plan")

    elif decision['action'] == 'REPLAN':
        # Update remaining steps with new plan
        logger.info(f"Planning Review {step_number}: REPLAN - updating remaining steps")

        new_steps = decision.get('updated_steps', [])
        if new_steps:
            self._replace_remaining_steps(step_number, new_steps)

            # Update plan in network metadata
            network.metadata["current_plan"] = self.current_plan

            logger.info(f"Plan updated with {len(new_steps)} new remaining steps")
        else:
            logger.warning(f"REPLAN decision but no updated steps provided")

    elif decision['action'] == 'COMPLETE':
        # Task completed early
        logger.info(f"Planning Review {step_number}: COMPLETE - objective achieved")

        # Mark all remaining steps as skipped
        for step in self.current_plan['steps']:
            if step['step_number'] > step_number:
                self.plan_status[step['step_number']] = 'skipped'

        # Set final output
        final_result = decision.get('final_result', 'Task completed successfully')
        network.outputs = AIMessage(content=f"FINAL {final_result}")
        network.metadata['plan_completed_early'] = True

    elif decision['action'] == 'ABORT':
        # Abort execution
        logger.warning(f"Planning Review {step_number}: ABORT - execution terminated")

        # Mark remaining steps as aborted
        for step in self.current_plan['steps']:
            if step['step_number'] > step_number:
                self.plan_status[step['step_number']] = 'aborted'

        # Set abort message
        abort_reason = decision.get('abort_reason', decision.get('reasoning', 'Execution aborted by Planning Review'))
        network.outputs = AIMessage(content=f"ABORT: {abort_reason}")
        network.metadata['plan_aborted'] = True
        network.metadata['abort_reason'] = abort_reason


def _parse_review_decision(self, content: str) -> Dict[str, Any]:
    """Parse the Planning Review decision from response.

    Extracts:
    - DECISION: CONTINUE | REPLAN | COMPLETE | ABORT
    - REASONING: Explanation
    - UPDATED_PLAN: New steps (if REPLAN)
    - FINAL_RESULT: Summary (if COMPLETE)
    - ABORT_REASON: Reason (if ABORT)

    Args:
        content: The review response content

    Returns:
        dict: Parsed decision with action, reasoning, and relevant fields
    """
    import re

    decision_match = re.search(r'DECISION:\s*(CONTINUE|REPLAN|COMPLETE|ABORT)', content, re.IGNORECASE)
    reasoning_match = re.search(
        r'REASONING:\s*(.+?)(?=\n\n|\nUPDATED_PLAN:|\nFINAL_RESULT:|\nABORT_REASON:|$)',
        content,
        re.DOTALL | re.IGNORECASE
    )

    result = {
        'action': decision_match.group(1).upper() if decision_match else 'CONTINUE',
        'reasoning': reasoning_match.group(1).strip() if reasoning_match else 'No reasoning provided'
    }

    # Parse UPDATED_PLAN if REPLAN
    if result['action'] == 'REPLAN':
        plan_match = re.search(r'UPDATED_PLAN:\s*(.+?)(?=\n*$)', content, re.DOTALL | re.IGNORECASE)
        if plan_match:
            updated_plan_text = plan_match.group(1).strip()
            result['updated_steps'] = self._parse_updated_steps(updated_plan_text)
        else:
            result['updated_steps'] = []

    # Parse FINAL_RESULT if COMPLETE
    if result['action'] == 'COMPLETE':
        final_match = re.search(r'FINAL_RESULT:\s*(.+?)(?=\n\n|$)', content, re.DOTALL | re.IGNORECASE)
        if final_match:
            result['final_result'] = final_match.group(1).strip()

    # Parse ABORT_REASON if ABORT
    if result['action'] == 'ABORT':
        abort_match = re.search(r'ABORT_REASON:\s*(.+?)(?=\n\n|$)', content, re.DOTALL | re.IGNORECASE)
        if abort_match:
            result['abort_reason'] = abort_match.group(1).strip()

    return result


def _parse_updated_steps(self, plan_text: str) -> List[Dict]:
    """Parse updated steps from replanned text.

    Extracts steps in format:
    Step N: Title
    - Detail 1
    - Detail 2

    Args:
        plan_text: The UPDATED_PLAN section text

    Returns:
        list: List of step dictionaries
    """
    import re

    steps = []
    step_matches = re.finditer(r'Step (\d+):\s*(.+?)(?=\nStep \d+:|$)', plan_text, re.DOTALL)

    for match in step_matches:
        step_number = int(match.group(1))
        step_content = match.group(2).strip()

        # Parse title and details
        lines = step_content.split("\n")
        title = lines[0].strip()

        # Detect if it's a Planning Review step
        step_type = "action"
        review_context = None

        if "Planning Review" in title or "planning review" in title.lower():
            step_type = "planning_review"
            review_context = self._extract_review_context(step_content)

        # Extract details
        details = [
            line.strip()[1:].strip()
            for line in lines[1:]
            if line.strip().startswith("-") and not any(
                line.strip().startswith(prefix) for prefix in
                ["- Review focus:", "- Previous steps:", "- Decision points:", "- Potential outcomes:"]
            )
        ]

        steps.append({
            "step_number": step_number,
            "title": title,
            "step_type": step_type,
            "details": details,
            "review_context": review_context
        })

    return steps


def _replace_remaining_steps(self, current_step_number: int, new_steps: List[Dict]):
    """Replace remaining steps in the plan with new steps from replanning.

    Keeps all completed steps (including the current review step) and replaces
    all pending steps with the new steps, renumbering as needed.

    Args:
        current_step_number: The review step number (last completed)
        new_steps: List of new step dictionaries
    """
    # Keep completed steps (up to and including current review step)
    completed_steps = [
        step for step in self.current_plan['steps']
        if step['step_number'] <= current_step_number
    ]

    # Add new steps with renumbered step numbers
    next_step_number = current_step_number + 1
    for new_step in new_steps:
        new_step['step_number'] = next_step_number
        completed_steps.append(new_step)

        # Initialize status as pending
        self.plan_status[next_step_number] = 'pending'
        next_step_number += 1

    # Remove old pending steps from status
    for step_num in list(self.plan_status.keys()):
        if step_num > current_step_number and step_num not in [s['step_number'] for s in completed_steps]:
            del self.plan_status[step_num]

    # Update the plan
    self.current_plan['steps'] = completed_steps

    logger.info(f"Plan updated: kept {len([s for s in completed_steps if s['step_number'] <= current_step_number])} completed steps, added {len(new_steps)} new steps")


def _get_current_step_number(self, node) -> Optional[int]:
    """Get the step number currently being executed from node metadata.

    Args:
        node: The node that just completed

    Returns:
        int: Step number if found, None otherwise
    """
    # Check if this node corresponds to a plan step
    tool_call_name = node.metadata.get("tool_call_name")
    if not tool_call_name:
        return None

    # Find which step is currently in_progress or was just completed
    for step_num, status in self.plan_status.items():
        if status == "in_progress":
            return step_num

    return None


def _get_step_by_number(self, step_number: int) -> Optional[Dict]:
    """Get step data by step number.

    Args:
        step_number: The step number to find

    Returns:
        dict: Step data if found, None otherwise
    """
    for step in self.current_plan.get('steps', []):
        if step['step_number'] == step_number:
            return step
    return None
```

### 5. Integration with MultiAgent Supervisor

**No changes required to MultiAgent code!**

The beauty of this design is that Planning Review steps integrate seamlessly:

1. **Planning Review steps are registered tools**: When the plan contains a Planning Review step, the MultiAgent supervisor sees it just like any other tool in `route_nodes`

2. **Routing happens naturally**: When supervisor needs to execute "Planning Review - X", it routes to the planning tool (just like routing to any other agent/tool)

3. **PlanningModifier handles the logic**: The modifier detects it's a review step and invokes the Planning Agent in review mode

4. **Results flow back naturally**: Review decision flows through normal post_invoke hooks

**Execution flow:**
```
MultiAgent Supervisor
  ↓
Sees: "Step 2: Planning Review - Infrastructure Assessment"
  ↓
Routes to: "planning" tool (or whatever the planning node is named)
  ↓
PlanningModifier.on_pre_invoke_async() detects: step_type == "planning_review"
  ↓
Injects Planning Agent node with review prompt
  ↓
Planning Agent evaluates and responds with decision
  ↓
PlanningModifier.on_post_invoke_async() processes decision
  ↓
If REPLAN: Updates plan with new steps
If CONTINUE: Next step executes normally
If COMPLETE/ABORT: Terminates execution
```

---

## File-by-File Changes

### Critical Files Requiring Modification

#### 1. `planning_gen_system.md`
**Location:** `source/modules/agents/planning/src/omni_aiq_planning/nodes/systems/planning_gen_system.md`

**Changes:**
- Add comprehensive section on "Planning Review Steps"
- Define when to insert review steps
- Provide review step format template
- Add examples of plans with reviews
- Include guidelines and anti-patterns

**Lines to modify:** Add new section after current content (~200 new lines)

**Priority:** HIGH - This teaches the Planning Agent how to generate reviews

---

#### 2. `planning_modifier.py`
**Location:** `source/modules/agents/planning/src/omni_aiq_planning/modifiers/planning_modifier.py`

**Changes:**

**A. Add constants (top of file, ~after line 38):**
```python
PLANNING_REVIEW_SYSTEM_PROMPT = """..."""  # ~100 lines
```

**B. Update `_extract_plan()` method (~line 433):**
- Add detection of step_type ("action" | "planning_review")
- Call `_extract_review_context()` for review steps
- Include review_context in step dict

**C. Add new method `_extract_review_context()` (~line 478):**
- Parse review_focus, previous_steps, decision_points, potential_outcomes
- ~40 lines

**D. Update `on_pre_invoke_async()` method (~line 281):**
- Add branch to detect Planning Review steps
- Call `_inject_planning_review_step()` for review steps
- Keep existing logic for action steps

**E. Add new method `_inject_planning_review_step()` (~after line 347):**
- Create planning node with review prompt
- Add PLANNING_REVIEW_SYSTEM_PROMPT
- Mark with metadata for post_invoke
- ~50 lines

**F. Add new method `_build_review_prompt()` (~after _inject_planning_review_step):**
- Build comprehensive review prompt
- Include completed steps, decision points, remaining steps
- ~80 lines

**G. Add new method `_get_completed_steps_summary()` (~after _build_review_prompt):**
- Format completed steps with results
- Truncate long outputs
- ~30 lines

**H. Add new method `_get_remaining_steps_summary()` (~after _get_completed_steps_summary):**
- Format remaining steps
- Mark review steps
- ~20 lines

**I. Update `on_post_invoke_async()` method (~line 163):**
- Add BRANCH 3: Handle Planning Review decision results
- Add BRANCH 4: Track completed action steps with results
- ~30 lines added

**J. Add new method `_process_planning_review_decision()` (~after on_post_invoke_async):**
- Process CONTINUE/REPLAN/COMPLETE/ABORT decisions
- Update plan, status, outputs accordingly
- ~80 lines

**K. Add new method `_parse_review_decision()` (~after _process_planning_review_decision):**
- Parse decision, reasoning, updated_plan, final_result, abort_reason
- ~50 lines

**L. Add new method `_parse_updated_steps()` (~after _parse_review_decision):**
- Parse UPDATED_PLAN text into step dictionaries
- Detect review steps in updated plan
- ~50 lines

**M. Add new method `_replace_remaining_steps()` (~after _parse_updated_steps):**
- Keep completed steps, replace pending with new steps
- Renumber steps, update plan_status
- ~40 lines

**N. Add new method `_get_current_step_number()` (~after _replace_remaining_steps):**
- Get step number from node metadata
- ~15 lines

**O. Add new method `_get_step_by_number()` (~after _get_current_step_number):**
- Find step data by number
- ~10 lines

**Total additions:** ~600 lines

**Priority:** CRITICAL - Core implementation logic

---

#### 3. No changes to other files!

**Files that DO NOT need modification:**
- ✅ `multi_agent_network_node.py` - Works as-is
- ✅ `multi_agent_register.py` - Works as-is
- ✅ `lc_agent_function.py` - Works as-is
- ✅ `planning_node.py` - Works as-is
- ✅ `planning_tools_system.md` - Works as-is

---

### Summary of Changes

| File | Lines Added | Lines Modified | Complexity | Priority |
|------|-------------|----------------|------------|----------|
| `planning_gen_system.md` | ~200 | 0 | Low | HIGH |
| `planning_modifier.py` | ~600 | ~50 | High | CRITICAL |
| **TOTAL** | **~800** | **~50** | - | - |

---

## Integration with MultiAgent Supervisor

### How Planning Review Steps Flow Through the System

**1. Plan Generation Phase**
```
User Request: "Deploy ML model to production"
  ↓
MultiAgent Supervisor calls: planning tool
  ↓
PlanningGenNode generates plan with Planning Review steps
  ↓
PlanningModifier.on_post_invoke_async() - BRANCH 1:
  - Extracts plan including step_type and review_context
  - Shares with MultiAgentNetworkNode via metadata
  - Injects self into MultiAgent with priority -1000
  - Initializes plan_status for all steps
```

**2. Action Step Execution**
```
MultiAgent Supervisor executes regular action step:
  ↓
PlanningModifier.on_pre_invoke_async():
  - Detects step_type == "action"
  - Injects step instruction as RunnableHumanNode
  - Updates plan_status[step_number] = "in_progress"
  ↓
Supervisor sees instruction, routes to appropriate tool
  ↓
Tool executes, returns result
  ↓
PlanningModifier.on_post_invoke_async() - BRANCH 4:
  - Stores result in step['result']
  - Marks plan_status[step_number] = "completed"
```

**3. Planning Review Step Execution**
```
MultiAgent Supervisor encounters Planning Review step:
  ↓
PlanningModifier.on_pre_invoke_async():
  - Detects step_type == "planning_review"
  - Calls _inject_planning_review_step()
  - Creates planning node with:
    * PLANNING_REVIEW_SYSTEM_PROMPT
    * Review prompt with completed steps & decision points
  - Marks node.metadata["planning_review"] = True
  ↓
Planning Agent invoked in review mode
  ↓
Planning Agent evaluates:
  - Analyzes completed step results
  - Answers decision points
  - Makes strategic decision (CONTINUE/REPLAN/COMPLETE/ABORT)
  - Outputs decision with reasoning
  ↓
PlanningModifier.on_post_invoke_async() - BRANCH 3:
  - Detects metadata["planning_review"] == True
  - Calls _process_planning_review_decision()
  - Parses decision from output
  - Takes action based on decision:

    IF CONTINUE:
      - Marks review step as completed
      - Next step executes normally

    IF REPLAN:
      - Parses UPDATED_PLAN
      - Calls _replace_remaining_steps()
      - Updates current_plan and plan_status
      - Next step is first new step

    IF COMPLETE:
      - Marks remaining steps as "skipped"
      - Sets network.outputs = AIMessage("FINAL ...")
      - Execution terminates early

    IF ABORT:
      - Marks remaining steps as "aborted"
      - Sets network.outputs = AIMessage("ABORT: ...")
      - Sets metadata["plan_aborted"] = True
      - Execution terminates
```

### Metadata Flow

**Planning Modifier → MultiAgent Network:**
```python
# In PlanningModifier.on_post_invoke_async() - BRANCH 1
multi_agent_network_node.metadata["current_plan"] = self.current_plan
multi_agent_network_node.metadata["plan_status"] = self.plan_status
multi_agent_network_node.metadata["plan_gen_node"] = network.default_node  # For detail generation
```

**Step Results Storage:**
```python
# In PlanningModifier.on_post_invoke_async() - BRANCH 4
current_step['result'] = {
    'output': node.outputs.content,
    'metadata': dict(node.metadata)
}
```

**Review Node Marking:**
```python
# In PlanningModifier._inject_planning_review_step()
planning_review_node.metadata["planning_review"] = True
planning_review_node.metadata["review_step_number"] = step_number
```

### Network Discovery Mechanism

```python
# In PlanningModifier._find_planning_and_multi_agent_networks()
active_networks = list(RunnableNetwork.get_active_networks())

# Stack looks like:
# [0] - Outermost (usually MultiAgentNetworkNode)
# [1] - PlanningNetworkNode (current)
# [2] - Inner networks (if any)

for it_network in active_networks:
    if it_network is current_network:
        planning_network_node = it_network
        continue
    if planning_network_node is not None:
        # Next network after planning is MultiAgent
        multi_agent_network_node = it_network
        break
```

---

## Testing Strategy

### Unit Tests

**Test File:** `source/modules/agents/planning/tests/test_planning_review.py`

**Test Cases:**

1. **test_extract_plan_with_review_steps**
   - Generate plan with Planning Review steps
   - Verify step_type correctly identified
   - Verify review_context extracted

2. **test_extract_review_context**
   - Parse review step content
   - Verify review_focus, decision_points, potential_outcomes

3. **test_parse_review_decision_continue**
   - Parse "DECISION: CONTINUE" response
   - Verify action and reasoning extracted

4. **test_parse_review_decision_replan**
   - Parse "DECISION: REPLAN" with UPDATED_PLAN
   - Verify updated_steps parsed correctly

5. **test_parse_review_decision_complete**
   - Parse "DECISION: COMPLETE" with FINAL_RESULT
   - Verify final_result extracted

6. **test_parse_review_decision_abort**
   - Parse "DECISION: ABORT" with ABORT_REASON
   - Verify abort_reason extracted

7. **test_replace_remaining_steps**
   - Setup plan with 5 steps
   - Call _replace_remaining_steps() after step 2
   - Verify completed steps preserved
   - Verify new steps added and renumbered

8. **test_get_completed_steps_summary**
   - Create plan with completed steps
   - Verify summary formatting

9. **test_get_remaining_steps_summary**
   - Create plan with remaining steps
   - Verify summary includes action and review steps

### Integration Tests

**Test File:** `source/modules/agents/planning/tests/test_planning_integration.py`

**Test Cases:**

1. **test_full_planning_review_flow_continue**
   - Mock Planning Agent to generate plan with review step
   - Execute step 1 (action)
   - Execute step 2 (review) → CONTINUE decision
   - Verify step 3 executes

2. **test_full_planning_review_flow_replan**
   - Execute steps 1-2 (actions)
   - Execute step 3 (review) → REPLAN decision
   - Verify new steps replace remaining steps
   - Verify new step 4 executes

3. **test_full_planning_review_flow_complete**
   - Execute step 1 (action)
   - Execute step 2 (review) → COMPLETE decision
   - Verify remaining steps marked as skipped
   - Verify final output set

4. **test_full_planning_review_flow_abort**
   - Execute step 1 (action) → failure
   - Execute step 2 (review) → ABORT decision
   - Verify remaining steps marked as aborted
   - Verify abort metadata set

5. **test_nested_replanning**
   - Execute review → REPLAN with new review step
   - Execute new action step
   - Execute new review → CONTINUE
   - Verify nested replanning works

### Manual Testing Scenarios

**Scenario 1: Simple Deployment with Review**
```yaml
# config.yaml
workflow:
  _type: MultiAgent
  tool_names:
    - planning
    - deployment_tool
    - testing_tool
```

**User Query:** "Deploy the application to staging"

**Expected Plan:**
```
PLAN: Deploy Application to Staging

Step 1: Build application artifacts
Step 2: Deploy to staging server
Step 3: Planning Review - Deployment Verification
Step 4: Run smoke tests
```

**Expected Flow:**
- Step 1 executes → builds artifacts
- Step 2 executes → deploys
- Step 3 (review) → Planning Agent evaluates deployment
  - If successful → CONTINUE → Step 4 runs
  - If failed → REPLAN → New steps to fix and retry

**Scenario 2: ML Model Deployment with Multiple Reviews**

**User Query:** "Deploy the ML model to production"

**Expected Plan:**
```
PLAN: Deploy ML Model to Production

Step 1: Prepare model artifacts
Step 2: Planning Review - Model Readiness
Step 3: Deploy to staging
Step 4: Run validation tests
Step 5: Planning Review - Staging Validation
Step 6: Deploy to production
Step 7: Planning Review - Production Verification
```

**Expected Flow:**
- Steps 1-2: Prepare → Review (verify artifacts)
- Steps 3-5: Stage → Test → Review (verify staging)
- Steps 6-7: Deploy to prod → Review (verify production)

**Test Replanning:**
- If Step 4 tests fail (accuracy < 95%)
- Step 5 review should REPLAN:
  ```
  UPDATED_PLAN:
  Step 6: Analyze test failures
  Step 7: Retrain model
  Step 8: Planning Review - Retraining Assessment
  Step 9: Redeploy to staging
  Step 10: Rerun validation tests
  Step 11: Planning Review - Final Staging Check
  Step 12: Deploy to production
  ```

**Scenario 3: Early Completion**

**User Query:** "Fix the authentication bug"

**Expected Plan:**
```
PLAN: Fix Authentication Bug

Step 1: Analyze authentication logs
Step 2: Identify root cause
Step 3: Planning Review - Root Cause Analysis
Step 4: Implement fix
Step 5: Test fix
Step 6: Planning Review - Fix Validation
Step 7: Deploy fix to production
```

**Test Early Completion:**
- Step 1 discovers: "Bug already fixed in latest deployment"
- Step 3 review: COMPLETE
- Expected: Steps 4-7 marked as skipped

---

## Future Enhancements

### 1. User-in-the-Loop Reviews

**Concept:** Allow human approval at critical review points

**Implementation:**
```python
# In plan structure
{
    "step_type": "planning_review",
    "review_context": {
        "requires_human_approval": True,  # NEW
        "approval_timeout": 3600  # seconds
    }
}

# In _process_planning_review_decision()
if review_context.get("requires_human_approval"):
    # Pause execution, wait for user approval
    network.metadata["awaiting_user_approval"] = {
        "step": step_number,
        "prompt": "Approve proceeding with production deployment?",
        "decision": decision
    }
    # Resume when user sets metadata["user_approval_granted"] = True
```

### 2. Review Step Metrics

**Concept:** Track how often reviews trigger replanning

**Metrics to collect:**
- Review decision distribution (CONTINUE vs REPLAN vs COMPLETE vs ABORT)
- Average steps between reviews
- Replan frequency by objective type
- Success rate after replanning

**Implementation:**
```python
# In _process_planning_review_decision()
metrics = {
    "review_step": step_number,
    "decision": decision['action'],
    "reasoning": decision['reasoning'],
    "timestamp": datetime.now(),
    "objective": network.metadata.get('original_objective')
}
# Log to analytics system
```

### 3. Learning from Review Decisions

**Concept:** Use review decisions to improve future planning

**Approach:**
- Store (objective, original_plan, execution_results, review_decisions)
- Build a dataset of planning patterns
- Fine-tune planning agent on successful patterns
- Prompt augmentation with similar past scenarios

### 4. Conditional Review Steps

**Concept:** Reviews that only execute if certain conditions met

**Implementation:**
```python
{
    "step_type": "planning_review",
    "conditional": True,
    "condition": {
        "previous_step_failed": True,
        # OR
        "metric_threshold": {"accuracy": "<0.95"}
    }
}

# In on_pre_invoke_async()
if step.get("conditional"):
    if not self._evaluate_condition(step["condition"]):
        # Skip this review, mark as skipped
        plan_status[step_number] = "skipped"
        return
```

### 5. Multi-Agent Reviews

**Concept:** Multiple agents participate in review decision

**Implementation:**
```python
{
    "step_type": "planning_review",
    "review_agents": ["planning", "safety_agent", "cost_optimizer"]
}

# Each agent provides input
# Aggregate decisions or require consensus
```

### 6. Review Templates

**Concept:** Pre-defined review patterns for common scenarios

**Examples:**
- **Pre-Deployment Review**: Standard checks before production
- **Data Quality Review**: Verify data meets requirements
- **Cost Review**: Confirm operation within budget
- **Safety Review**: Security and compliance checks

**Implementation:**
```python
# In planning_gen_system.md
"""
## Review Templates

Use these templates for common review scenarios:

### Pre-Deployment Review Template
Step N: Planning Review - Pre-Deployment Check
- Review focus: Verify system is ready for deployment
- Decision points:
  * Are all tests passing?
  * Is the staging environment validated?
  * Are rollback procedures in place?
  * Have stakeholders approved?
- Potential outcomes:
  * CONTINUE - Proceed with deployment
  * REPLAN - Address missing prerequisites
"""
```

---

## Appendix: Complete Example

### Example: ML Model Deployment with Replanning

**User Request:**
```
"Deploy the fraud detection model to production"
```

**Generated Plan (by Planning Agent):**
```markdown
PLAN: Deploy Fraud Detection Model to Production

Step 1: Export model to ONNX format
- Load trained model from checkpoint
- Convert to ONNX with opset 14
- Validate exported model structure

Step 2: Package model dependencies
- Create requirements.txt with versions
- Package preprocessing pipeline
- Include configuration files

Step 3: Planning Review - Model Artifacts Assessment
- Review focus: Verify model and dependencies are production-ready
- Previous steps: Steps 1, 2
- Decision points:
  * Is the ONNX export successful and validated?
  * Are all dependencies correctly versioned?
  * Is the package size within deployment limits?
- Potential outcomes:
  * CONTINUE - Proceed to staging deployment
  * REPLAN - Fix export or packaging issues
  * ABORT - Critical issues prevent deployment

Step 4: Deploy model to staging environment
- Provision staging infrastructure
- Deploy model container
- Configure health check endpoints

Step 5: Run comprehensive validation tests
- Execute functional test suite (100+ test cases)
- Verify prediction accuracy on holdout set
- Load test with realistic traffic (1000 req/min)
- Check latency requirements (<100ms p95)

Step 6: Planning Review - Staging Validation Assessment
- Review focus: Evaluate staging performance and readiness
- Previous steps: Steps 4, 5
- Decision points:
  * Did staging deployment succeed without errors?
  * Are all functional tests passing?
  * Is prediction accuracy above 97% threshold?
  * Does the model meet latency requirements?
  * Can it handle production load?
- Potential outcomes:
  * CONTINUE - Ready for production deployment
  * REPLAN - Address performance or accuracy issues
  * ABORT - Model unsuitable for production

Step 7: Deploy to production cluster
- Deploy to production Kubernetes cluster
- Configure autoscaling (2-10 replicas)
- Set up monitoring and alerting

Step 8: Run production smoke tests
- Execute minimal test suite on production
- Verify basic functionality
- Check initial error rates

Step 9: Planning Review - Production Deployment Verification
- Review focus: Confirm production deployment success
- Previous steps: Steps 7, 8
- Decision points:
  * Is production deployment healthy?
  * Are smoke tests passing?
  * Are error rates acceptable (<0.1%)?
  * Is latency within SLA (<100ms p95)?
- Potential outcomes:
  * CONTINUE - Complete deployment
  * REPLAN - Rollback and fix issues
  * COMPLETE - Deployment successful, all criteria met

Step 10: Enable gradual traffic ramp-up
- Start with 5% production traffic
- Monitor for 15 minutes
- Gradually increase to 100%

Step 11: Planning Review - Final Deployment Verification
- Review focus: Confirm stable production operation
- Previous steps: Steps 10
- Decision points:
  * Is the model handling production traffic well?
  * Are metrics stable?
  * Any anomalies detected?
- Potential outcomes:
  * COMPLETE - Deployment complete and stable
  * REPLAN - Adjust traffic or rollback
```

**Execution with Replanning:**

```
=== EXECUTION LOG ===

[MultiAgent Supervisor] Starting plan execution...

[Step 1] Action: Export model to ONNX format
  → Routing to: ml_toolkit
  → ml_toolkit executing...
  → Result: "Model exported successfully to fraud_model.onnx (127 MB)"
  → Status: COMPLETED

[Step 2] Action: Package model dependencies
  → Routing to: ml_toolkit
  → ml_toolkit executing...
  → Result: "Dependencies packaged. Total size: 145 MB"
  → Status: COMPLETED

[Step 3] Planning Review: Model Artifacts Assessment
  → Routing to: planning
  → Planning Agent (Review Mode) evaluating...

  Review Input:
    Completed Steps:
      Step 1: Export successful (127 MB)
      Step 2: Dependencies packaged (145 MB)

    Decision Points:
      ✓ ONNX export successful
      ✓ Dependencies versioned correctly
      ⚠ Package size: 145 MB (limit: 200 MB) - acceptable

  → Planning Agent Decision:
    DECISION: CONTINUE
    REASONING: Model artifacts are production-ready. ONNX export validated
    successfully and dependency package is within size limits. Proceeding
    to staging deployment.

  → Status: COMPLETED

[Step 4] Action: Deploy to staging environment
  → Routing to: deployment_agent
  → deployment_agent executing...
  → Result: "Staging deployment complete. Service: fraud-model-staging:8080"
  → Status: COMPLETED

[Step 5] Action: Run comprehensive validation tests
  → Routing to: testing_agent
  → testing_agent executing...
  → Result: "Tests: 98/100 passed (2 failures). Accuracy: 94.3%. Latency: p95=85ms"
  → Status: COMPLETED ⚠️

[Step 6] Planning Review: Staging Validation Assessment
  → Routing to: planning
  → Planning Agent (Review Mode) evaluating...

  Review Input:
    Completed Steps:
      Step 4: Staging deployed successfully
      Step 5: Tests - 98/100 pass, Accuracy: 94.3%, Latency: 85ms

    Decision Points:
      ✓ Staging deployment: SUCCESS
      ⚠ Functional tests: 98% pass rate (2 failures)
      ✗ Accuracy: 94.3% < 97% threshold ← CRITICAL
      ✓ Latency: 85ms < 100ms requirement
      ✓ Load handling: Acceptable

  → Planning Agent Decision:
    DECISION: REPLAN
    REASONING: Accuracy of 94.3% is below the 97% threshold required for
    production deployment. This is a critical quality requirement for fraud
    detection. We need to investigate and improve model performance before
    proceeding.

    UPDATED_PLAN:
    Step 7: Analyze accuracy issues and test failures
    - Review the 2 failed test cases in detail
    - Analyze prediction errors on validation set
    - Identify patterns in misclassifications
    - Determine root cause (data drift, model issue, etc.)

    Step 8: Improve model performance
    - If data quality issue: Clean and augment training data
    - If model issue: Retune hyperparameters or adjust architecture
    - Retrain model with improvements
    - Target accuracy: >97%

    Step 9: Planning Review - Model Improvement Assessment
    - Review focus: Verify model improvements are effective
    - Previous steps: Steps 7, 8
    - Decision points:
      * Has root cause been addressed?
      * Is retrained model showing improvement?
      * Are we confident in proceeding?
    - Potential outcomes:
      * CONTINUE - Proceed to staging redeployment
      * REPLAN - Further iteration needed
      * ABORT - Cannot achieve accuracy target

    Step 10: Redeploy improved model to staging
    - Export new ONNX model
    - Package updated dependencies if needed
    - Deploy to staging environment

    Step 11: Rerun validation tests
    - Execute full test suite
    - Verify accuracy >97%
    - Confirm latency still acceptable

    Step 12: Planning Review - Final Staging Validation
    - Review focus: Confirm model meets all requirements
    - Previous steps: Steps 10, 11
    - Decision points:
      * All tests passing?
      * Accuracy >97%?
      * Ready for production?
    - Potential outcomes:
      * CONTINUE - Proceed to production
      * REPLAN - Additional fixes needed

    Step 13: Deploy to production cluster
    - Deploy to production Kubernetes cluster
    - Configure autoscaling

    Step 14: Run production smoke tests
    - Execute minimal test suite
    - Verify basic functionality

    Step 15: Planning Review - Production Verification
    - Review focus: Confirm production deployment
    - Previous steps: Steps 13, 14
    - Decision points:
      * Production deployment healthy?
      * Smoke tests passing?
    - Potential outcomes:
      * CONTINUE - Enable traffic
      * REPLAN - Rollback and fix

    Step 16: Enable gradual traffic ramp-up
    - Start with 5% traffic
    - Monitor and increase to 100%

  → Status: COMPLETED
  → Plan updated with 10 new steps (7-16)

[Step 7] Action: Analyze accuracy issues and test failures
  → Routing to: ml_toolkit
  → ml_toolkit executing...
  → Result: "Analysis complete. Root cause: Data drift in recent transactions.
             Retraining recommended with last 30 days of data."
  → Status: COMPLETED

[Step 8] Action: Improve model performance
  → Routing to: ml_toolkit
  → ml_toolkit executing (retraining)...
  → Result: "Model retrained with augmented dataset. New checkpoint saved."
  → Status: COMPLETED

[Step 9] Planning Review: Model Improvement Assessment
  → Routing to: planning
  → Planning Agent (Review Mode) evaluating...

  Review Input:
    Completed Steps:
      Step 7: Root cause identified (data drift)
      Step 8: Model retrained with recent data

    Decision Points:
      ✓ Root cause addressed (included recent 30 days)
      ? Model improvement not yet validated

  → Planning Agent Decision:
    DECISION: CONTINUE
    REASONING: Root cause clearly identified and addressed by including recent
    data in retraining. Retraining completed successfully. Ready to validate
    improvements on staging.

  → Status: COMPLETED

[Step 10] Action: Redeploy improved model to staging
  → Routing to: deployment_agent
  → deployment_agent executing...
  → Result: "Staging redeployed with new model version"
  → Status: COMPLETED

[Step 11] Action: Rerun validation tests
  → Routing to: testing_agent
  → testing_agent executing...
  → Result: "Tests: 100/100 passed. Accuracy: 98.1%. Latency: p95=82ms"
  → Status: COMPLETED ✓

[Step 12] Planning Review: Final Staging Validation
  → Routing to: planning
  → Planning Agent (Review Mode) evaluating...

  Review Input:
    Completed Steps:
      Step 10: Redeployed to staging
      Step 11: Tests - 100/100 pass, Accuracy: 98.1%, Latency: 82ms

    Decision Points:
      ✓ All tests passing: 100%
      ✓ Accuracy: 98.1% > 97% threshold
      ✓ Latency: 82ms < 100ms
      ✓ Ready for production: YES

  → Planning Agent Decision:
    DECISION: CONTINUE
    REASONING: All validation criteria met. Accuracy improved to 98.1%, all
    tests passing, and latency excellent. Model is production-ready.

  → Status: COMPLETED

[Step 13] Action: Deploy to production cluster
  → Routing to: deployment_agent
  → deployment_agent executing...
  → Result: "Production deployment complete. Service: fraud-model-prod:8080"
  → Status: COMPLETED

[Step 14] Action: Run production smoke tests
  → Routing to: testing_agent
  → testing_agent executing...
  → Result: "Smoke tests: 10/10 passed. Error rate: 0%. Latency: 78ms"
  → Status: COMPLETED

[Step 15] Planning Review: Production Verification
  → Routing to: planning
  → Planning Agent (Review Mode) evaluating...

  Review Input:
    Completed Steps:
      Step 13: Production deployed
      Step 14: Smoke tests - 10/10 pass, 0% errors, 78ms latency

    Decision Points:
      ✓ Production deployment: HEALTHY
      ✓ Smoke tests: ALL PASSING
      ✓ Error rates: EXCELLENT (0%)
      ✓ Latency: EXCELLENT (78ms)

  → Planning Agent Decision:
    DECISION: CONTINUE
    REASONING: Production deployment successful with perfect smoke test results.
    Ready to enable traffic ramp-up.

  → Status: COMPLETED

[Step 16] Action: Enable gradual traffic ramp-up
  → Routing to: deployment_agent
  → deployment_agent executing...
  → Result: "Traffic ramp-up complete. 5%→25%→50%→100%. All metrics stable.
             Deployment successful."
  → Status: COMPLETED

=== PLAN EXECUTION COMPLETE ===

Final Status: SUCCESS
Total Steps Executed: 16 (10 action, 4 review, 2 replanned)
Replanning Events: 1 (Step 6 - accuracy issue)
Outcome: Fraud detection model successfully deployed to production with 98.1% accuracy
```

**Key Takeaways from Example:**

1. **Planning Review at Step 3**: Verified artifacts ready, allowed CONTINUE
2. **Planning Review at Step 6**: Detected accuracy issue (94.3% < 97%), triggered REPLAN
3. **Replanning**: Added 10 new steps to analyze, improve, revalidate
4. **Nested Reviews**: New plan included 2 additional review steps
5. **Final Success**: After improvement, accuracy 98.1%, all criteria met
6. **Transparent**: All review decisions visible in execution log

---

## Conclusion

**Planning Review Steps** transform the planning system from a static plan executor into an intelligent, adaptive system that:

- **Thinks strategically** at critical junctures
- **Adapts to reality** when execution deviates from plan
- **Self-corrects** when issues arise
- **Makes decisions** transparently

By making reviews explicit steps in the plan, we achieve:
- ✅ **Simplicity**: No hidden checkpoints or complex metadata
- ✅ **Intelligibility**: Reviews visible in plan and logs
- ✅ **Flexibility**: Planning Agent decides where reviews are needed
- ✅ **Integration**: Seamless with existing MultiAgent supervisor
- ✅ **Effectiveness**: Can recover from failures and adjust approach

**Implementation effort:** ~800 lines of code across 2 files
**Architectural impact:** Minimal - builds on existing infrastructure
**Value delivered:** Transforms planning from static to adaptive

---

**Ready for implementation in your new environment!**
