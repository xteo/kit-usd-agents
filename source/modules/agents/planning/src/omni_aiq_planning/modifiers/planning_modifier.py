## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import re
import logging
from typing import Any, Dict, List, Optional, Set
from langchain_core.messages import AIMessage, HumanMessage
from lc_agent import (
    MultiAgentNetworkNode,
    NetworkModifier,
    RunnableHumanNode,
    RunnableNetwork,
    RunnableSystemAppend,
    RunnableAppend,
    get_node_factory,
)
from lc_agent.utils.multi_agent_utils import get_routing_tools_info
from pathlib import Path

logger = logging.getLogger(__name__)


# Helper function to read markdown files
def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


# Get the system prompts directory
SYSTEM_PATH = Path(__file__).parent.parent.joinpath("nodes", "systems")

# Load the planning tools system prompt
PLAN_TOOLS_SYSTEM_PATH = SYSTEM_PATH.joinpath("planning_tools_system.md")
PLAN_TOOLS_SYSTEM = read_md_file(str(PLAN_TOOLS_SYSTEM_PATH))

# Plan instruction message format
# This format string controls how plan instructions are formatted when injected into the conversation
# Variables available:
#   {step_number} - The current step number
#   {step_title} - The title/description of the step
#   {step_details} - The detailed instructions for the step (newline-separated)
PLAN_INSTRUCTION_FORMAT = (
    "(Please follow the plan and respond the next action. "
    'Either "<tool_name> <question>" or "FINAL <answer>". '
    "Step {step_number}: {step_title}{step_details})"
)

# Alternative format examples:
# PLAN_INSTRUCTION_FORMAT = "**Step {step_number}**: {step_title}\n{step_details}"
# PLAN_INSTRUCTION_FORMAT = "Let's proceed with step {step_number} - {step_title}. {step_details}"
# PLAN_INSTRUCTION_FORMAT = "[PLAN] Step {step_number}: {step_title}\nDetails:\n{step_details}"


# Plan details request message
# This message is used when the planning system needs to provide more details for a specific step
# It instructs the planning agent on how to respond when asked for step details
PLAN_DETAILS_SYSTEM = """You are being asked to explain how to do a specific step from the plan.

Here's what you need to do:

1. Give a short answer - just a few sentences
2. Use simple words that anyone can understand
3. Get straight to the point - no long introductions
4. Only talk about the actions needed for this step
5. Be specific about what to do

Things to avoid:
- Don't reprint the whole plan
- Don't say things like "To execute Step 1" or "For this step"
- Don't use complex technical words when simple ones work
- Don't give long explanations

Just tell them what to do in the simplest way possible. Think of it like giving quick directions to a friend."""


class DependencyGraph:
    """
    Manages task dependencies and determines execution readiness for parallel planning.

    This class implements a directed acyclic graph (DAG) to track dependencies between
    plan steps and enable parallel execution of independent tasks.

    Attributes:
        steps: Dictionary mapping step_number to step data
        adjacency_list: Map from step_number to list of steps that depend on it
        in_degree: Map from step_number to count of unsatisfied dependencies
        completed: Set of completed step numbers
    """

    def __init__(self, plan_steps: List[Dict]):
        """
        Build dependency graph from plan steps.

        Args:
            plan_steps: List of plan step dictionaries with dependencies
        """
        self.steps = {step["step_number"]: step for step in plan_steps}
        self.adjacency_list = {}  # step_num → [steps that depend on it]
        self.in_degree = {}  # step_num → count of unsatisfied dependencies
        self.completed: Set[int] = set()

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

        A step is ready when:
        1. All its dependencies are completed (in_degree == 0)
        2. It hasn't been completed yet

        Returns:
            List of step numbers ready for execution, sorted by step number
        """
        ready = []
        for step_num, degree in self.in_degree.items():
            if degree == 0 and step_num not in self.completed:
                ready.append(step_num)
        return sorted(ready)

    def mark_completed(self, step_num: int):
        """
        Mark a step as completed and update dependent steps.

        When a step completes:
        1. Add it to the completed set
        2. Reduce in-degree for all steps that depend on it
        3. This may cause new steps to become ready (in-degree reaches 0)

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

        Checks for:
        1. Non-existent step references
        2. Forward references (step depends on later step)
        3. Self-references
        4. Circular dependencies

        Returns:
            (is_valid, error_message): True with None if valid, False with error message if invalid
        """
        # Check for non-existent dependencies and forward/self references
        for step in self.steps.values():
            step_num = step["step_number"]
            for dep in step.get("dependencies", []):
                if dep not in self.steps:
                    return False, f"Step {step_num} depends on non-existent step {dep}"
                if dep >= step_num:
                    return False, f"Step {step_num} has forward/self dependency on step {dep}"

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
                    return False, "Circular dependency detected in plan"

        return True, None

    def get_dependency_status(self, step_num: int) -> Dict[str, Any]:
        """
        Get detailed dependency status for a step.

        Args:
            step_num: The step number to check

        Returns:
            Dictionary with:
                - ready: bool (all dependencies satisfied)
                - dependencies: List[int] (all dependency step numbers)
                - satisfied: List[int] (completed dependencies)
                - unsatisfied: List[int] (pending dependencies)
        """
        step = self.steps.get(step_num)
        if not step:
            return {
                "ready": False,
                "dependencies": [],
                "unsatisfied": [],
                "satisfied": []
            }

        dependencies = step.get("dependencies", [])
        satisfied = [d for d in dependencies if d in self.completed]
        unsatisfied = [d for d in dependencies if d not in self.completed]

        return {
            "ready": len(unsatisfied) == 0 and step_num not in self.completed,
            "dependencies": dependencies,
            "satisfied": satisfied,
            "unsatisfied": unsatisfied
        }

    def is_complete(self) -> bool:
        """
        Check if all steps in the plan have been completed.

        Returns:
            True if all steps completed, False otherwise
        """
        return len(self.completed) == len(self.steps)


class PlanningModifier(NetworkModifier):
    """
    Modifier that processes and enhances planning responses.

    This modifier handles two main workflows:
    
    1. Plan Generation and Sharing:
       - Captures plans generated by PlanningGenNode
       - Validates and extracts plan structure
       - Shares plans with MultiAgentNetworkNode for execution
       - Tracks plan execution status
    
    2. Plan Execution Guidance:
       - Injects step-by-step instructions to guide the supervisor
       - Provides tools information to the planning node
       - Handles dynamic detail generation for plan steps
    
    The modifier operates across different network types:
    - PlanningNetworkNode: Where plans are generated
    - MultiAgentNetworkNode: Where plans are executed
    """

    def __init__(self, max_parallel_steps: int = 5):
        super().__init__()
        self.current_plan = None
        self.plan_status = {}
        self.dependency_graph: Optional[DependencyGraph] = None
        self.max_parallel_steps = max_parallel_steps  # Maximum concurrent step execution

    def _find_planning_and_multi_agent_networks(self, current_network):
        """
        Find the planning network node and multi-agent network node from active networks.
        
        The multi-agent network is always the next network after the planning network
        in the active networks stack.
        
        Returns:
            tuple: (planning_network_node, multi_agent_network_node) or (None, None)
        """
        active_networks = list(RunnableNetwork.get_active_networks())
        planning_network_node = None
        multi_agent_network_node = None
        
        for it_network in active_networks:
            if it_network is current_network:
                # Found the planning network node, mark it
                planning_network_node = it_network
                continue
            if planning_network_node is not None:
                # The next network after planning is the multi-agent network
                multi_agent_network_node = it_network
                break
                
        return planning_network_node, multi_agent_network_node

    def _get_tools_system_message(self, network):
        """
        Get the tools system message for the given network.
        
        Returns:
            str: The formatted tools system message
        """
        tools_descriptions, tool_call_formats, example_tool_name = get_routing_tools_info(network, False)
        return PLAN_TOOLS_SYSTEM.replace("<tools>", tools_descriptions)

    def _format_plan_for_context(self, plan):
        """
        Format the plan into a readable string for context.
        
        Args:
            plan: The plan dictionary with title and steps
            
        Returns:
            str: Formatted plan string
        """
        formatted_plan = f"PLAN: {plan['title']}\n\n"
        
        for step in plan['steps']:
            formatted_plan += f"Step {step['step_number']}: {step['title']}\n"
            if step.get('details'):
                for detail in step['details']:
                    formatted_plan += f"- {detail}\n"
            formatted_plan += "\n"
            
        return formatted_plan.strip()

    async def on_post_invoke_async(self, network, node):
        """
        Post-invoke hook that processes plans from the PlanningGenNode.

        Args:
            network: The current network being executed
            node: The node that was just invoked
        """
        from ..nodes.planning_node import PlanningGenNode, PlanningNetworkNode

        # ============================================
        # BRANCH 1: Process plan generation from PlanningGenNode
        # This handles when the planning node has generated a plan
        # ============================================
        if (
            node.invoked
            and isinstance(node, PlanningGenNode)
            and isinstance(network, PlanningNetworkNode)
            and isinstance(node.outputs, AIMessage)
            and node.outputs.content
            and not network.get_children(node)
        ):
            # Get the active networks
            planning_network_node, multi_agent_network_node = self._find_planning_and_multi_agent_networks(network)

            # Validation - ensure we have the right network types
            if planning_network_node:
                from ..nodes.planning_node import PlanningNetworkNode
                if not isinstance(planning_network_node, PlanningNetworkNode):
                    planning_network_node = None
                    
            if multi_agent_network_node and not isinstance(multi_agent_network_node, MultiAgentNetworkNode):
                multi_agent_network_node = None

            # Extract and validate the plan from the node output
            plan_content = node.outputs.content.strip()
            if self._is_valid_plan(plan_content):
                # Store the extracted plan in multiple places for accessibility
                self.current_plan = self._extract_plan(plan_content)
                node.metadata["current_plan"] = self.current_plan
                network.metadata["current_plan"] = self.current_plan

                # Initialize dependency graph for parallel execution
                try:
                    self.dependency_graph = DependencyGraph(self.current_plan["steps"])

                    # Validate dependency graph
                    is_valid, error = self.dependency_graph.validate_dependencies()
                    if not is_valid:
                        logger.error(f"Invalid dependency graph: {error}")
                        # Plan has invalid dependencies - log error but continue
                        # Could potentially trigger re-planning here in future
                    else:
                        logger.info(f"Dependency graph initialized successfully for plan: {self.current_plan['title']}")
                except Exception as e:
                    logger.error(f"Failed to initialize dependency graph: {e}")
                    self.dependency_graph = None

                # Share the plan with multi-agent network if available
                if multi_agent_network_node:
                    # Check if we need to enable dynamic details addition
                    add_details = node.metadata.get("add_details", False)
                    if add_details:
                        multi_agent_network_node.metadata["plan_gen_node"] = network.default_node

                    # Share plan and register this modifier with the multi-agent network
                    multi_agent_network_node.metadata["current_plan"] = self.current_plan
                    multi_agent_network_node.add_modifier(self, once=True, priority=-1000)

                    # Initialize plan execution status tracking
                    self.plan_status = {step["step_number"]: "pending" for step in self.current_plan["steps"]}
                    multi_agent_network_node.metadata["plan_status"] = self.plan_status

                    # Share dependency graph with multi-agent network
                    if self.dependency_graph:
                        multi_agent_network_node.metadata["dependency_graph"] = self.dependency_graph
                    
        # ============================================
        # BRANCH 2: Add details to plan steps on request
        # This handles when user asks for more details about a plan step
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
            # Check if we can add details to the plan
            children = network.get_children(node)
            if len(children) == 1 and children[0].outputs is None:
                planning_node_name = network.metadata.get("plan_gen_node", None)

                # Get the current plan and execution status
                plan_metadata = self._get_plan_metadata(network)
                if not plan_metadata:
                    return

                plan_status, current_plan = plan_metadata

                # Find the next pending step that needs details
                next_step_info = self._get_next_pending_step(plan_status, current_plan)
                if not next_step_info:
                    return

                current_step_number, current_step = next_step_info

                # Skip if this step already has details
                if current_step.get("details", None):
                    return

                # Create a planning node to generate step details
                with network:
                    current_step_title = current_step["title"]

                    # Create messages and system prompts for detail generation
                    # Include the full plan for context
                    formatted_plan = self._format_plan_for_context(current_plan)
                    provide_details_message = HumanMessage(
                        content=f"Here is the full plan:\n\n{formatted_plan}\n\n"
                               f"Please provide specific implementation details for Step {current_step_number}: {current_step_title}"
                    )

                    tools_system_message = self._get_tools_system_message(network)

                    # Set up the planning node with appropriate prompts
                    planning_node = get_node_factory().create_node(planning_node_name)
                    planning_node.inputs.append(RunnableSystemAppend(system_message=tools_system_message))
                    planning_node.inputs.append(RunnableSystemAppend(system_message=PLAN_DETAILS_SYSTEM))
                    planning_node.inputs.append(RunnableAppend(message=provide_details_message))
                    planning_node.parents.clear()
                    planning_node.metadata["plan_details"] = True

                    # Connect the planning node to generate details
                    children[0]._add_parent(planning_node)

    async def on_pre_invoke_async(self, network, node):
        """
        Pre-invoke hook that guides execution and enables parallel step launching.

        Enhanced to support parallel execution via dependency graph:
        - Identifies all ready steps (dependencies satisfied)
        - Launches multiple steps concurrently up to max_parallel_steps
        - Respects dependency constraints automatically

        This method injects plan step instructions before the default node executes,
        ensuring the agent follows the established plan.
        """
        from ..nodes.planning_node import PlanningGenNode, PlanningNetworkNode

        # ============================================
        # BRANCH 1: Inject plan step instructions for supervisor (PARALLEL EXECUTION)
        # This guides the multi-agent supervisor to follow the plan
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

            # Check if we have a dependency graph for parallel execution
            if self.dependency_graph:
                # PARALLEL EXECUTION MODE: Use dependency graph to find ready steps
                ready_steps = self.dependency_graph.get_ready_steps()

                if not ready_steps:
                    # No steps ready
                    if self.dependency_graph.is_complete():
                        logger.info("All plan steps completed")
                    return

                # Determine how many steps we can launch in parallel
                currently_in_progress = sum(
                    1 for status in plan_status.values()
                    if status == "in_progress"
                )
                available_slots = self.max_parallel_steps - currently_in_progress

                if available_slots <= 0:
                    # Already at max parallelism, wait for some steps to complete
                    logger.debug(f"At max parallelism ({self.max_parallel_steps}), waiting for steps to complete")
                    return

                # Launch up to available_slots ready steps
                steps_to_launch = ready_steps[:available_slots]

                logger.info(f"Launching {len(steps_to_launch)} parallel steps: {steps_to_launch}")

                # Inject instructions for each ready step
                for step_num in steps_to_launch:
                    step = next(
                        (s for s in current_plan["steps"] if s["step_number"] == step_num),
                        None
                    )

                    if not step:
                        logger.warning(f"Step {step_num} not found in plan")
                        continue

                    # Mark as in_progress before launching
                    plan_status[step_num] = "in_progress"

                    # Build step instruction message
                    follow_the_plan_message = self._build_step_instruction_message(step_num, step)

                    # Inject the plan instruction as a parent node
                    with network:
                        follow_the_plan_node = RunnableHumanNode(human_message=follow_the_plan_message)
                        follow_the_plan_node.parents.clear()
                        follow_the_plan_node.metadata["plan_step_number"] = step_num
                        node._add_parent(follow_the_plan_node)

                    logger.debug(f"Launched step {step_num}: {step['title']}")

            else:
                # SEQUENTIAL EXECUTION MODE: Fallback to original behavior
                # Find the next pending step to execute
                next_step_info = self._get_next_pending_step(plan_status, current_plan)
                if not next_step_info:
                    return

                current_step_number, current_step = next_step_info

                # Build the instruction message for this step
                follow_the_plan_message = self._build_step_instruction_message(current_step_number, current_step)

                # Inject the plan instruction as a parent node
                with network:
                    # Inject the current step instruction
                    follow_the_plan_node = RunnableHumanNode(human_message=follow_the_plan_message)
                    follow_the_plan_node.parents.clear()
                    follow_the_plan_node.metadata["plan_step_number"] = current_step_number
                    node._add_parent(follow_the_plan_node)

                # Update the step status to in_progress
                plan_status[current_step_number] = "in_progress"

        # ============================================
        # BRANCH 2: Add tools information to planning node
        # This provides the planning node with available tools
        # ============================================
        elif (
            not node.invoked
            and isinstance(node, PlanningGenNode)
            and isinstance(network, PlanningNetworkNode)
            and node.outputs is None
            and not network.get_children(node)
        ):
            # Add the system prompt with available tools to the planning node
            # This enables the planning node to create tool-aware plans
            planning_network_node, multi_agent_network_node = self._find_planning_and_multi_agent_networks(network)

            if multi_agent_network_node is None:
                return

            system_message = self._get_tools_system_message(multi_agent_network_node)
            node.inputs.append(RunnableSystemAppend(system_message=system_message))

    def _get_plan_metadata(self, network):
        """
        Retrieve plan and status from network metadata.

        Returns:
            tuple: (plan_status, current_plan) or None if not available
        """
        plan_status = network.metadata.get("plan_status", {})
        current_plan = network.metadata.get("current_plan", {})

        if not plan_status or not current_plan:
            return None

        return plan_status, current_plan

    def _get_next_pending_step(self, plan_status, current_plan):
        """
        Find the next step that needs to be executed.

        Args:
            plan_status: Dictionary mapping step numbers to their status
            current_plan: The current execution plan

        Returns:
            tuple: (step_number, step_data) or None if no pending steps
        """
        # Find the first pending step number
        current_step_number = next(
            (step_num for step_num in plan_status.keys() if plan_status[step_num] == "pending"), None
        )

        if current_step_number is None:
            return None

        # Find the corresponding step data
        current_step = next(
            (step for step in current_plan["steps"] if step["step_number"] == current_step_number), None
        )

        if current_step is None:
            return None

        return current_step_number, current_step

    def _build_step_instruction_message(self, step_number, step_data):
        """
        Build the instruction message for a plan step using the global format template.

        Args:
            step_number: The step number
            step_data: Dictionary containing step title and details

        Returns:
            str: Formatted instruction message
        """
        # Format step details if available
        step_details = ""
        step_title = step_data["title"]
        if step_data.get("details"):
            # Add newline before details and join them with newlines
            step_details = "\n" + "\n".join(step_data["details"])

        # Format the complete message using the template
        message = PLAN_INSTRUCTION_FORMAT.format(
            step_number=step_number, step_title=step_title, step_details=step_details
        )

        return message

    def _mark_step_completed(self, step_number: int):
        """
        Mark a step as completed and update dependency graph.

        This enables parallel execution by:
        1. Updating plan_status to "completed"
        2. Updating dependency graph (reduces in-degree for dependent steps)
        3. Potentially making new steps ready for execution

        Args:
            step_number: The step number that completed
        """
        if self.plan_status.get(step_number) == "completed":
            return  # Already marked as completed

        # Update plan status
        self.plan_status[step_number] = "completed"
        logger.info(f"Step {step_number} marked as completed")

        # Update dependency graph if available
        if self.dependency_graph:
            self.dependency_graph.mark_completed(step_number)

            # Log newly ready steps
            ready_steps = self.dependency_graph.get_ready_steps()
            if ready_steps:
                logger.info(f"Steps now ready for execution: {ready_steps}")

    def _handle_step_completion(self, network, node):
        """
        Handle step completion - called from on_post_invoke_async.

        Detects when a step finishes executing and updates tracking.

        Args:
            network: The network where step executed
            node: The node that completed
        """
        # Check if this node corresponds to a plan step
        step_number = node.metadata.get("plan_step_number")

        if step_number is None:
            # Try to infer from plan_status (currently in_progress)
            for step_num, status in self.plan_status.items():
                if status == "in_progress":
                    # This might be the completing step
                    # In sequential mode, there's only one in_progress at a time
                    # In parallel mode, we need better tracking
                    # For now, we'll rely on explicit metadata
                    pass

        if step_number and self.plan_status.get(step_number) == "in_progress":
            self._mark_step_completed(step_number)

    def _is_valid_plan(self, content: str) -> bool:
        """
        Check if the content contains a valid plan.

        Args:
            content: The content to check

        Returns:
            bool: True if content contains a valid plan
        """
        # Check if content contains a plan header and at least one step
        plan_header = re.search(r"PLAN:\s*(.+?)(?:\n|$)", content)
        steps = re.findall(r"Step \d+:", content)
        return plan_header is not None and len(steps) > 0

    def _extract_plan(self, content: str) -> Dict[str, Any]:
        """
        Extract and structure the plan from the content.

        Enhanced to extract dependencies for parallel execution support.

        Args:
            content: The content containing the plan

        Returns:
            dict: Structured plan with the following format:
                {
                    "title": "Plan title",
                    "steps": [
                        {
                            "step_number": 1,
                            "title": "Step title",
                            "details": ["detail1", "detail2", ...],
                            "dependencies": [2, 3],  # NEW: List of prerequisite step numbers
                            "step_type": "action" | "planning_review"  # NEW: Type of step
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

            # Determine step type based on title
            step_type = "action"  # default
            if "Planning Review" in title or "planning review" in title.lower():
                step_type = "planning_review"

            # Extract dependencies
            dependencies = []
            dependency_found = False
            for line in lines:
                line_stripped = line.strip()
                # Look for "Dependencies: ..." pattern
                dep_match = re.match(
                    r"Dependencies?:\s*(.+?)$",
                    line_stripped,
                    re.IGNORECASE
                )
                if dep_match:
                    dep_str = dep_match.group(1).strip()
                    # Handle "None" case
                    if dep_str.lower() not in ["none", "n/a", ""]:
                        # Parse comma-separated numbers
                        try:
                            dependencies = [
                                int(d.strip())
                                for d in dep_str.replace("and", ",").split(",")
                                if d.strip().isdigit()
                            ]
                        except ValueError:
                            logger.warning(f"Could not parse dependencies for step {step_number}: {dep_str}")
                    dependency_found = True
                    break

            # Extract details from bullet points (excluding the Dependencies line)
            details = []
            for line in lines[1:]:  # Skip first line (title)
                line_stripped = line.strip()
                if line_stripped.startswith("-") and not re.match(r"Dependencies?:", line_stripped, re.IGNORECASE):
                    # Remove leading dash and strip
                    detail = line_stripped[1:].strip()
                    # Skip Planning Review metadata lines
                    if not any(detail.startswith(prefix) for prefix in [
                        "Review focus:", "Previous steps:", "Decision points:",
                        "Potential outcomes:", "*"
                    ]):
                        details.append(detail)

            step_dict = {
                "step_number": step_number,
                "title": title,
                "step_type": step_type,
                "details": details,
                "dependencies": dependencies
            }

            steps.append(step_dict)

        # Sort steps by step number to ensure proper ordering
        steps.sort(key=lambda x: x["step_number"])

        return {"title": plan_title, "steps": steps}
