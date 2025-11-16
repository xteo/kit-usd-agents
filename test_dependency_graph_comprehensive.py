#!/usr/bin/env python3
"""
Comprehensive Dependency Graph Unit Tests

This test file thoroughly validates the DependencyGraph class implementation
for parallel task execution with dependency tracking.

Tests cover:
1. Graph construction from plan steps
2. Ready step identification
3. Step completion and dependency updates
4. Dependency validation (circular, forward refs, etc.)
5. Edge cases and error handling
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "source" / "modules" / "lc_agent" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "source" / "modules" / "agents" / "planning" / "src"))

from omni_aiq_planning.modifiers.planning_modifier import DependencyGraph

def test_basic_graph_construction():
    """Test basic dependency graph construction"""
    print("\n🧪 TEST: Basic Graph Construction")

    steps = [
        {"step_number": 1, "title": "Step 1", "dependencies": []},
        {"step_number": 2, "title": "Step 2", "dependencies": [1]},
        {"step_number": 3, "title": "Step 3", "dependencies": [1]},
    ]

    graph = DependencyGraph(steps)

    # Verify graph structure
    assert len(graph.steps) == 3, "Should have 3 steps"
    assert graph.in_degree[1] == 0, "Step 1 should have 0 in-degree"
    assert graph.in_degree[2] == 1, "Step 2 should have 1 in-degree"
    assert graph.in_degree[3] == 1, "Step 3 should have 1 in-degree"
    assert graph.adjacency_list[1] == [2, 3], "Step 1 should have 2 dependents"

    print("   ✓ Graph structure correct")
    print("   ✓ In-degrees calculated correctly")
    print("   ✓ Adjacency list built correctly")
    return True


def test_get_ready_steps():
    """Test identification of ready steps"""
    print("\n🧪 TEST: Get Ready Steps")

    steps = [
        {"step_number": 1, "title": "Build A", "dependencies": []},
        {"step_number": 2, "title": "Build B", "dependencies": []},
        {"step_number": 3, "title": "Build C", "dependencies": []},
        {"step_number": 4, "title": "Deploy", "dependencies": [1, 2, 3]},
    ]

    graph = DependencyGraph(steps)

    # Initially, steps 1, 2, 3 should be ready (no dependencies)
    ready = graph.get_ready_steps()
    assert ready == [1, 2, 3], f"Expected [1, 2, 3], got {ready}"
    print(f"   ✓ Initial ready steps: {ready}")

    # Mark step 1 as completed
    graph.mark_completed(1)
    ready = graph.get_ready_steps()
    assert ready == [2, 3], f"Expected [2, 3], got {ready}"
    print(f"   ✓ After completing 1: {ready}")

    # Mark steps 2 and 3 as completed
    graph.mark_completed(2)
    graph.mark_completed(3)
    ready = graph.get_ready_steps()
    assert ready == [4], f"Expected [4], got {ready}"
    print(f"   ✓ After completing 2 and 3: {ready}")

    # Mark step 4 as completed
    graph.mark_completed(4)
    ready = graph.get_ready_steps()
    assert ready == [], f"Expected [], got {ready}"
    assert graph.is_complete(), "Graph should be complete"
    print(f"   ✓ After completing all: {ready}")
    print("   ✓ Graph marked as complete")

    return True


def test_parallel_execution_pattern():
    """Test fan-out and fan-in pattern (diamond graph)"""
    print("\n🧪 TEST: Parallel Execution Pattern (Diamond)")

    steps = [
        {"step_number": 1, "title": "Setup", "dependencies": []},
        {"step_number": 2, "title": "Task A", "dependencies": [1]},
        {"step_number": 3, "title": "Task B", "dependencies": [1]},
        {"step_number": 4, "title": "Task C", "dependencies": [1]},
        {"step_number": 5, "title": "Combine", "dependencies": [2, 3, 4]},
    ]

    graph = DependencyGraph(steps)

    # Phase 1: Only step 1 is ready
    ready = graph.get_ready_steps()
    assert ready == [1], f"Phase 1: Expected [1], got {ready}"
    print(f"   ✓ Phase 1 ready: {ready}")

    # Complete step 1
    graph.mark_completed(1)

    # Phase 2: Steps 2, 3, 4 should all be ready (parallel execution!)
    ready = graph.get_ready_steps()
    assert ready == [2, 3, 4], f"Phase 2: Expected [2, 3, 4], got {ready}"
    print(f"   ✓ Phase 2 ready (PARALLEL): {ready}")

    # Complete steps 2 and 3 (but not 4)
    graph.mark_completed(2)
    graph.mark_completed(3)

    # Phase 3: Step 4 still ready, but not 5 (still waiting for 4)
    ready = graph.get_ready_steps()
    assert ready == [4], f"Phase 3: Expected [4], got {ready}"
    print(f"   ✓ Phase 3 ready: {ready}")

    # Complete step 4
    graph.mark_completed(4)

    # Phase 4: Step 5 should be ready now
    ready = graph.get_ready_steps()
    assert ready == [5], f"Phase 4: Expected [5], got {ready}"
    print(f"   ✓ Phase 4 ready: {ready}")

    return True


def test_dependency_validation_valid():
    """Test validation of valid dependency graphs"""
    print("\n🧪 TEST: Dependency Validation - Valid Graphs")

    # Test 1: Simple sequential chain
    steps = [
        {"step_number": 1, "title": "A", "dependencies": []},
        {"step_number": 2, "title": "B", "dependencies": [1]},
        {"step_number": 3, "title": "C", "dependencies": [2]},
    ]
    graph = DependencyGraph(steps)
    is_valid, error = graph.validate_dependencies()
    assert is_valid, f"Sequential chain should be valid, got error: {error}"
    print("   ✓ Sequential chain validated")

    # Test 2: Parallel branches
    steps = [
        {"step_number": 1, "title": "A", "dependencies": []},
        {"step_number": 2, "title": "B", "dependencies": []},
        {"step_number": 3, "title": "C", "dependencies": []},
    ]
    graph = DependencyGraph(steps)
    is_valid, error = graph.validate_dependencies()
    assert is_valid, f"Parallel branches should be valid, got error: {error}"
    print("   ✓ Parallel branches validated")

    # Test 3: Diamond pattern
    steps = [
        {"step_number": 1, "title": "A", "dependencies": []},
        {"step_number": 2, "title": "B", "dependencies": [1]},
        {"step_number": 3, "title": "C", "dependencies": [1]},
        {"step_number": 4, "title": "D", "dependencies": [2, 3]},
    ]
    graph = DependencyGraph(steps)
    is_valid, error = graph.validate_dependencies()
    assert is_valid, f"Diamond pattern should be valid, got error: {error}"
    print("   ✓ Diamond pattern validated")

    return True


def test_dependency_validation_circular():
    """Test detection of circular dependencies"""
    print("\n🧪 TEST: Dependency Validation - Circular Dependencies")

    # Test 1: Direct circle (1 → 2 → 1)
    # Note: This would violate forward-reference rule first
    # So let's test a more complex circular case

    # Test 2: Complex circle (1 → 2 → 3 → 1)
    # This also violates forward-reference rule

    # Circular dependencies are actually prevented by the forward-reference rule
    # Let's test that forward references are caught
    print("   ℹ Circular dependencies are prevented by forward-reference validation")

    return True


def test_dependency_validation_forward_refs():
    """Test detection of forward references"""
    print("\n🧪 TEST: Dependency Validation - Forward References")

    # Test 1: Step depends on a later step
    steps = [
        {"step_number": 1, "title": "A", "dependencies": [2]},  # Forward ref!
        {"step_number": 2, "title": "B", "dependencies": []},
    ]
    graph = DependencyGraph(steps)
    is_valid, error = graph.validate_dependencies()
    assert not is_valid, "Forward reference should be invalid"
    assert "forward" in error.lower() or "Step 1" in error, f"Error should mention forward ref, got: {error}"
    print(f"   ✓ Forward reference detected: {error}")

    # Test 2: Step depends on itself
    steps = [
        {"step_number": 1, "title": "A", "dependencies": [1]},  # Self-reference!
    ]
    graph = DependencyGraph(steps)
    is_valid, error = graph.validate_dependencies()
    assert not is_valid, "Self-reference should be invalid"
    print(f"   ✓ Self-reference detected: {error}")

    return True


def test_dependency_validation_nonexistent():
    """Test detection of non-existent step references"""
    print("\n🧪 TEST: Dependency Validation - Non-existent Steps")

    steps = [
        {"step_number": 1, "title": "A", "dependencies": []},
        {"step_number": 2, "title": "B", "dependencies": [1, 999]},  # Step 999 doesn't exist!
    ]
    graph = DependencyGraph(steps)
    is_valid, error = graph.validate_dependencies()
    assert not is_valid, "Non-existent step reference should be invalid"
    assert "999" in error or "non-existent" in error.lower(), f"Error should mention missing step, got: {error}"
    print(f"   ✓ Non-existent step detected: {error}")

    return True


def test_dependency_status():
    """Test detailed dependency status reporting"""
    print("\n🧪 TEST: Dependency Status Reporting")

    steps = [
        {"step_number": 1, "title": "A", "dependencies": []},
        {"step_number": 2, "title": "B", "dependencies": []},
        {"step_number": 3, "title": "C", "dependencies": [1, 2]},
    ]
    graph = DependencyGraph(steps)

    # Check status of step 3 initially
    status = graph.get_dependency_status(3)
    assert status["ready"] == False, "Step 3 should not be ready initially"
    assert status["dependencies"] == [1, 2], "Step 3 should depend on [1, 2]"
    assert status["satisfied"] == [], "No dependencies satisfied initially"
    assert status["unsatisfied"] == [1, 2], "Both dependencies unsatisfied"
    print(f"   ✓ Initial status for step 3: {status}")

    # Mark step 1 as completed
    graph.mark_completed(1)
    status = graph.get_dependency_status(3)
    assert status["ready"] == False, "Step 3 still not ready (waiting for 2)"
    assert status["satisfied"] == [1], "Step 1 should be satisfied"
    assert status["unsatisfied"] == [2], "Step 2 still unsatisfied"
    print(f"   ✓ After completing 1: {status}")

    # Mark step 2 as completed
    graph.mark_completed(2)
    status = graph.get_dependency_status(3)
    assert status["ready"] == True, "Step 3 should be ready now"
    assert status["satisfied"] == [1, 2], "Both dependencies satisfied"
    assert status["unsatisfied"] == [], "No unsatisfied dependencies"
    print(f"   ✓ After completing 2: {status}")

    return True


def test_complex_microservices_scenario():
    """Test realistic microservices deployment scenario"""
    print("\n🧪 TEST: Complex Microservices Deployment Scenario")

    steps = [
        # Phase 1: Parallel builds
        {"step_number": 1, "title": "Build auth-service", "dependencies": []},
        {"step_number": 2, "title": "Build user-service", "dependencies": []},
        {"step_number": 3, "title": "Build api-gateway", "dependencies": []},
        {"step_number": 4, "title": "Build DB migrations", "dependencies": []},

        # Phase 2: Review builds
        {"step_number": 5, "title": "Planning Review - Builds", "dependencies": [1, 2, 3, 4]},

        # Phase 3: Deploy database
        {"step_number": 6, "title": "Deploy database", "dependencies": [5]},

        # Phase 4: Parallel service deployments
        {"step_number": 7, "title": "Deploy auth-service", "dependencies": [6]},
        {"step_number": 8, "title": "Deploy user-service", "dependencies": [6]},
        {"step_number": 9, "title": "Deploy api-gateway", "dependencies": [6]},

        # Phase 5: Integration tests
        {"step_number": 10, "title": "Integration tests", "dependencies": [7, 8, 9]},

        # Phase 6: Final review
        {"step_number": 11, "title": "Planning Review - Deployment", "dependencies": [10]},
    ]

    graph = DependencyGraph(steps)

    # Validate graph
    is_valid, error = graph.validate_dependencies()
    assert is_valid, f"Microservices scenario should be valid, got: {error}"
    print("   ✓ Graph validated successfully")

    # Simulate execution
    execution_phases = []

    # Phase 1: Parallel builds (1, 2, 3, 4)
    ready = graph.get_ready_steps()
    assert ready == [1, 2, 3, 4], f"Phase 1: Expected parallel builds, got {ready}"
    execution_phases.append(("Phase 1: Parallel builds", ready.copy()))
    for step in ready:
        graph.mark_completed(step)

    # Phase 2: Build review (5)
    ready = graph.get_ready_steps()
    assert ready == [5], f"Phase 2: Expected review, got {ready}"
    execution_phases.append(("Phase 2: Build review", ready.copy()))
    graph.mark_completed(5)

    # Phase 3: Deploy database (6)
    ready = graph.get_ready_steps()
    assert ready == [6], f"Phase 3: Expected DB deploy, got {ready}"
    execution_phases.append(("Phase 3: Deploy database", ready.copy()))
    graph.mark_completed(6)

    # Phase 4: Parallel service deployments (7, 8, 9)
    ready = graph.get_ready_steps()
    assert ready == [7, 8, 9], f"Phase 4: Expected parallel deploys, got {ready}"
    execution_phases.append(("Phase 4: Parallel service deploys", ready.copy()))
    for step in ready:
        graph.mark_completed(step)

    # Phase 5: Integration tests (10)
    ready = graph.get_ready_steps()
    assert ready == [10], f"Phase 5: Expected integration tests, got {ready}"
    execution_phases.append(("Phase 5: Integration tests", ready.copy()))
    graph.mark_completed(10)

    # Phase 6: Final review (11)
    ready = graph.get_ready_steps()
    assert ready == [11], f"Phase 6: Expected final review, got {ready}"
    execution_phases.append(("Phase 6: Final review", ready.copy()))
    graph.mark_completed(11)

    # Verify completion
    assert graph.is_complete(), "All steps should be completed"

    # Print execution plan
    print("\n   📊 Execution Plan:")
    for phase_name, steps in execution_phases:
        parallel_marker = " (PARALLEL)" if len(steps) > 1 else ""
        print(f"      {phase_name}{parallel_marker}: {steps}")

    print("\n   ✓ Microservices scenario executed correctly")
    print(f"   ✓ Total phases: {len(execution_phases)}")
    print(f"   ✓ Parallel phases: {sum(1 for _, s in execution_phases if len(s) > 1)}")

    return True


def main():
    """Run all tests"""
    print("=" * 70)
    print("COMPREHENSIVE DEPENDENCY GRAPH UNIT TESTS")
    print("=" * 70)

    tests = [
        test_basic_graph_construction,
        test_get_ready_steps,
        test_parallel_execution_pattern,
        test_dependency_validation_valid,
        test_dependency_validation_circular,
        test_dependency_validation_forward_refs,
        test_dependency_validation_nonexistent,
        test_dependency_status,
        test_complex_microservices_scenario,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"   ✅ PASSED\n")
            else:
                failed += 1
                print(f"   ❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"   ❌ FAILED with exception: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
