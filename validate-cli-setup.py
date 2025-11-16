#!/usr/bin/env python3
"""
VALIDATION SCRIPT: CLI Setup Check

This script validates that the LC Agent CLI environment is properly set up
for parallel execution testing, WITHOUT requiring an NVIDIA API key.

It checks:
1. Python modules can be imported
2. Parallel execution code is accessible
3. NVIDIA model registration works
4. Basic functionality is intact

Usage:
    python validate-cli-setup.py
"""

import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color


def print_success(msg):
    print(f"{GREEN}✓{NC} {msg}")


def print_error(msg):
    print(f"{RED}✗{NC} {msg}")


def print_warning(msg):
    print(f"{YELLOW}⚠{NC} {msg}")


def print_header(msg):
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)
    print()


def main():
    print_header("LC Agent CLI Setup Validation")

    all_checks_passed = True

    # Add source directories to path
    repo_root = Path(__file__).parent
    lc_agent_src = repo_root / "source" / "modules" / "lc_agent" / "src"
    lc_agent_cli_src = repo_root / "source" / "modules" / "lc_agent_cli" / "src"
    sys.path.insert(0, str(lc_agent_src))
    sys.path.insert(0, str(lc_agent_cli_src))

    # Check 1: Import lc_agent
    print("Check 1: Importing lc_agent module...")
    try:
        import lc_agent
        print_success(f"lc_agent imported from {lc_agent_src}")
    except ImportError as e:
        print_error(f"Failed to import lc_agent: {e}")
        all_checks_passed = False
        return 1

    # Check 2: Import core classes
    print("\nCheck 2: Importing core classes...")
    try:
        from lc_agent.runnable_node import RunnableNode
        from lc_agent.runnable_network import RunnableNetwork
        from lc_agent.node_factory import get_node_factory
        print_success("RunnableNode imported")
        print_success("RunnableNetwork imported")
        print_success("NodeFactory imported")
    except ImportError as e:
        print_error(f"Failed to import core classes: {e}")
        all_checks_passed = False
        return 1

    # Check 3: Verify parallel execution method exists
    print("\nCheck 3: Verifying parallel execution implementation...")
    try:
        # Check if the method exists
        if hasattr(RunnableNode, '_group_by_dependency_level'):
            print_success("_group_by_dependency_level method found")
        else:
            print_error("_group_by_dependency_level method NOT found!")
            all_checks_passed = False

        if hasattr(RunnableNode, '_aprocess_parents'):
            print_success("_aprocess_parents method found")

            # Check if it uses asyncio.gather (inspect the code)
            import inspect
            source = inspect.getsource(RunnableNode._aprocess_parents)
            if 'asyncio.gather' in source:
                print_success("_aprocess_parents uses asyncio.gather for parallelism")
            else:
                print_warning("_aprocess_parents may not be using asyncio.gather")

        else:
            print_error("_aprocess_parents method NOT found!")
            all_checks_passed = False
    except Exception as e:
        print_error(f"Failed to verify parallel execution: {e}")
        all_checks_passed = False

    # Check 4: Import lc_agent_cli
    print("\nCheck 4: Importing lc_agent_cli module...")
    try:
        import lc_agent_cli
        print_success(f"lc_agent_cli imported from {lc_agent_cli_src}")
    except ImportError as e:
        print_warning(f"Could not import lc_agent_cli: {e}")
        print_warning("This is optional - you can still use ChatNVCF directly")

    # Check 5: Test NVIDIA model registration
    print("\nCheck 5: Testing NVIDIA model registration...")
    try:
        from lc_agent_cli import register_all
        register_all()
        print_success("NVIDIA models registered via lc_agent_cli.register_all()")

        # Check if models are in factory
        factory = get_node_factory()
        registered_models = [k for k in factory._registry.keys() if 'llama' in k.lower() or 'gpt' in k.lower()]
        if registered_models:
            print_success(f"Found {len(registered_models)} registered model(s)")
            for model in registered_models[:3]:  # Show first 3
                print(f"    - {model}")
        else:
            print_warning("No models found in registry")
    except Exception as e:
        print_warning(f"Model registration skipped: {e}")
        print_warning("You can still use ChatNVCF directly")

    # Check 6: Verify ChatNVCF is available
    print("\nCheck 6: Verifying ChatNVCF model...")
    try:
        from lc_agent.chat_models.chat_nvcf import ChatNVCF
        print_success("ChatNVCF class imported")
    except ImportError as e:
        print_error(f"Failed to import ChatNVCF: {e}")
        all_checks_passed = False

    # Check 7: Test file locations
    print("\nCheck 7: Verifying test files...")
    test_files = [
        "test_dependency_grouping.py",
        "test_parallel_live.py",
        "test_real_llm_parallel.py",
        "test_cli_parallel_execution.py",
        "run-cli-parallel-test.sh"
    ]

    for test_file in test_files:
        file_path = repo_root / test_file
        if file_path.exists():
            print_success(f"{test_file} found")
        else:
            print_warning(f"{test_file} not found")

    # Check 8: Verify modified runnable_node.py
    print("\nCheck 8: Verifying parallel execution changes in runnable_node.py...")
    try:
        runnable_node_file = lc_agent_src / "lc_agent" / "runnable_node.py"
        with open(runnable_node_file, 'r') as f:
            content = f.read()

        if 'import asyncio' in content:
            print_success("asyncio import found")
        else:
            print_error("asyncio import NOT found!")
            all_checks_passed = False

        if '_group_by_dependency_level' in content:
            print_success("_group_by_dependency_level implementation found")
        else:
            print_error("_group_by_dependency_level NOT found!")
            all_checks_passed = False

        if 'asyncio.gather(*tasks)' in content or 'asyncio.gather(' in content:
            print_success("asyncio.gather usage found in source")
        else:
            print_error("asyncio.gather NOT found in source!")
            all_checks_passed = False

    except Exception as e:
        print_error(f"Failed to verify source file: {e}")
        all_checks_passed = False

    # Final summary
    print_header("VALIDATION SUMMARY")

    if all_checks_passed:
        print(f"{GREEN}✓✓✓ ALL CHECKS PASSED! ✓✓✓{NC}")
        print()
        print("Your LC Agent CLI environment is properly set up for")
        print("parallel execution testing with real LLMs.")
        print()
        print("Next steps:")
        print("  1. Set your NVIDIA_API_KEY:")
        print("     export NVIDIA_API_KEY='your_key_here'")
        print()
        print("  2. Run the full test:")
        print("     ./run-cli-parallel-test.sh")
        print()
        print("  OR run directly:")
        print("     python test_cli_parallel_execution.py")
        print()
        return 0
    else:
        print(f"{RED}✗ SOME CHECKS FAILED{NC}")
        print()
        print("Please review the errors above and ensure:")
        print("  - Parallel execution changes are merged")
        print("  - Source files are in the correct locations")
        print("  - Python path is set up correctly")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
