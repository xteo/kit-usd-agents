# Parallel Execution Testing Guide

This document explains how to test and validate the parallel execution implementation in LC Agent using the CLI infrastructure and real LLM API calls.

## Overview

The parallel execution implementation enables independent graph branches to execute concurrently, achieving 2-5x speedup for multi-branch workflows. This guide covers testing with:

1. **Unit tests** (no API key required)
2. **Live demonstration** (no API key required)
3. **Real LLM tests** (NVIDIA API key required)
4. **CLI-based tests** (NVIDIA API key required)

## Quick Start

### 1. Validate Setup (No API Key Required)

First, verify that everything is properly set up:

```bash
python3 validate-cli-setup.py
```

This will check:
- ✓ Module imports work correctly
- ✓ Parallel execution code is in place
- ✓ All test files are present
- ✓ Source modifications are correct

### 2. Run Unit Tests (No API Key Required)

Test the dependency grouping algorithm:

```bash
python3 test_dependency_grouping.py
```

**What it tests:**
- Diamond graph grouping (A → B,C → D)
- Wide graph grouping (A → B,C,D,E → F)
- Linear graph grouping (A → B → C → D)
- Complex graph grouping (multiple patterns)

**Expected output:**
```
All dependency grouping tests passed!
✓ Diamond graph: 3 levels
✓ Wide graph: 3 levels
✓ Linear graph: 4 levels
✓ Complex graph: 4 levels
```

### 3. Run Live Demonstration (No API Key Required)

See parallel execution in action with async sleeps:

```bash
python3 test_parallel_live.py
```

**What it tests:**
- Diamond graph: A(0.1s) → B,C(1.0s each) → D
- Wide graph: A(0.1s) → B,C,D,E(0.5s each) → F

**Expected output:**
```
✓✓✓ B and C EXECUTED IN PARALLEL! ✓✓✓
Expected ~1.1s, actual 1.102s
Speedup vs sequential: 1.91x
```

**What to look for:**
- Timestamps show B and C starting at the SAME time
- Total execution time matches parallel expectation (~1.1s, not 2.1s)
- Measured speedup of ~2x

## Real LLM Testing (Requires API Key)

### Getting an NVIDIA API Key

1. Go to [https://build.nvidia.com/](https://build.nvidia.com/)
2. Sign in with your NVIDIA account (free)
3. Navigate to any model (e.g., `openai/gpt-oss-120b`)
4. Click **"Get API Key"**
5. Copy the API key

### Setting the API Key

**Linux/Mac:**
```bash
export NVIDIA_API_KEY='nvapi-your-key-here'
```

**Windows PowerShell:**
```powershell
$env:NVIDIA_API_KEY="nvapi-your-key-here"
```

**Windows CMD:**
```cmd
set NVIDIA_API_KEY=nvapi-your-key-here
```

### Test with Real LLMs (Original Test)

Run the original real LLM test:

```bash
python3 test_real_llm_parallel.py
```

**What it does:**
- Makes 3 actual API calls to NVIDIA NIM
- Node B: "Explain the history of AI" (concurrent)
- Node C: "Explain the future of AI" (concurrent)
- Node D: "Summarize both perspectives"

**Expected output:**
```
[12345.123] B-HistoryOfAI - Starting LLM call...
[12345.123] C-FutureOfAI - Starting LLM call...
[12347.456] B-HistoryOfAI - FINISHED (took 2.33s)
[12347.512] C-FutureOfAI - FINISHED (took 2.39s)
[12349.234] D-Summarize - FINISHED (took 1.72s)

✓✓✓ SUCCESS! REAL LLMs ran in PARALLEL! ✓✓✓
Speedup: 1.67x faster than sequential
```

### Test with CLI Infrastructure (Recommended)

Run the CLI-based test (uses the full CLI stack):

```bash
./run-cli-parallel-test.sh
```

Or directly:

```bash
python3 test_cli_parallel_execution.py
```

**What it does:**
- Registers NVIDIA models via `lc_agent_cli` (gpt-120b, llama-maverick)
- Uses **gpt-120b** (openai/gpt-oss-120b) as default model
- Creates diamond graph with real LLM nodes
- Node B: "What is artificial intelligence?" (concurrent)
- Node C: "What is machine learning?" (concurrent)
- Node D: "Combine these concepts"
- Measures parallel execution timing
- Validates speedup vs sequential execution

**Expected output:**
```
✓ NVIDIA_API_KEY found: nvapi-xxxx...yyyy
✓ NVIDIA models registered successfully

[12345.123] B-WhatIsAI - Starting LLM call...
[12345.124] C-WhatIsML - Starting LLM call...
[12347.456] B-WhatIsAI - FINISHED (took 2.33s)
[12347.501] C-WhatIsML - FINISHED (took 2.38s)
[12349.123] D-Summarize - FINISHED (took 1.62s)

✓✓✓ B and C EXECUTED IN PARALLEL! ✓✓✓
Overlap: 2.33s

Timing analysis:
  If SEQUENTIAL: ~6.33s (B + C + D)
  If PARALLEL:   ~4.00s (max(B,C) + D)
  ACTUAL:        4.12s

✓✓✓ SUCCESS! REAL LLMs ran in PARALLEL via CLI! ✓✓✓
Speedup: 1.54x faster than sequential
```

**What to look for:**
- B and C start at nearly the same timestamp
- Overlap duration is significant (>80% of call duration)
- Total time is much closer to parallel prediction than sequential
- Speedup factor is > 1.3x

## Understanding the Results

### How to Tell if It's Really Running in Parallel

1. **Timestamp Overlap**: Independent nodes should START at the same time
   ```
   ✓ GOOD:  [123.456] B STARTED
            [123.456] C STARTED  ← Same timestamp!

   ✗ BAD:   [123.456] B STARTED
            [125.678] C STARTED  ← Started after B finished!
   ```

2. **Total Execution Time**: Should match parallel prediction
   ```
   ✓ GOOD:  Sequential: 6.0s, Parallel: 4.0s, ACTUAL: 4.1s
   ✗ BAD:   Sequential: 6.0s, Parallel: 4.0s, ACTUAL: 5.9s
   ```

3. **Speedup Factor**: Should be > 1.3x for diamond graphs
   ```
   ✓ GOOD:  Speedup: 1.54x
   ✗ BAD:   Speedup: 1.02x
   ```

### Expected Performance

| Graph Type | Branches | Sequential | Parallel | Speedup |
|------------|----------|------------|----------|---------|
| Diamond    | 2        | 6.0s       | 4.0s     | 1.5x    |
| Wide (4)   | 4        | 8.0s       | 2.5s     | 3.2x    |
| Wide (8)   | 8        | 14.0s      | 2.5s     | 5.6x    |

*Times are approximate and depend on LLM API latency*

## Troubleshooting

### "NVIDIA_API_KEY environment variable not set"

**Solution:** Set the API key as described in "Setting the API Key" above.

### "No module named 'langchain_nvidia_ai_endpoints'"

**Solution:** This is expected if you're using ChatNVCF directly. The tests will fall back automatically.

### "AttributeError: install_layout"

**Solution:** This is a setuptools compatibility issue. The tests are designed to work without full installation by using PYTHONPATH. Use the provided scripts which handle this automatically.

### "Tests show sequential execution instead of parallel"

**Solution:**
1. Verify you're on the correct branch: `claude/read-it-all-01K1MDmfSBFKYvYVz4dAMHDf`
2. Run validation: `python3 validate-cli-setup.py`
3. Check that asyncio.gather is present in the output
4. Ensure you merged the latest changes

### "API calls failing with 401 Unauthorized"

**Solution:**
1. Verify your API key is correct
2. Check that it starts with `nvapi-`
3. Get a fresh key from https://build.nvidia.com/
4. Ensure the key is properly exported in your environment

## Test Files Reference

| File | Purpose | API Key Required |
|------|---------|------------------|
| `validate-cli-setup.py` | Validate environment setup | No |
| `test_dependency_grouping.py` | Unit test grouping algorithm | No |
| `test_parallel_live.py` | Demo with async sleeps | No |
| `test_real_llm_parallel.py` | Test with real LLM calls | Yes |
| `test_cli_parallel_execution.py` | Test with CLI infrastructure | Yes |
| `run-cli-parallel-test.sh` | Wrapper script for CLI test | Yes |

## Architecture Notes

### How Parallel Execution Works

1. **Dependency Analysis**: `_group_by_dependency_level()` analyzes the graph and groups nodes by their distance from root nodes

2. **Level-by-Level Execution**: Each level is executed as a batch using `asyncio.gather()`

3. **Concurrent API Calls**: Within each level, all nodes run concurrently (real I/O parallelism)

### Code Locations

- **Core implementation**: `source/modules/lc_agent/src/lc_agent/runnable_node.py`
  - Lines 1003-1076: `_group_by_dependency_level()`
  - Lines 1078-1117: `_aprocess_parents()` (uses asyncio.gather)

- **CLI integration**: `source/modules/lc_agent_cli/src/lc_agent_cli/`
  - `register_models.py`: NVIDIA model registration
  - `cli.py`: CLI implementation

### Changes Summary

**158 lines of code changed:**
- 74 lines: New `_group_by_dependency_level()` method
- 40 lines: Modified `_aprocess_parents()` method
- 3 lines: Added imports
- 41 lines: Documentation and type hints

## Next Steps

After validating that parallel execution works:

1. **Integrate into your workflow**: Use the pattern in your own agents
2. **Measure real-world impact**: Test with your specific use cases
3. **Monitor performance**: Use the built-in profiling tools
4. **Optimize further**: Tune concurrency limits if needed

## Support

For issues or questions:
- Review the comprehensive report: `PARALLEL_EXECUTION_COMPLETE_REPORT.md`
- Check the visual guide: `PARALLEL_EXECUTION_VISUAL_GUIDE.md`
- Examine test output for detailed timing information

---

**Last Updated**: 2025-11-16
**Branch**: `claude/read-it-all-01K1MDmfSBFKYvYVz4dAMHDf`
**Status**: Ready for testing with real LLMs
