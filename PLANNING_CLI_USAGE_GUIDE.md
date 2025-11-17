# Planning CLI Usage Guide

## Overview

The Planning CLI demonstrates the **parallel task execution with dependency graphs** feature. It shows how the planning agent creates plans with explicit dependencies and executes them with automatic parallelization.

---

## Available CLIs

### 1. Demo Planning CLI (Recommended for Testing)

**File**: `demo-planning-cli.py`

**Purpose**: Demonstrates parallel planning without requiring network access or API keys.

**Features**:
- Pre-generated sample plans with dependencies
- Visual plan display showing parallel opportunities
- Real-time execution simulation
- Performance metrics and speedup calculations
- No API key required

**Usage**:
```bash
# View a microservices deployment plan
python demo-planning-cli.py --scenario microservices

# View a web application deployment plan
python demo-planning-cli.py --scenario webapp

# View a data pipeline plan
python demo-planning-cli.py --scenario data-pipeline

# Execute the plan simulation (interactive)
python demo-planning-cli.py --scenario microservices --execute
```

### 2. Real Planning CLI (Requires NVIDIA API Key)

**File**: `lc-agent-planning.py`

**Purpose**: Uses real LLM to generate plans with dependencies based on your queries.

**Features**:
- Real-time plan generation using NVIDIA NIM
- Custom queries for any scenario
- Actual LLM-powered planning
- Requires NVIDIA API key and network access

**Usage**:
```bash
# Set your API key first
export NVIDIA_API_KEY="nvapi-your-key-here"

# Interactive mode
python lc-agent-planning.py

# Single query mode
python lc-agent-planning.py --query "Deploy a microservices app with 3 services"

# Use different model
python lc-agent-planning.py --model llama-maverick --query "Build a CI/CD pipeline"
```

---

## Demo Scenarios

### Scenario 1: Microservices Deployment

```bash
python demo-planning-cli.py --scenario microservices --execute
```

**Plan**: Deploy authentication, user, and API gateway services

**Highlights**:
- 11 total steps
- 6 execution phases
- 2 parallel phases (4 builds in parallel, 3 deploys in parallel)
- 1.8x speedup vs sequential
- Includes Planning Review steps for validation

**Output Example**:
```
Phase 1 (🚀 PARALLEL): Steps [1, 2, 3, 4]
   └─ Step 1: Build authentication service...
   └─ Step 2: Build user service...
   └─ Step 3: Build API gateway...
   └─ Step 4: Build database migrations...
Phase 2: Step 5 - Planning Review - Build Verification
Phase 3: Step 6 - Deploy database and run migrations
Phase 4 (🚀 PARALLEL): Steps [7, 8, 9]
   └─ Step 7: Deploy auth service...
   └─ Step 8: Deploy user service...
   └─ Step 9: Deploy API gateway...

📈 Summary:
   Total steps: 11
   Total phases: 6
   Parallel phases: 2
   Potential speedup: 1.8x vs sequential
```

### Scenario 2: Web Application Deployment

```bash
python demo-planning-cli.py --scenario webapp --execute
```

**Plan**: Build frontend, backend, and database; deploy to production

**Highlights**:
- 7 total steps
- 4 execution phases
- 2 parallel phases (3 builds, 2 deploys)
- Includes frontend build, backend build, database setup

### Scenario 3: Data Pipeline

```bash
python demo-planning-cli.py --scenario data-pipeline --execute
```

**Plan**: Extract, transform, and load data from multiple sources

**Highlights**:
- 9 total steps
- 5 execution phases
- 2 parallel phases (3 extractions, 3 transformations)
- Classic ETL pattern with parallel data processing

---

## Understanding the Output

### Plan Display

```
⚙️ Step 1: Build authentication service
    ⭐ INDEPENDENT - Can start immediately
   - Compile Go code
   - Run unit tests
   - Build Docker image

🔍 Step 5: Planning Review - Build Verification
    (depends on: 1, 2, 3, 4)
   - Review focus: Verify all builds succeeded
```

**Icons**:
- ⚙️ = Action step (executes a task)
- 🔍 = Planning Review step (validates and decides next steps)
- ⭐ = Independent step (can run immediately)

**Dependency Indicators**:
- `INDEPENDENT` = No dependencies, starts immediately
- `(depends on: 1, 2, 3)` = Must wait for steps 1, 2, 3 to complete

### Execution Analysis

```
📊 EXECUTION ANALYSIS:
----------------------------------------------------------------------
✅ Dependency graph is valid

Phase 1 (🚀 PARALLEL): Steps [1, 2, 3, 4]
Phase 2: Step 5 - Planning Review
Phase 3: Step 6 - Database deploy

📈 Summary:
   Total steps: 11
   Total phases: 6
   Parallel phases: 2
   Potential speedup: 1.8x vs sequential
```

**Metrics**:
- **Total steps**: Number of steps in the plan
- **Total phases**: Number of execution phases (some parallel, some sequential)
- **Parallel phases**: Phases where multiple steps run concurrently
- **Potential speedup**: How much faster vs running all steps sequentially

### Execution Simulation

```
🚀 Started Step 1: Build authentication service
🚀 Started Step 2: Build user service [🔀 Running in parallel with: 1]
🚀 Started Step 3: Build API gateway [🔀 Running in parallel with: 1, 2]
🚀 Started Step 4: Build database migrations [🔀 Running in parallel with: 1, 2, 3]
✅ Completed Step 1: Build authentication service (0.56s)
✅ Completed Step 2: Build user service (0.56s)
✅ Completed Step 3: Build API gateway (0.56s)
✅ Completed Step 4: Build database migrations (0.56s)
```

**Indicators**:
- 🚀 = Step started
- ✅ = Step completed
- 🔀 = Running in parallel with other steps
- (time) = Duration of step execution

### Performance Summary

```
📈 EXECUTION SUMMARY
======================================================================
⏱️  Total execution time: 5.96s
⏱️  Sequential time would be: 10.19s
🚀 Actual speedup: 1.71x
⚡ Time saved: 4.23s (71% faster)
```

**Metrics**:
- **Total execution time**: Actual time with parallel execution
- **Sequential time**: Time if all steps ran one-by-one
- **Actual speedup**: Performance multiplier (higher is better)
- **Time saved**: Absolute and percentage time reduction

---

## Dependency Patterns Demonstrated

### Pattern 1: Fan-Out (Parallel Start)

**Example**: Steps 1, 2, 3, 4 in microservices scenario

```
All have Dependencies: None
→ All start simultaneously in Phase 1
```

**Use Cases**:
- Multiple independent builds
- Parallel data extraction from different sources
- Concurrent service initialization

### Pattern 2: Fan-In (Parallel Converge)

**Example**: Step 5 depends on [1, 2, 3, 4]

```
Step 5 waits for ALL of [1, 2, 3, 4] to complete
→ Acts as a synchronization barrier
```

**Use Cases**:
- Review steps after parallel builds
- Aggregation after parallel processing
- Validation before proceeding

### Pattern 3: Diamond Pattern

**Example**: Microservices scenario

```
         Step 1-4 (parallel)
              ↓
          Step 5 (review)
              ↓
          Step 6 (setup)
              ↓
        Step 7-9 (parallel)
              ↓
         Step 10 (tests)
```

**Use Cases**:
- Multi-phase deployments
- Build → Review → Deploy workflows
- Data processing pipelines

---

## Creating Your Own Plans

To create custom plans (when using the real CLI with API key):

### Example Query

```bash
python lc-agent-planning.py --query "Create a plan to deploy a React frontend, Node.js backend, and MongoDB database with testing"
```

### Tips for Good Queries

**DO**:
- ✅ Mention multiple components that could run in parallel
- ✅ Specify testing/validation steps
- ✅ Include clear phases (build, test, deploy)
- ✅ Mention dependencies explicitly

**Example**: "Deploy a web app with React frontend build, Python backend build, and PostgreSQL database. Include testing and deployment."

**DON'T**:
- ❌ Make vague requests
- ❌ Request only single-step tasks
- ❌ Omit important phases like testing

**Example**: "Do something with a web app" (too vague)

### Good Query Templates

**Microservices**:
```
Deploy a microservices application with:
- Auth service (Go)
- User service (Python)
- API gateway (Node.js)
- PostgreSQL database
Include builds, tests, and deployment
```

**CI/CD Pipeline**:
```
Create a CI/CD pipeline for a Python application:
- Lint code
- Run unit tests
- Run integration tests
- Build Docker image
- Deploy to staging
- Deploy to production
```

**Data Pipeline**:
```
Build a data processing pipeline:
- Extract data from MySQL, API, and S3
- Validate all extracted data
- Transform customer and order data
- Load into data warehouse
- Verify data quality
```

---

## Command Reference

### Demo CLI

```bash
# View all scenarios
python demo-planning-cli.py --help

# Run specific scenario
python demo-planning-cli.py --scenario [webapp|microservices|data-pipeline]

# Execute simulation
python demo-planning-cli.py --scenario microservices --execute

# Skip execution prompt (auto-execute)
echo "yes" | python demo-planning-cli.py --scenario microservices --execute
```

### Real CLI (Requires API Key)

```bash
# Set API key
export NVIDIA_API_KEY="nvapi-your-key-here"

# Interactive mode
python lc-agent-planning.py

# Single query
python lc-agent-planning.py --query "your planning request"

# Use specific model
python lc-agent-planning.py --model gpt-120b --query "your request"
python lc-agent-planning.py --model llama-maverick --query "your request"

# Verbose mode (debug)
python lc-agent-planning.py --verbose --query "your request"
```

---

## Troubleshooting

### Issue: "Cannot import DependencyGraph"

**Solution**: Install the planning module
```bash
cd source/modules/agents/planning
pip install -e .
```

### Issue: "NVIDIA_API_KEY not set" (Real CLI)

**Solution**: Set your API key
```bash
# Get key from https://build.nvidia.com
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### Issue: "Model not found"

**Solution**: Use a registered model
```bash
# Available models:
# - gpt-120b (default)
# - llama-maverick
# - openai/gpt-oss-120b

python lc-agent-planning.py --model gpt-120b --query "your request"
```

### Issue: Network errors (Real CLI)

**Solution**: Check internet connection
```bash
# Test connection to NVIDIA API
curl -I https://integrate.api.nvidia.com

# If offline, use demo CLI instead
python demo-planning-cli.py --scenario microservices
```

---

## Performance Expectations

### Speedup by Scenario

| Scenario | Steps | Phases | Parallel Phases | Speedup |
|----------|-------|--------|-----------------|---------|
| Microservices | 11 | 6 | 2 | 1.8x |
| Web App | 7 | 4 | 2 | 1.8x |
| Data Pipeline | 9 | 5 | 2 | 1.8x |

### Real-World Estimates

**Example**: Microservices deployment

**Without parallelism** (sequential):
```
Build 1 (20s) → Build 2 (20s) → Build 3 (20s) → Build 4 (20s) →
Review (5s) → DB Deploy (30s) →
Deploy 1 (15s) → Deploy 2 (15s) → Deploy 3 (15s) →
Tests (30s) → Review (5s)

Total: ~195 seconds
```

**With parallelism**:
```
Phase 1: [Build 1, Build 2, Build 3, Build 4] in parallel → 20s
Phase 2: Review → 5s
Phase 3: DB Deploy → 30s
Phase 4: [Deploy 1, Deploy 2, Deploy 3] in parallel → 15s
Phase 5: Tests → 30s
Phase 6: Review → 5s

Total: ~105 seconds
Speedup: 1.86x (46% faster)
```

---

## Next Steps

### For Testing

1. **Run the demo**: `python demo-planning-cli.py --scenario microservices --execute`
2. **Try different scenarios**: webapp, data-pipeline
3. **Observe parallel execution**: Watch steps run concurrently

### For Real Usage

1. **Get NVIDIA API key**: https://build.nvidia.com
2. **Set environment variable**: `export NVIDIA_API_KEY="your-key"`
3. **Run real CLI**: `python lc-agent-planning.py --query "your planning request"`

### For Development

1. **Review implementation**: Check `planning_modifier.py` for parallel execution logic
2. **Run tests**: `python test_dependency_graph_comprehensive.py`
3. **Read design docs**: `PARALLEL_PLANNING_DEPENDENCY_GRAPH_DESIGN.md`

---

## Support

**Documentation**:
- Design: `PARALLEL_PLANNING_DEPENDENCY_GRAPH_DESIGN.md`
- Tests: `PARALLEL_PLANNING_TEST_REPORT.md`
- Code: `source/modules/agents/planning/`

**Examples**:
- Demo CLI: `demo-planning-cli.py`
- Real CLI: `lc-agent-planning.py`
- Tests: `test_*.py`

**Questions**:
- Check the design documents for detailed explanations
- Review test files for usage examples
- Run demo CLI to see the system in action

---

**Ready to see parallel planning in action?**

```bash
python demo-planning-cli.py --scenario microservices --execute
```

Enjoy the power of parallel task execution! 🚀
