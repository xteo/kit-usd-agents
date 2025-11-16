# Claude Code Development Guide for Kit USD Agents

This document provides essential information for AI assistants (like Claude) working on this repository.

## Repository Overview

**Kit USD Agents** is a repository containing Chat USD and its supporting framework for AI-assisted Universal Scene Description (USD) development in NVIDIA Omniverse Kit.

### Key Components

1. **LC Agent** (`source/modules/lc_agent/`) - Core AI agent framework built on LangChain
2. **USD Agents** (`source/modules/agents/usd/`) - USD-specific agent implementations
3. **RAG Components** (`source/modules/rags/`) - Retrieval-augmented generation modules
4. **AIQ Integration** (`source/modules/aiq/`) - NVIDIA NeMo Agent Toolkit integration
5. **Extensions** (`source/extensions/`) - Omniverse Kit extensions

## LC Agent Development Workflow

### Quick Start for Development

The LC Agent module is the core component you'll likely iterate on most frequently. Here's how to set it up for local development:

#### 1. Create a Python Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 2. Install LC Agent in Editable Mode

```bash
# Linux/Mac
./dev-install.sh

# Windows
dev-install.bat
```

This installs the `lc_agent` module in editable mode, meaning changes to the source code are immediately reflected without reinstallation.

**Location**: `source/modules/lc_agent/`

**What gets installed**:
- Main package: `lc_agent` (from `source/modules/lc_agent/src/lc_agent/`)
- Dependencies from `requirements.txt`:
  - langchain-core
  - langchain
  - langchainhub
  - aioredis
  - libcst
  - tiktoken
  - toml

#### 3. Run the CLI

```bash
# Interactive mode
./run-lc-agent.sh

# Single query
./run-lc-agent.sh --query "Explain USD prims"

# Use USD assistant mode
./run-lc-agent.sh --assistant usd

# With custom model
./run-lc-agent.sh --model gpt-4

# Help
./run-lc-agent.sh --help
```

**Windows:**
```batch
run-lc-agent.bat --query "Explain USD prims"
```

### Module Structure

```
source/modules/lc_agent/
├── src/lc_agent/           # Main source code
│   ├── __init__.py         # Public API exports
│   ├── runnable_node.py    # Core node implementation
│   ├── runnable_network.py # Network/graph management
│   ├── network_modifier.py # Middleware for behavior modification
│   ├── node_factory.py     # Node type registry
│   ├── multi_agent_network_node.py  # Multi-agent coordination
│   ├── usd_assistant.py    # USD-specific assistant
│   ├── chat_models/        # Chat model integrations
│   ├── code_atlas/         # Code analysis and tools
│   └── utils/              # Utilities (profiling, etc.)
├── tests/                  # Unit tests
├── doc/                    # Documentation
├── requirements.txt        # Python dependencies
├── setup.py               # Package configuration
└── README.md              # Module documentation
```

### Key Concepts

From the LC Agent documentation:

1. **Dynamic Graph Construction**: Networks grow dynamically during execution rather than being pre-defined
2. **Immutable Nodes**: Each node is a snapshot of a result (LLM response, tool output, or user query)
3. **State as Structure**: The graph itself represents the state, not a separate state object
4. **Reactive Flow**: NetworkModifiers inspect results and decide what nodes to create next

### Core Classes

- **RunnableNode**: Fundamental building block for processing units
- **RunnableNetwork**: Container and manager for nodes
- **NetworkNode**: Hybrid component functioning as both node and network
- **MultiAgentNetworkNode**: Router for multi-agent systems
- **NetworkModifier**: Middleware for safe behavior modification
- **NodeFactory**: Centralized registry for node types

### Running Tests

```bash
cd source/modules/lc_agent
python -m pytest tests/
```

### Making Changes

When iterating on the LC Agent module:

1. **Edit source files** in `source/modules/lc_agent/src/lc_agent/`
2. **Changes are immediately active** (editable install)
3. **Test your changes** with the CLI or unit tests
4. **Document** significant changes in the module's CHANGELOG.md

### Related Modules

If you're working on specific features, you may need to install related modules:

```bash
# USD agent
cd source/modules/agents/usd && pip install -e .

# RAG components
cd source/modules/rags/retrievers && pip install -e .
cd source/modules/rags/rag_nodes && pip install -e .
cd source/modules/rags/rag_modifiers && pip install -e .

# AIQ integration
cd source/modules/aiq/lc_agent_aiq && pip install -e .
```

## Environment Variables

- **PYTHON**: Override Python executable (default: `python3` on Linux/Mac, `python` on Windows)
- **LC_AGENT_MODEL**: Default chat model for CLI (default: `gpt-4`)

## Common Tasks

### Adding a New Node Type

1. Create a new class inheriting from `RunnableNode`
2. Implement required methods (`invoke`, `ainvoke`, `astream`)
3. Register with NodeFactory: `get_node_factory().register(YourNode)`
4. Use in networks: `with RunnableNetwork(default_node="YourNode") as network: ...`

### Creating a Custom Assistant

See `source/modules/lc_agent/src/lc_agent/usd_assistant.py` for example:

```python
from lc_agent import RunnableNode, RunnableSystemAppend

class MyAssistant(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.append(
            RunnableSystemAppend(system_message="Your system prompt")
        )
```

### Working with Multi-Agent Systems

See documentation in `source/modules/lc_agent/doc/multi_agent/`

### Profiling Performance

LC Agent includes built-in profiling:

```python
from lc_agent import enable_profiling, create_profiling_html

enable_profiling()
# ... run your network ...
html = create_profiling_html()
# Save or serve the HTML for interactive visualization
```

## Build System (Full Kit Extensions)

The repository uses NVIDIA's Kit build system for building Omniverse extensions:

```bash
# Linux/Mac
./build.sh -r

# Windows
build.bat -r
```

This builds all extensions and creates the full Chat USD application in `_build/`.

## Documentation References

- **LC Agent Core**: `source/modules/lc_agent/README.md`
- **LC Agent Docs**: `source/modules/lc_agent/doc/`
- **Chat USD**: `source/extensions/omni.ai.chat_usd.bundle/docs/`
- **Repository**: `README.md`

## Tips for AI Assistants

1. **Focus on LC Agent**: Most development iteration happens in `source/modules/lc_agent/`
2. **Use editable install**: Always install with `-e` flag for rapid iteration
3. **Test frequently**: Use the CLI (`run-lc-agent.sh`) to quickly test changes
4. **Check examples**: Look at tests in `tests/` for usage patterns
5. **Read the philosophy**: Understanding the dynamic graph approach is key (see module README)
6. **Virtual environment**: Always use a virtual environment to avoid conflicts

## File Locations Summary

| Purpose | File | Description |
|---------|------|-------------|
| Dev Install | `dev-install.sh` / `dev-install.bat` | Install lc_agent in editable mode |
| CLI Runner | `run-lc-agent.sh` / `run-lc-agent.bat` | Run the LC Agent CLI |
| CLI Script | `lc_agent_cli.py` | Python CLI implementation |
| LC Agent Source | `source/modules/lc_agent/src/lc_agent/` | Main source code directory |
| Tests | `source/modules/lc_agent/tests/` | Unit tests |
| Docs | `source/modules/lc_agent/doc/` | Module documentation |

## Version Information

- **LC Agent**: v0.2.9
- **Python**: >= 3.10 required
- **LangChain**: Core dependency (see requirements.txt for specific versions)

---

*This document is intended to help Claude Code and other AI assistants quickly understand and work with the Kit USD Agents repository. For human developers, also refer to the official documentation in each module's README.*
