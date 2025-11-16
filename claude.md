# Claude Code Development Guide for Kit USD Agents

This document provides essential information for AI assistants (like Claude) working on this repository.

## Repository Overview

**Kit USD Agents** is a repository containing Chat USD and its supporting framework for AI-assisted Universal Scene Description (USD) development in NVIDIA Omniverse Kit.

### Key Components

1. **LC Agent** (`source/modules/lc_agent/`) - Core AI agent framework built on LangChain (generic, model-agnostic)
2. **LC Agent CLI** (`source/modules/lc_agent_cli/`) - Command-line interface with NVIDIA model support
3. **USD Agents** (`source/modules/agents/usd/`) - USD-specific agent implementations
4. **RAG Components** (`source/modules/rags/`) - Retrieval-augmented generation modules
5. **AIQ Integration** (`source/modules/aiq/`) - NVIDIA NeMo Agent Toolkit integration
6. **Extensions** (`source/extensions/`) - Omniverse Kit extensions

## LC Agent Development Workflow

### Quick Start for Development

The LC Agent module is the core component you'll likely iterate on most frequently. The `lc_agent_cli` module provides a convenient CLI for testing and evaluating the core `lc_agent` framework. Here's how to set it up for local development:

#### 1. Set up NVIDIA API Key

The LC Agent CLI uses NVIDIA Cloud Functions (NVCF) models by default. You'll need an NVIDIA API key:

```bash
# Linux/Mac
export NVIDIA_API_KEY=nvapi-your-key-here

# Windows PowerShell
$env:NVIDIA_API_KEY="nvapi-your-key-here"

# Windows CMD
set NVIDIA_API_KEY=nvapi-your-key-here
```

To make this permanent, add it to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or Windows environment variables.

**Get your API key**: Visit [NVIDIA API Catalog](https://build.nvidia.com) to get a free API key.

#### 2. Create a Python Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install LC Agent CLI in Editable Mode

```bash
# Linux/Mac
./dev-install.sh

# Windows
dev-install.bat
```

This installs the `lc_agent_cli` module in editable mode, which also installs `lc_agent` as a dependency in editable mode. Changes to either module's source code are immediately reflected without reinstallation.

**Locations**:
- CLI module: `source/modules/lc_agent_cli/`
- Core module: `source/modules/lc_agent/`

**What gets installed**:
- `lc_agent_cli` - CLI with NVIDIA model registration
- `lc_agent` - Core framework (installed automatically as dependency)
- `lc-agent` command-line tool

**Dependencies**:
- Core (`lc_agent`): langchain-core, langchain, langchainhub, aioredis, libcst, tiktoken, toml, aiohttp, requests
- CLI (`lc_agent_cli`): langchain-nvidia-ai-endpoints (for NVIDIA models)

#### 4. Run the CLI

You can run the CLI in three ways:

**Option 1: Using the installed command**
```bash
lc-agent                              # Interactive mode
lc-agent --query "Explain USD prims"  # Single query
lc-agent --help                       # Show all options
```

**Option 2: Using wrapper scripts**
```bash
# Linux/Mac
./run-lc-agent.sh
./run-lc-agent.sh --query "Explain USD prims"

# Windows
run-lc-agent.bat
run-lc-agent.bat --query "Explain USD prims"
```

**Option 3: Using Python module**
```bash
python -m lc_agent_cli --query "Explain USD prims"
```

**CLI Options:**
```bash
# Interactive mode (uses gpt-120b by default)
lc-agent

# Single query
lc-agent --query "Explain USD prims"

# Use USD assistant mode
lc-agent --assistant usd

# Use a different NVIDIA model
lc-agent --model llama-maverick

# Verbose mode for debugging
lc-agent --verbose --query "Hello"
```

**Available Models** (via NVCF):
- `gpt-120b` - Default, GPT-4 class model (openai/gpt-oss-120b on NVIDIA Build)
- `openai/gpt-oss-120b` - Full name alias for gpt-120b
- `llama-maverick` - Fast Llama 4 model (meta/llama-4-maverick-17b-128e-instruct)

**Note**: The CLI defaults to `gpt-120b` which runs on NVIDIA's infrastructure for free with an API key.

### Module Structure

**Core LC Agent** (generic, model-agnostic):
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
│   ├── chat_models/        # Chat model base classes
│   ├── code_atlas/         # Code analysis and tools
│   └── utils/              # Utilities (profiling, etc.)
├── tests/                  # Unit tests
├── doc/                    # Documentation
├── requirements.txt        # Python dependencies (no NVIDIA-specific packages)
├── setup.py               # Package configuration
└── README.md              # Module documentation
```

**LC Agent CLI** (NVIDIA model support):
```
source/modules/lc_agent_cli/
├── src/lc_agent_cli/
│   ├── __init__.py         # Exports register_all()
│   ├── cli.py              # CLI implementation
│   ├── register_models.py  # NVIDIA model registration
│   └── __main__.py         # Entry point for python -m
├── requirements.txt        # Includes -e ../lc_agent and NVIDIA packages
├── setup.py               # Defines lc-agent console script
└── README.md              # CLI documentation
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

**When iterating on the core LC Agent framework:**

1. **Edit source files** in `source/modules/lc_agent/src/lc_agent/`
2. **Changes are immediately active** (editable install)
3. **Test your changes** with the CLI: `lc-agent --query "test"`
4. **Run unit tests**: `cd source/modules/lc_agent && pytest tests/`
5. **Document** significant changes in the module's CHANGELOG.md

**When adding NVIDIA model support or modifying the CLI:**

1. **Edit CLI files** in `source/modules/lc_agent_cli/src/lc_agent_cli/`
2. **Changes are immediately active** (editable install)
3. **Test**: `lc-agent --verbose --query "test"`

**Important**: Keep `lc_agent` generic and model-agnostic. All NVIDIA-specific code goes in `lc_agent_cli`.

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

- **NVIDIA_API_KEY**: NVIDIA API key for NVCF models (required for CLI, get from https://build.nvidia.com)
- **PYTHON**: Override Python executable (default: `python3` on Linux/Mac, `python` on Windows)
- **LC_AGENT_MODEL**: Default chat model for CLI (default: `gpt-120b`)

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
2. **Module separation**: Keep `lc_agent` generic; put NVIDIA/model-specific code in `lc_agent_cli`
3. **Use editable install**: The dev-install scripts install both modules in editable mode
4. **Test frequently**: Use the CLI (`lc-agent` or `run-lc-agent.sh`) to quickly test changes
5. **Check examples**: Look at tests in `tests/` for usage patterns
6. **Read the philosophy**: Understanding the dynamic graph approach is key (see module README)
7. **Virtual environment**: Always use a virtual environment to avoid conflicts

## File Locations Summary

| Purpose | File | Description |
|---------|------|-------------|
| Dev Install | `dev-install.sh` / `dev-install.bat` | Install lc_agent_cli (and lc_agent) in editable mode |
| CLI Runner | `run-lc-agent.sh` / `run-lc-agent.bat` | Run the LC Agent CLI (wrapper scripts) |
| CLI Module | `source/modules/lc_agent_cli/` | CLI with NVIDIA model support |
| LC Agent Source | `source/modules/lc_agent/src/lc_agent/` | Core framework (generic) |
| LC Agent Tests | `source/modules/lc_agent/tests/` | Core framework unit tests |
| LC Agent Docs | `source/modules/lc_agent/doc/` | Module documentation |

## Version Information

- **LC Agent**: v0.2.9
- **Python**: >= 3.10 required
- **LangChain**: Core dependency (see requirements.txt for specific versions)

---

*This document is intended to help Claude Code and other AI assistants quickly understand and work with the Kit USD Agents repository. For human developers, also refer to the official documentation in each module's README.*
