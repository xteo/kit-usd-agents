# Kit USD Agents

This repository contains Chat USD and its supporting framework for AI-assisted Universal Scene Description (USD) development in NVIDIA Omniverse Kit.

## What is Chat USD?

Chat USD is a specialized AI assistant that enables natural language interaction with USD scenes. Built on top of LangChain, Chat USD provides a multi-agent system for USD development workflows.

### Core Capabilities
- **USD Code Generation & Execution**: Generate and execute USD code from natural language descriptions
- **Asset Search**: Search for USD assets using natural language queries
- **Scene Information**: Analyze and retrieve information about USD scenes
- **Interactive Development**: Real-time scene modification through conversation
- **Extensibility**: Add custom agents like navigation, UI generation, and more

## Repository Structure

### Extensions
- `omni.ai.chat_usd.bundle` - The main Chat USD extension bundle
- `omni.ai.langchain.agent.usd_code` - USD code generation and execution agent
- `omni.ai.langchain.agent.navigation` - Example custom agent for scene navigation
- `omni.ai.langchain.widget.core` - UI components for AI-powered interfaces
- `omni.ai.langchain.core` - Bridge between LangChain and Omniverse
- `omni.ai.langchain.aiq` - NVIDIA NeMo Agent Toolkit platform integration
- `omni.ai.aiq.agent.chat_usd` - Chat USD integration with NVIDIA NeMo Agent Toolkit

### Modules
- `lc_agent` - Core LC Agent built on LangChain
- `agents/usd` - USD-specific agent implementations
- `data_generation/usdcode` - USD meta-functions for optimized operations
- `rags` - Retrieval-augmented generation components
- `aiq` - NVIDIA NeMo Agent Toolkit integration utilities

## Getting Started

### Full Kit Extension Build

Build and run the complete Chat USD application with Omniverse Kit:

1. Build: `./build.sh -r` (Linux/Mac) or `build.bat -r` (Windows)
2. Run: `_build/windows-x86_64/release/omni.app.chat_usd.bat` (Windows) or similar path for Linux

### LC Agent Development (Python Module)

For rapid iteration on the LC Agent framework without building Kit extensions:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install LC Agent in editable mode**:
   ```bash
   ./dev-install.sh  # Linux/Mac
   # or
   dev-install.bat  # Windows
   ```

3. **Run the CLI**:
   ```bash
   ./run-lc-agent.sh --help  # Linux/Mac
   # or
   run-lc-agent.bat --help  # Windows
   ```

**Examples**:
```bash
# Interactive mode
./run-lc-agent.sh

# Single query
./run-lc-agent.sh --query "Explain USD prims"

# USD assistant mode
./run-lc-agent.sh --assistant usd
```

This approach allows you to:
- Quickly test changes to the LC Agent core framework
- Develop and debug agent logic without Kit dependencies
- Run unit tests and experiments in a lightweight environment

## Documentation

- **Development Guide**: See `claude.md` for detailed development workflow and AI assistant guidance
- **LC Agent**: `source/modules/lc_agent/README.md` - Core agent framework documentation
- **Chat USD**: `source/extensions/omni.ai.chat_usd.bundle/docs/` - Full extension documentation
