# LC Agent CLI

Command-line interface for the LC Agent framework with NVIDIA model support.

## Installation

```bash
# From the lc_agent_cli directory
pip install -e .

# This will automatically install lc_agent in editable mode
```

## Usage

```bash
# As a command (after installation)
lc-agent --query "Hello, how are you?"

# Or as a Python module
python -m lc_agent_cli --query "Hello"

# Interactive mode
lc-agent

# With specific model
lc-agent --model llama-maverick --query "Explain USD"

# Verbose debugging
lc-agent --verbose --query "Test"
```

## Environment Variables

- `NVIDIA_API_KEY` - Your NVIDIA API key (get one from https://build.nvidia.com)
- `LC_AGENT_MODEL` - Default model to use (default: gpt-120b)

## Available Models

- `gpt-120b` - GPT-4 class model (openai/gpt-oss-120b)
- `openai/gpt-oss-120b` - Full name for gpt-120b
- `llama-maverick` - Llama 4 Maverick model

## Development

This CLI depends on `lc_agent` which is installed in editable mode. Any changes
you make to the `lc_agent` module will be immediately reflected in the CLI.

Perfect for iterating on LC Agent features!
