# DeepAgent to LC Agent Integration Analysis

## Executive Summary

After comprehensive analysis of both codebases, I've identified the key DeepAgent features that can be lifted into LC Agent, particularly the **file system tools** and **state management** patterns. This document provides architectural analysis, integration strategies, and implementation recommendations.

## Table of Contents

1. [Architectural Comparison](#architectural-comparison)
2. [DeepAgent File System Architecture](#deepagent-file-system-architecture)
3. [Integration Strategy](#integration-strategy)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Code Examples](#code-examples)

---

## Architectural Comparison

### LC Agent Architecture

**Core Design:**
- Built on LangChain (NOT LangGraph)
- Dynamic graph construction via RunnableNode and RunnableNetwork
- Graph-as-state: The network structure IS the state
- NetworkModifiers for safe mutations
- Pydantic-based serialization

**Current State Management:**
```python
# State stored in RunnableNetwork.nodes
# Each node caches its outputs
node.outputs = result  # Immutable once invoked
node.metadata = {...}  # Execution metrics

# Persistence via NetworkList
JsonNetworkList.save(network)  # Entire network serialized
RedisNetworkList.save(network)  # Async Redis storage
```

**Current Limitations:**
- ❌ No file system tools for file read/write during execution
- ❌ No built-in artifact management
- ❌ No automatic state checkpointing during execution
- ❌ No tool result eviction to disk
- ❌ Must manually save entire network

### DeepAgent Architecture

**Core Design:**
- Built on LangGraph (state machine approach)
- Predefined graph with explicit state passing
- Middleware-based extensibility
- LangGraph's built-in checkpointing

**State Management:**
```python
class FilesystemState(AgentState):
    files: Annotated[dict[str, FileData], _file_data_reducer]
    messages: Annotated[list[BaseMessage], add_messages]
```

**Key Capabilities:**
- ✅ File system tools (ls, read_file, write_file, edit_file, glob, grep, execute)
- ✅ Automatic tool result eviction (saves large results to filesystem)
- ✅ Multiple backend support (StateBackend, StoreBackend, FilesystemBackend, CompositeBackend)
- ✅ State updates via Command pattern
- ✅ Artifact management with /large_tool_results/

### Architectural Compatibility Analysis

| Feature | LC Agent | DeepAgent | Lift Strategy |
|---------|----------|-----------|---------------|
| **State Model** | Graph structure | Dict-based state | ✅ Add `files` field to RunnableNetwork |
| **Tool Integration** | BaseTool compatible | BaseTool compatible | ✅ Direct port |
| **Reducers** | N/A (immutable nodes) | Annotated reducers | ✅ Implement custom reducer in NetworkModifier |
| **Middleware** | NetworkModifier | AgentMiddleware | ✅ 1:1 mapping |
| **Serialization** | Pydantic | Pydantic | ✅ Compatible |
| **Command Pattern** | Not used | Core pattern | ✅ Extend node outputs to support updates |

**Key Insight:** LC Agent's NetworkModifier pattern is functionally equivalent to DeepAgent's AgentMiddleware. This makes integration straightforward.

---

## DeepAgent File System Architecture

### 1. Backend Protocol System

DeepAgent uses a **protocol-based backend abstraction** that separates storage logic from tool implementation:

```python
# Core protocol
class BackendProtocol(Protocol):
    def ls_info(self, path: str) -> list[FileInfo]: ...
    def read(self, file_path: str, offset: int, limit: int) -> str: ...
    def write(self, file_path: str, content: str) -> WriteResult: ...
    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool) -> EditResult: ...
    def grep_raw(self, pattern: str, path: str | None, glob: str | None) -> list[GrepMatch] | str: ...
    def glob_info(self, pattern: str, path: str) -> list[FileInfo]: ...
```

**Four Backend Implementations:**

#### 1. **StateBackend** (Ephemeral In-Memory)
```python
# Files stored in LangGraph state
runtime.state["files"] = {
    "/file.txt": {
        "content": ["line1", "line2"],
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-01T00:00:00Z"
    }
}

# Returns files_update for state merging
WriteResult(path="/file.txt", files_update={"/file.txt": FileData(...)}, error=None)
```

**Use Case:** Temporary files during agent execution, lost after session ends.

#### 2. **StoreBackend** (Persistent Cross-Session)
```python
# Uses LangChain Store for persistence
store.put(
    namespace=("assistant_id", "filesystem"),
    key="/memories/notes.txt",
    value={"content": [...], "created_at": "...", "modified_at": "..."}
)

# External storage - no state updates needed
WriteResult(path="/memories/notes.txt", files_update=None, error=None)
```

**Use Case:** Long-term memory, persists across sessions and threads.

#### 3. **FilesystemBackend** (Direct Disk Access)
```python
# Direct filesystem operations
backend = FilesystemBackend(root_dir="/workspace", virtual_mode=True)
backend.write("/config.json", content)
# → Writes to /workspace/config.json

# Security: virtual_mode sandboxes paths to root_dir
WriteResult(path="/config.json", files_update=None, error=None)
```

**Use Case:** Real file I/O, integration with existing codebases.

#### 4. **CompositeBackend** (Hybrid Routing)
```python
# Route different paths to different backends
composite = CompositeBackend(
    default=FilesystemBackend(root_dir="/workspace"),
    routes={
        "/memories/": StoreBackend(runtime),  # Persistent
        "/temp/": StateBackend(runtime)       # Ephemeral
    }
)

# /memories/notes.txt → StoreBackend
# /temp/cache.json → StateBackend
# /workspace/file.py → FilesystemBackend
```

**Use Case:** Complex agents with mixed storage requirements.

### 2. File System Tools

DeepAgent provides **7 core tools** via FilesystemMiddleware:

#### Tool 1: `ls` - List Directory
```python
@tool
def ls(path: str, runtime: ToolRuntime) -> str:
    """List files and directories at the given absolute path."""
    backend = _get_backend(backend_config, runtime)
    results = backend.ls_info(path)
    return "\n".join(r["path"] for r in results)
```

#### Tool 2: `read_file` - Read with Pagination
```python
@tool
def read_file(file_path: str, offset: int = 0, limit: int = 500, runtime: ToolRuntime) -> str:
    """Read file contents with line numbers. Supports pagination."""
    backend = _get_backend(backend_config, runtime)
    return backend.read(file_path, offset, limit)
```

**Output Format:**
```
     1  import os
     2  import sys
     3
     4  def main():
     5      print("Hello")
```

#### Tool 3: `write_file` - Create New File
```python
@tool
def write_file(file_path: str, content: str, runtime: ToolRuntime) -> Command | str:
    """Create a new file. Fails if file exists."""
    backend = _get_backend(backend_config, runtime)
    res: WriteResult = backend.write(file_path, content)

    if res.error:
        return res.error

    if res.files_update is not None:
        # State-backed: return Command to update state
        return Command(
            update={
                "files": res.files_update,
                "messages": [ToolMessage(f"Created {res.path}", ...)]
            }
        )
    else:
        # External storage: just return success message
        return f"Successfully wrote to {res.path}"
```

#### Tool 4: `edit_file` - String Replacement
```python
@tool
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    runtime: ToolRuntime
) -> Command | str:
    """Replace old_string with new_string in file."""
    backend = _get_backend(backend_config, runtime)
    res: EditResult = backend.edit(file_path, old_string, new_string, replace_all)

    if res.error:
        return res.error

    return Command(update={...}) if res.files_update else f"Edited {res.path}"
```

**Validation:**
- Fails if `old_string` not found
- Fails if multiple occurrences found and `replace_all=False`
- Returns occurrence count

#### Tool 5: `glob` - Pattern Matching
```python
@tool
def glob(pattern: str, path: str = "/", runtime: ToolRuntime) -> str:
    """Find files matching glob pattern (e.g., **/*.py)."""
    backend = _get_backend(backend_config, runtime)
    results = backend.glob_info(pattern, path)
    return "\n".join(r["path"] for r in results)
```

#### Tool 6: `grep` - Content Search
```python
@tool
def grep(
    pattern: str,
    path: str = "/",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    runtime: ToolRuntime
) -> str:
    """Search file contents using regex."""
    backend = _get_backend(backend_config, runtime)
    results = backend.grep_raw(pattern, path, glob)

    if output_mode == "files_with_matches":
        return "\n".join(unique(r["path"] for r in results))
    elif output_mode == "content":
        return "\n".join(f"{r['path']}:{r['line']}:{r['text']}" for r in results)
    elif output_mode == "count":
        counts = Counter(r["path"] for r in results)
        return "\n".join(f"{path}: {count}" for path, count in counts.items())
```

#### Tool 7: `execute` - Shell Commands (Conditional)
```python
@tool
def execute(command: str, runtime: ToolRuntime) -> str:
    """Execute shell command in sandbox. Only available if backend supports it."""
    backend = _get_backend(backend_config, runtime)
    if not isinstance(backend, SandboxBackendProtocol):
        return "Execute tool not available - backend doesn't support execution"

    res = backend.execute(command)
    return f"Exit code: {res.exit_code}\n{res.output}"
```

### 3. State Management Pattern

DeepAgent uses **Annotated reducers** for automatic state merging:

```python
def _file_data_reducer(
    left: dict[str, FileData] | None,
    right: dict[str, FileData | None]
) -> dict[str, FileData]:
    """
    Merge file dictionaries.
    - None in right deletes files
    - New entries added
    - Existing entries replaced
    """
    result = left.copy() if left else {}
    for key, value in (right or {}).items():
        if value is None:
            result.pop(key, None)  # Delete
        else:
            result[key] = value    # Add/Update
    return result

class FilesystemState(AgentState):
    files: Annotated[dict[str, FileData], _file_data_reducer]
```

**Command Pattern for Updates:**
```python
# Tool returns Command instead of just string
return Command(
    update={
        "files": {"/new_file.txt": FileData(...)},  # Merged via reducer
        "messages": [ToolMessage("Created file", ...)]
    }
)
```

### 4. Tool Result Eviction

DeepAgent automatically saves large tool results to disk:

```python
def wrap_tool_call(self, tool_call, handler):
    result = handler(tool_call)  # Execute tool normally

    # Check result size
    if len(result.content) > 4 * self.tool_token_limit_before_evict:
        # Save to /large_tool_results/{tool_call_id}
        file_path = f"/large_tool_results/{sanitize(tool_call.id)}"
        backend.write(file_path, result.content)

        # Replace with summary
        summary = f"Tool result too large ({len(result.content)} chars). "
        summary += f"Saved to {file_path}. First 10 lines:\n"
        summary += "\n".join(result.content.split("\n")[:10])
        summary += f"\n... [Use read_file('{file_path}') to view full result]"

        return Command(
            update={
                "files": {file_path: FileData(...)},
                "messages": [ToolMessage(summary, ...)]
            }
        )

    return result
```

**Benefits:**
- Prevents context overflow from large tool outputs
- Automatic pagination via read_file
- Transparent to agent (just uses read_file to access)

---

## Integration Strategy

### Phase 1: Core Filesystem Support (CRITICAL PATH)

**Goal:** Enable LC Agent to read/write files during execution using DeepAgent's backend system.

#### 1.1 Add Filesystem State to RunnableNetwork

**Current LC Agent State:**
```python
class RunnableNetwork(RunnableSerializable):
    nodes: List[RunnableNode] = []
    metadata: Dict[str, Any] = {}
```

**Enhanced with Filesystem:**
```python
class RunnableNetwork(RunnableSerializable):
    nodes: List[RunnableNode] = []
    metadata: Dict[str, Any] = {}
    files: Dict[str, FileData] = Field(default_factory=dict)  # NEW
```

**FileData Definition (lift directly from DeepAgent):**
```python
class FileData(TypedDict):
    content: list[str]
    created_at: str
    modified_at: str
```

#### 1.2 Create FilesystemNetworkModifier

Map DeepAgent's FilesystemMiddleware to LC Agent's NetworkModifier:

```python
from lc_agent import NetworkModifier
from deepagents.backends import BackendProtocol, StateBackend

class FilesystemNetworkModifier(NetworkModifier):
    """
    Provides file system tools to LC Agent networks.
    Directly uses DeepAgent's backend implementations.
    """

    def __init__(
        self,
        backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol] | None = None,
        tool_token_limit_before_evict: int = 20000
    ):
        self.backend = backend
        self.tool_token_limit = tool_token_limit_before_evict
        self._tools = self._create_tools()

    def _create_tools(self) -> List[BaseTool]:
        """Create filesystem tools using DeepAgent's implementations."""
        # Import tools from deepagents.middleware.filesystem
        from deepagents.middleware.filesystem import (
            ls, read_file, write_file, edit_file, glob, grep
        )

        # Wrap to adapt DeepAgent's ToolRuntime to LC Agent's context
        return [
            self._wrap_tool(ls),
            self._wrap_tool(read_file),
            self._wrap_tool(write_file),
            self._wrap_tool(edit_file),
            self._wrap_tool(glob),
            self._wrap_tool(grep),
        ]

    def _wrap_tool(self, deepagent_tool: BaseTool) -> BaseTool:
        """Wrap DeepAgent tool to work with LC Agent."""
        # Shim: Adapt LC Agent's network context to DeepAgent's ToolRuntime
        # Details in implementation section
        pass

    def on_begin_invoke(self, network: RunnableNetwork):
        """Register tools when network starts."""
        # Add tools to chat model via bind_tools()
        pass

    def on_post_invoke(self, network: RunnableNetwork, node: RunnableNode):
        """Handle tool results, check for eviction."""
        if node.outputs and hasattr(node.outputs, 'tool_calls'):
            # Check for large results, evict if needed
            pass
```

#### 1.3 Implement Minimal ToolRuntime Shim

DeepAgent tools expect `ToolRuntime` with `state`, `store`, `config`:

```python
class LCAgentToolRuntime:
    """
    Shim to adapt LC Agent's context to DeepAgent's ToolRuntime interface.
    """

    def __init__(self, network: RunnableNetwork, node: RunnableNode):
        self.network = network
        self.node = node

    @property
    def state(self) -> dict:
        """Expose network.files as state["files"]."""
        return {"files": self.network.files}

    @property
    def store(self):
        """TODO: Integrate with LC Agent's persistence layer."""
        # Could connect to RedisNetworkList or custom store
        return None

    @property
    def config(self) -> dict:
        """Expose network metadata as config."""
        return self.network.metadata
```

#### 1.4 Lift StateBackend (Zero Dependencies)

DeepAgent's StateBackend is **completely standalone** - just uses Python dicts:

```python
# Copy from reference_repos/deepagents/libs/deepagents/deepagents/backends/state.py
# to source/modules/lc_agent/src/lc_agent/backends/state.py

# NO MODIFICATIONS NEEDED - it's pure Python!
```

### Phase 2: Tool Result Eviction (HIGH VALUE)

**Goal:** Prevent context overflow from large tool outputs.

#### 2.1 Add Eviction Logic to FilesystemNetworkModifier

```python
class FilesystemNetworkModifier(NetworkModifier):
    def on_post_invoke_async(self, network: RunnableNetwork, node: RunnableNode):
        """Intercept large tool results."""
        if not node.outputs or not hasattr(node.outputs, 'content'):
            return

        content = node.outputs.content
        if len(content) > 4 * self.tool_token_limit:
            # Evict to filesystem
            tool_call_id = node.metadata.get('tool_call_id')
            file_path = f"/large_tool_results/{sanitize(tool_call_id)}"

            # Use backend to save
            backend = self._get_backend(network)
            result = backend.write(file_path, content)

            # Update network files
            if result.files_update:
                network.files.update(result.files_update)

            # Replace node output with summary
            summary = self._create_summary(content, file_path)
            node.outputs.content = summary
```

### Phase 3: Persistent Storage (MEDIUM PRIORITY)

**Goal:** Enable cross-session file persistence.

#### 3.1 Lift StoreBackend

```python
# Copy from reference_repos/deepagents/libs/deepagents/deepagents/backends/store.py
# Requires: LangChain Store integration

# LC Agent already has RedisNetworkList - could extend to support StoreBackend
```

#### 3.2 Integrate with NetworkList

```python
class FilesystemNetworkList(NetworkList):
    """
    Extended NetworkList that persists both network structure AND files.
    """

    def __init__(self, backend: BackendProtocol):
        self.backend = backend

    def save(self, network: RunnableNetwork):
        # Save network structure (existing logic)
        super().save(network)

        # Save files via backend
        for file_path, file_data in network.files.items():
            self.backend.write(file_path, "\n".join(file_data["content"]))

    def load(self) -> List[RunnableNetwork]:
        networks = super().load()

        # Hydrate files from backend
        for network in networks:
            network.files = self._load_files_for_network(network)

        return networks
```

### Phase 4: CompositeBackend & Advanced Features (OPTIONAL)

#### 4.1 Lift CompositeBackend

```python
# Direct copy - zero modifications needed
# from reference_repos/deepagents/libs/deepagents/deepagents/backends/composite.py
```

#### 4.2 Add FilesystemBackend for Real I/O

```python
# Direct copy with optional virtual_mode for sandboxing
# from reference_repos/deepagents/libs/deepagents/deepagents/backends/filesystem.py
```

---

## Implementation Roadmap

### Milestone 1: Proof of Concept (1-2 days)

**Deliverables:**
- [ ] Copy `StateBackend` to LC Agent codebase
- [ ] Copy utility functions from `backends/utils.py`
- [ ] Add `files: Dict[str, FileData]` field to RunnableNetwork
- [ ] Create basic `FilesystemNetworkModifier` with `read_file` and `write_file` tools only
- [ ] Write integration test

**Success Criteria:**
```python
from lc_agent import RunnableNetwork
from lc_agent.modifiers import FilesystemNetworkModifier

network = RunnableNetwork(modifiers=[FilesystemNetworkModifier()])

with network:
    # Agent can call write_file and read_file during execution
    node = get_node_factory().create_node("ChatNode", inputs=[...])

# network.files should contain written files
assert "/test.txt" in network.files
```

### Milestone 2: Full Tool Suite (2-3 days)

**Deliverables:**
- [ ] Add all 6 tools: ls, read_file, write_file, edit_file, glob, grep
- [ ] Implement `LCAgentToolRuntime` shim
- [ ] Add system prompt injection (copy from DeepAgent's FILESYSTEM_SYSTEM_PROMPT)
- [ ] Handle both Command and string returns from tools
- [ ] Write comprehensive tests

**Success Criteria:**
- Agent can perform all file operations
- State updates work correctly
- Tools appear in model's function calling interface

### Milestone 3: Tool Result Eviction (1-2 days)

**Deliverables:**
- [ ] Implement `on_post_invoke_async` eviction logic
- [ ] Add `/large_tool_results/` directory management
- [ ] Create summary formatting
- [ ] Add configurable `tool_token_limit_before_evict`

**Success Criteria:**
- Large tool outputs automatically saved to files
- Agent receives summary with pagination instructions
- Context stays within token limits

### Milestone 4: Persistent Storage (2-3 days)

**Deliverables:**
- [ ] Copy `StoreBackend` to LC Agent
- [ ] Integrate with existing NetworkList implementations
- [ ] Create `FilesystemNetworkList` that persists files
- [ ] Add namespace support for multi-agent isolation

**Success Criteria:**
- Files persist across sessions
- Network structure and files both serialized
- Load/save roundtrip works correctly

### Milestone 5: Advanced Features (3-5 days)

**Deliverables:**
- [ ] Copy `CompositeBackend` for routing
- [ ] Copy `FilesystemBackend` for real I/O
- [ ] Add `SandboxBackendProtocol` support
- [ ] Implement conditional `execute` tool
- [ ] Integration with MultiAgentNetworkNode

**Success Criteria:**
- Hybrid storage (ephemeral + persistent)
- Real filesystem I/O option
- Sandbox execution support
- Multi-agent file sharing

---

## Code Examples

### Example 1: Basic Filesystem Integration

```python
from lc_agent import RunnableNetwork, RunnableNode
from lc_agent.modifiers.filesystem import FilesystemNetworkModifier
from lc_agent.backends.state import StateBackend

# Create network with filesystem support
network = RunnableNetwork(
    modifiers=[
        FilesystemNetworkModifier(
            backend=StateBackend,  # Factory function
            tool_token_limit_before_evict=20000
        )
    ]
)

with network:
    # Agent can now use file tools
    user_msg = RunnableNode(inputs=["Write a Python script to hello.py"])
    chat_node = get_node_factory().create_node("ChatNode")

    user_msg >> chat_node

# Execute
await network.ainvoke({"messages": []})

# Access files created during execution
print(network.files.keys())  # ['/hello.py', ...]
print(network.files['/hello.py']['content'])  # ['print("Hello")', ...]
```

### Example 2: Persistent File Storage

```python
from lc_agent.backends.store import StoreBackend
from lc_agent.backends.composite import CompositeBackend

def create_persistent_backend(runtime):
    """Create hybrid backend with ephemeral + persistent storage."""
    return CompositeBackend(
        default=StateBackend(runtime),  # Ephemeral
        routes={
            "/memories/": StoreBackend(runtime),  # Persistent
        }
    )

network = RunnableNetwork(
    modifiers=[
        FilesystemNetworkModifier(backend=create_persistent_backend)
    ],
    metadata={"assistant_id": "my-agent"}  # For namespace isolation
)

# Files in /memories/ persist across sessions
# All other files are ephemeral
```

### Example 3: Multi-Agent with Shared Files

```python
from lc_agent import MultiAgentNetworkNode

# Create multi-agent with filesystem
coordinator = MultiAgentNetworkNode(
    route_nodes=["researcher", "analyzer"],
    modifiers=[
        FilesystemNetworkModifier(backend=StateBackend)
    ]
)

# Researcher writes data to /research_data.json
# Analyzer reads from /research_data.json
# Files shared via network.files state
```

### Example 4: Tool Result Eviction

```python
# Automatically configured when tool_token_limit_before_evict is set
network = RunnableNetwork(
    modifiers=[
        FilesystemNetworkModifier(
            backend=StateBackend,
            tool_token_limit_before_evict=10000  # Evict results > 10k tokens
        )
    ]
)

# When a tool returns large result:
# 1. Saved to /large_tool_results/{tool_call_id}
# 2. Node receives summary: "Result saved to /large_tool_results/abc123. First 10 lines: ..."
# 3. Agent can use read_file("/large_tool_results/abc123", offset=0, limit=100) to paginate
```

---

## Detailed Integration Guide

### Step 1: Copy Backend Files

Create `source/modules/lc_agent/src/lc_agent/backends/` directory:

```bash
mkdir -p source/modules/lc_agent/src/lc_agent/backends
```

Copy these files from DeepAgent (NO modifications needed):
```bash
cp reference_repos/deepagents/libs/deepagents/deepagents/backends/protocol.py \
   source/modules/lc_agent/src/lc_agent/backends/

cp reference_repos/deepagents/libs/deepagents/deepagents/backends/utils.py \
   source/modules/lc_agent/src/lc_agent/backends/

cp reference_repos/deepagents/libs/deepagents/deepagents/backends/state.py \
   source/modules/lc_agent/src/lc_agent/backends/
```

### Step 2: Create ToolRuntime Shim

Create `source/modules/lc_agent/src/lc_agent/backends/runtime.py`:

```python
"""
Shim to adapt LC Agent's context to DeepAgent's ToolRuntime interface.
"""
from typing import Any, Dict, Optional

class LCAgentToolRuntime:
    """Provides DeepAgent ToolRuntime interface for LC Agent."""

    def __init__(self, network: "RunnableNetwork", node: Optional["RunnableNode"] = None):
        self._network = network
        self._node = node

    @property
    def state(self) -> Dict[str, Any]:
        """Expose network files as state['files']."""
        return {"files": self._network.files}

    @property
    def store(self):
        """Store access - TODO: integrate with NetworkList."""
        return None

    @property
    def config(self) -> Dict[str, Any]:
        """Expose network metadata as config."""
        return self._network.metadata.copy()
```

### Step 3: Create FilesystemNetworkModifier

Create `source/modules/lc_agent/src/lc_agent/modifiers/filesystem.py`:

```python
"""
Filesystem Network Modifier - provides file system tools to LC Agent.
Directly uses DeepAgent's backend implementations.
"""
from typing import Callable, List, Optional
from langchain_core.tools import BaseTool

from lc_agent import NetworkModifier, RunnableNetwork, RunnableNode
from lc_agent.backends.protocol import BackendProtocol
from lc_agent.backends.state import StateBackend
from lc_agent.backends.runtime import LCAgentToolRuntime


class FilesystemNetworkModifier(NetworkModifier):
    """
    Provides file system tools using DeepAgent's backend system.

    Args:
        backend: Backend implementation or factory function
        tool_token_limit_before_evict: Token threshold for result eviction
    """

    def __init__(
        self,
        backend: BackendProtocol | Callable[[LCAgentToolRuntime], BackendProtocol] | None = None,
        tool_token_limit_before_evict: int = 20000
    ):
        self.backend = backend or StateBackend
        self.tool_token_limit = tool_token_limit_before_evict
        self._system_prompt = self._build_system_prompt()

    def _get_backend(self, network: RunnableNetwork) -> BackendProtocol:
        """Resolve backend (handle factory pattern)."""
        if callable(self.backend):
            runtime = LCAgentToolRuntime(network)
            return self.backend(runtime)
        return self.backend

    def _build_system_prompt(self) -> str:
        """Build system prompt describing filesystem tools."""
        # Copy from deepagents.middleware.filesystem.FILESYSTEM_SYSTEM_PROMPT
        return """
You have access to filesystem tools:
- ls(path): List files
- read_file(file_path, offset=0, limit=500): Read file with pagination
- write_file(file_path, content): Create new file
- edit_file(file_path, old_string, new_string, replace_all=False): Replace text
- glob(pattern, path="/"): Find files matching pattern
- grep(pattern, path="/", glob=None, output_mode="files_with_matches"): Search contents

All paths must be absolute (start with /).
"""

    def _create_ls_tool(self, network: RunnableNetwork) -> BaseTool:
        """Create ls tool."""
        backend = self._get_backend(network)

        def ls_func(path: str) -> str:
            """List files and directories at path."""
            results = backend.ls_info(path)
            return "\n".join(r["path"] for r in results)

        from langchain_core.tools import tool
        return tool(ls_func)

    def _create_read_file_tool(self, network: RunnableNetwork) -> BaseTool:
        """Create read_file tool."""
        backend = self._get_backend(network)

        def read_file_func(file_path: str, offset: int = 0, limit: int = 500) -> str:
            """Read file contents with line numbers."""
            return backend.read(file_path, offset, limit)

        from langchain_core.tools import tool
        return tool(read_file_func)

    def _create_write_file_tool(self, network: RunnableNetwork) -> BaseTool:
        """Create write_file tool."""
        backend = self._get_backend(network)

        def write_file_func(file_path: str, content: str) -> str:
            """Create new file."""
            result = backend.write(file_path, content)

            if result.error:
                return result.error

            # Update network files if state-backed
            if result.files_update:
                network.files.update(result.files_update)

            return f"Successfully wrote to {result.path}"

        from langchain_core.tools import tool
        return tool(write_file_func)

    def _create_edit_file_tool(self, network: RunnableNetwork) -> BaseTool:
        """Create edit_file tool."""
        backend = self._get_backend(network)

        def edit_file_func(
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False
        ) -> str:
            """Replace old_string with new_string in file."""
            result = backend.edit(file_path, old_string, new_string, replace_all)

            if result.error:
                return result.error

            # Update network files if state-backed
            if result.files_update:
                network.files.update(result.files_update)

            return f"Edited {result.path} ({result.occurrences} occurrences)"

        from langchain_core.tools import tool
        return tool(edit_file_func)

    def _create_glob_tool(self, network: RunnableNetwork) -> BaseTool:
        """Create glob tool."""
        backend = self._get_backend(network)

        def glob_func(pattern: str, path: str = "/") -> str:
            """Find files matching glob pattern."""
            results = backend.glob_info(pattern, path)
            return "\n".join(r["path"] for r in results)

        from langchain_core.tools import tool
        return tool(glob_func)

    def _create_grep_tool(self, network: RunnableNetwork) -> BaseTool:
        """Create grep tool."""
        backend = self._get_backend(network)

        def grep_func(
            pattern: str,
            path: str = "/",
            glob: Optional[str] = None,
            output_mode: str = "files_with_matches"
        ) -> str:
            """Search file contents using regex."""
            results = backend.grep_raw(pattern, path, glob)

            # Format based on output_mode
            if isinstance(results, str):  # Error
                return results

            if output_mode == "files_with_matches":
                unique_paths = list(dict.fromkeys(r["path"] for r in results))
                return "\n".join(unique_paths)
            elif output_mode == "content":
                return "\n".join(f"{r['path']}:{r['line']}:{r['text']}" for r in results)
            elif output_mode == "count":
                from collections import Counter
                counts = Counter(r["path"] for r in results)
                return "\n".join(f"{path}: {count}" for path, count in counts.items())

        from langchain_core.tools import tool
        return tool(grep_func)

    def on_begin_invoke(self, network: RunnableNetwork):
        """Register tools when network starts."""
        # Create all tools
        tools = [
            self._create_ls_tool(network),
            self._create_read_file_tool(network),
            self._create_write_file_tool(network),
            self._create_edit_file_tool(network),
            self._create_glob_tool(network),
            self._create_grep_tool(network),
        ]

        # Store in network metadata for access by nodes
        network.metadata['filesystem_tools'] = tools

        # Inject system prompt into network
        # (Nodes can access via network.metadata['filesystem_system_prompt'])
        network.metadata['filesystem_system_prompt'] = self._system_prompt

    async def on_post_invoke_async(self, network: RunnableNetwork, node: RunnableNode):
        """Check for large tool results and evict if needed."""
        if not node.outputs or not hasattr(node.outputs, 'content'):
            return

        content = str(node.outputs.content)

        # Check size (rough estimate: 4 chars per token)
        if len(content) > 4 * self.tool_token_limit:
            # Evict to filesystem
            await self._evict_large_result(network, node, content)

    async def _evict_large_result(
        self,
        network: RunnableNetwork,
        node: RunnableNode,
        content: str
    ):
        """Save large result to file and replace with summary."""
        # Generate file path
        tool_call_id = node.metadata.get('tool_call_id', node.uuid)
        sanitized_id = tool_call_id.replace('/', '_').replace('\\', '_')
        file_path = f"/large_tool_results/{sanitized_id}"

        # Save to filesystem
        backend = self._get_backend(network)
        result = backend.write(file_path, content)

        if result.error:
            # Failed to evict - leave as is
            return

        # Update network files if state-backed
        if result.files_update:
            network.files.update(result.files_update)

        # Create summary
        lines = content.split('\n')
        preview = '\n'.join(lines[:10])
        summary = (
            f"Tool result too large ({len(content)} chars, ~{len(content)//4} tokens). "
            f"Saved to {file_path}.\n\n"
            f"First 10 lines:\n{preview}\n\n"
            f"... [Use read_file('{file_path}', offset=0, limit=100) to view more]"
        )

        # Replace node output
        node.outputs.content = summary
```

### Step 4: Add Files Field to RunnableNetwork

Modify `source/modules/lc_agent/src/lc_agent/runnable_network.py`:

```python
from typing import Dict
from pydantic import Field

class RunnableNetwork(RunnableSerializable):
    nodes: List[RunnableNode] = []
    modifiers: Dict = {}
    callbacks: Dict = {}
    default_node: str = ""
    chat_model_name: Optional[str] = None
    metadata: Dict[str, Any] = {}
    files: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # NEW
```

### Step 5: Update Serialization

Modify `serialize_model` in `runnable_network.py` to include files:

```python
@model_serializer
def serialize_model(self) -> Dict[str, Any]:
    result = {field: value for field, value in self
              if field not in ["modifiers", "callbacks", "parents"]}

    # Existing connection serialization
    result["__connections__"] = {...}

    # NEW: Include files
    result["files"] = self.files

    return result
```

### Step 6: Write Integration Test

Create `source/modules/lc_agent/tests/test_filesystem.py`:

```python
import pytest
from lc_agent import RunnableNetwork
from lc_agent.modifiers.filesystem import FilesystemNetworkModifier
from lc_agent.backends.state import StateBackend


@pytest.mark.asyncio
async def test_filesystem_basic():
    """Test basic file operations."""
    network = RunnableNetwork(
        modifiers=[FilesystemNetworkModifier(backend=StateBackend)]
    )

    # Get write_file tool
    tools = network.metadata['filesystem_tools']
    write_tool = next(t for t in tools if t.name == 'write_file')
    read_tool = next(t for t in tools if t.name == 'read_file')

    # Write file
    result = write_tool.invoke({"file_path": "/test.txt", "content": "Hello\nWorld"})
    assert "Successfully wrote" in result

    # Check network.files
    assert "/test.txt" in network.files
    assert network.files["/test.txt"]["content"] == ["Hello", "World"]

    # Read file
    content = read_tool.invoke({"file_path": "/test.txt"})
    assert "Hello" in content
    assert "World" in content


@pytest.mark.asyncio
async def test_filesystem_eviction():
    """Test large result eviction."""
    network = RunnableNetwork(
        modifiers=[
            FilesystemNetworkModifier(
                backend=StateBackend,
                tool_token_limit_before_evict=100  # Low threshold for testing
            )
        ]
    )

    # Create node with large output
    from lc_agent import RunnableNode
    node = RunnableNode()
    node.outputs = type('Output', (), {'content': 'x' * 1000})()
    node.metadata = {'tool_call_id': 'test123'}

    # Trigger eviction
    modifier = network.modifiers[0]
    await modifier.on_post_invoke_async(network, node)

    # Check eviction occurred
    assert node.outputs.content != 'x' * 1000
    assert "Saved to /large_tool_results/test123" in node.outputs.content
    assert "/large_tool_results/test123" in network.files
```

---

## Critical Success Factors

### 1. Minimal Modifications to DeepAgent Code

**Why:** DeepAgent's backends are production-tested and well-designed.

**Strategy:**
- Copy files as-is
- Use shim layer (LCAgentToolRuntime) to adapt interfaces
- No modifications to backend logic

### 2. Leverage LC Agent's Existing Patterns

**Why:** Maintain consistency with existing LC Agent architecture.

**Strategy:**
- NetworkModifier (not AgentMiddleware)
- RunnableNode outputs (not Command pattern initially)
- Pydantic serialization (already compatible)

### 3. Incremental Rollout

**Why:** Reduce risk, enable early testing.

**Strategy:**
- Phase 1: StateBackend only (no dependencies)
- Phase 2: Tool eviction (high value)
- Phase 3: Persistent storage (when needed)
- Phase 4: Advanced features (optional)

### 4. Maintain Backward Compatibility

**Why:** Don't break existing LC Agent users.

**Strategy:**
- Files field defaults to empty dict
- Filesystem modifier is optional
- Existing networks serialize/deserialize without changes

---

## Open Questions & Decisions

### Q1: Should we use Command pattern or modify node outputs?

**Option A: Direct Output Modification (Simpler)**
```python
# Tool modifies network.files directly
network.files["/file.txt"] = FileData(...)
return "File written successfully"
```

**Option B: Command Pattern (More aligned with DeepAgent)**
```python
# Tool returns Command-like object
return {"update": {"files": {"/file.txt": FileData(...)}}, "message": "..."}
```

**Recommendation:** Start with Option A for simplicity. Can add Command pattern later if needed for atomic updates.

### Q2: How to integrate with existing NetworkList persistence?

**Option A: Extend NetworkList (Simpler)**
```python
class JsonNetworkList(NetworkList):
    def save(self, network):
        # Existing: Save network structure
        # New: Save network.files
```

**Option B: Separate FileStore (More flexible)**
```python
FileStore.save(network.files, network_id)
NetworkList.save(network)  # Unchanged
```

**Recommendation:** Option A for MVP, Option B if files become very large.

### Q3: Should execute tool be included in Phase 1?

**No.** Execute tool requires SandboxBackendProtocol which adds complexity. Focus on read/write first.

### Q4: How to handle tool registration with chat models?

**Option A: Auto-register in NetworkModifier.on_begin_invoke**
```python
def on_begin_invoke(self, network):
    tools = self._create_tools()
    # Store in metadata for nodes to access
    network.metadata['filesystem_tools'] = tools
```

**Option B: Explicit registration in ChatNode**
```python
class ChatNode(RunnableNode):
    def invoke(self, input, config):
        tools = self.network.metadata.get('filesystem_tools', [])
        chat_model = get_chat_model().bind_tools(tools)
```

**Recommendation:** Both - A stores tools, B uses them. This matches LC Agent's existing pattern.

---

## Conclusion

The integration of DeepAgent's file system tools into LC Agent is **highly feasible** with minimal risk:

**Key Advantages:**
1. ✅ DeepAgent's backends are **standalone** - can copy directly
2. ✅ NetworkModifier maps 1:1 to AgentMiddleware
3. ✅ StateBackend has **zero dependencies** - perfect starting point
4. ✅ Tool implementations are LangChain-compatible
5. ✅ Pydantic serialization already compatible

**Recommended Approach:**
1. **Phase 1 (MVP):** StateBackend + 6 tools + basic modifier (2-3 days)
2. **Phase 2:** Tool result eviction (1-2 days)
3. **Phase 3:** Persistent storage when needed (2-3 days)
4. **Phase 4:** Advanced features as requirements emerge

**Expected Benefits:**
- Agents can create/read files during execution
- Context management via tool eviction
- Cross-session memory via persistent storage
- Hybrid storage strategies via CompositeBackend
- **Enables deep research workflows like DeepAgent's research example**

The architecture is sound. The implementation is straightforward. The value is high.

**Next Step:** Begin Phase 1 implementation with StateBackend + basic tools.
