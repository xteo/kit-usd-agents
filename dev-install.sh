#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default Python environment (use current environment if not specified)
PYTHON_CMD="${PYTHON:-python3}"
INSTALL_MODE="${1:-editable}"

print_info "================================================"
print_info "LC Agent Development Installation Script"
print_info "================================================"
print_info ""
print_info "Python executable: $PYTHON_CMD"
print_info "Install mode: $INSTALL_MODE"
print_info ""

# Verify Python is available
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    print_error "Python executable not found: $PYTHON_CMD"
    print_error "Please specify Python path: PYTHON=/path/to/python $0"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Verify we're in a virtual environment (recommended)
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_warn "Not running in a virtual environment!"
    print_warn "It's recommended to create and activate a virtual environment first:"
    print_warn "  python3 -m venv venv"
    print_warn "  source venv/bin/activate"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_info ""
print_info "Installing LC Agent module..."
print_info ""

cd "$SCRIPT_DIR/source/modules/lc_agent"

# Upgrade pip first
print_info "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip || print_warn "Failed to upgrade pip, continuing anyway..."

print_info ""
if [[ "$INSTALL_MODE" == "editable" || "$INSTALL_MODE" == "-e" ]]; then
    print_info "Installing in editable mode (changes to source will be reflected immediately)..."
    print_info "Running: pip install -e ."
    $PYTHON_CMD -m pip install -e . --verbose
else
    print_info "Installing in regular mode..."
    print_info "Running: pip install ."
    $PYTHON_CMD -m pip install . --verbose
fi

print_info ""
print_info "================================================"
print_info "Installation complete!"
print_info "================================================"
print_info ""
print_info "You can now import lc_agent in your Python scripts:"
print_info "  from lc_agent import RunnableNetwork, RunnableNode"
print_info ""
print_info "To run the CLI assistant:"
print_info "  ./run-lc-agent.sh"
print_info ""
print_info "For more information, see claude.md in the repository root."
