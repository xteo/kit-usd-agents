#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                    ║${NC}"
echo -e "${GREEN}║  LC Agent Parallel Execution Test - CLI Mode                       ║${NC}"
echo -e "${GREEN}║                                                                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Use the Python from the environment, or default to python3
PYTHON_CMD="${PYTHON:-python3}"

# Check if Python is available
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python executable not found: $PYTHON_CMD"
    echo "Please specify Python path: PYTHON=/path/to/python $0"
    exit 1
fi

# Check for NVIDIA_API_KEY
if [ -z "$NVIDIA_API_KEY" ]; then
    echo -e "${YELLOW}[WARNING]${NC} NVIDIA_API_KEY environment variable not set!"
    echo ""
    echo "This test requires an NVIDIA API key to make real LLM calls."
    echo ""
    echo "To get an API key:"
    echo "  1. Go to https://build.nvidia.com/"
    echo "  2. Sign in and navigate to any model"
    echo "  3. Click 'Get API Key'"
    echo ""
    echo "Then set it:"
    echo "  export NVIDIA_API_KEY='your_key_here'"
    echo ""
    echo "Or run this script with:"
    echo "  NVIDIA_API_KEY='your_key_here' ./run-cli-parallel-test.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Python executable: $PYTHON_CMD"
echo -e "${GREEN}[INFO]${NC} API key found: ${NVIDIA_API_KEY:0:10}...${NVIDIA_API_KEY: -4}"
echo ""

# Run the test
echo -e "${GREEN}[INFO]${NC} Running parallel execution test with real LLMs..."
echo ""

exec "$PYTHON_CMD" test_cli_parallel_execution.py
