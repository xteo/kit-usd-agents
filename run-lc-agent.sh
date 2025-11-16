#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use the Python from the environment, or default to python3
PYTHON_CMD="${PYTHON:-python3}"

# Check if lc_agent_cli is installed
if ! $PYTHON_CMD -c "import lc_agent_cli" 2>/dev/null; then
    echo "[ERROR] lc_agent_cli is not installed in the current Python environment"
    echo ""
    echo "Please run the installation script first:"
    echo "  ./dev-install.sh"
    echo ""
    echo "Or install it manually:"
    echo "  cd source/modules/lc_agent_cli && pip install -e ."
    exit 1
fi

# Run the CLI with all arguments passed through
exec $PYTHON_CMD -m lc_agent_cli "$@"
