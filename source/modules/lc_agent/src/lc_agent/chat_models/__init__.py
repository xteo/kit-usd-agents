## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
from .chat_nvcf import ChatNVCF


def register_all():
    """Register default NVCF chat models for CLI usage."""
    from lc_agent import get_chat_model_registry

    registry = get_chat_model_registry()

    # Get API key from environment
    api_key = os.environ.get("NVIDIA_API_KEY")

    # Register openai/gpt-oss-120b as default model
    registry.register(
        "openai/gpt-oss-120b",
        ChatNVCF(
            model="openai/gpt-oss-120b",
            max_tokens=4096,
            temperature=0.1,
            api_token=api_key,
        ),
        None,  # tokenizer (optional)
        128 * 1024 - 4096,  # max context tokens
        False,  # not hidden
    )

    # Also register as "gpt-4" alias for backward compatibility
    registry.register(
        "gpt-4",
        ChatNVCF(
            model="openai/gpt-oss-120b",
            max_tokens=4096,
            temperature=0.1,
            api_token=api_key,
        ),
        None,
        128 * 1024 - 4096,
        False,
    )

    # Register meta/llama-4-maverick model as well
    registry.register(
        "meta/llama-4-maverick-17b-128e-instruct",
        ChatNVCF(
            model="meta/llama-4-maverick-17b-128e-instruct",
            max_tokens=4096,
            temperature=0.0,
            api_token=api_key,
        ),
        None,
        256 * 1024 - 4096,
        False,
    )


__all__ = ["ChatNVCF", "register_all"]
