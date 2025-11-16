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


def register_all(verbose=False):
    """Register default NVCF chat models for CLI usage."""
    from lc_agent import get_chat_model_registry

    registry = get_chat_model_registry()

    # Get API key from environment
    api_key = os.environ.get("NVIDIA_API_KEY")
    if verbose:
        print(f"[DEBUG] Registering NVCF models, API key present: {api_key is not None}")

    try:
        # Register gpt-120b (openai/gpt-oss-120b on NVIDIA Build)
        if verbose:
            print("[DEBUG] Registering gpt-120b...")
        registry.register(
            "gpt-120b",
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
        if verbose:
            print("[DEBUG] Successfully registered gpt-120b")

        # Also register full name for clarity
        if verbose:
            print("[DEBUG] Registering openai/gpt-oss-120b...")
        registry.register(
            "openai/gpt-oss-120b",
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
        if verbose:
            print("[DEBUG] Successfully registered openai/gpt-oss-120b")

        # Register meta/llama-4-maverick model as well
        if verbose:
            print("[DEBUG] Registering llama-maverick...")
        registry.register(
            "llama-maverick",
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
        if verbose:
            print("[DEBUG] Successfully registered llama-maverick")

        if verbose:
            print(f"[DEBUG] Total models in registry: {len(registry.registered_names)}")
            print(f"[DEBUG] Registered model names: {registry.get_registered_names()}")
            print(f"[DEBUG] All registered names (including hidden): {registry.registered_names}")

    except Exception as e:
        print(f"[ERROR] Failed to register chat models: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


__all__ = ["ChatNVCF", "register_all"]
