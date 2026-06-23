from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

_TARGET_MODULE = "sglang.srt.entrypoints.openai.serving_chat"
_DISABLE_THINKING_ENV = "SGLANG_DISABLE_THINKING_BY_DEFAULT"
_THINKING_ENABLED_BY_DEFAULT_PARSERS = {"qwen3", "glm45", "nano_v3", "interns1"}
_ORIGINAL_ATTR = "_agent_bench_original_convert_to_internal_request"


def _env_bool(name: str) -> bool:
    raw = os.getenv(name)
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_disable_thinking_default(self: Any, request: Any) -> None:
    if not _env_bool(_DISABLE_THINKING_ENV):
        return
    if getattr(self, "reasoning_parser", None) not in _THINKING_ENABLED_BY_DEFAULT_PARSERS:
        return

    kwargs = getattr(request, "chat_template_kwargs", None)
    if kwargs is None:
        request.chat_template_kwargs = {"enable_thinking": False}
        return
    if not isinstance(kwargs, dict):
        return
    if "enable_thinking" in kwargs or "thinking" in kwargs:
        return
    kwargs["enable_thinking"] = False


def _patch_module(module: ModuleType) -> None:
    cls = getattr(module, "OpenAIServingChat", None)
    if cls is None or hasattr(cls, _ORIGINAL_ATTR):
        return

    original = cls._convert_to_internal_request
    setattr(cls, _ORIGINAL_ATTR, original)

    def _patched_convert_to_internal_request(self: Any, request: Any, raw_request: Any = None):
        _apply_disable_thinking_default(self, request)
        return original(self, request, raw_request)

    cls._apply_disable_thinking_default = _apply_disable_thinking_default
    cls._convert_to_internal_request = _patched_convert_to_internal_request


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_module(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _PatchLoader):
            return spec
        spec.loader = _PatchLoader(spec.loader)
        return spec


if _TARGET_MODULE in sys.modules:
    _patch_module(sys.modules[_TARGET_MODULE])
else:
    sys.meta_path.insert(0, _PatchFinder())
