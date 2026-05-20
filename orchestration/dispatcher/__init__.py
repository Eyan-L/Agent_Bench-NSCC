"""Harness-agnostic dispatcher package.

A *dispatcher* owns the per-run state that lets an agent runtime reach an
execution backend. See `base.Dispatcher` for the contract.
"""
from orchestration.dispatcher.base import Dispatcher
from orchestration.dispatcher.hermes import HermesDispatcher
from orchestration.dispatcher.openclaw_remote import OpenClawRemoteDispatcher
from orchestration.dispatcher.openhands import OpenHandsDispatcher

__all__ = [
    "Dispatcher",
    "HermesDispatcher",
    "OpenClawRemoteDispatcher",
    "OpenHandsDispatcher",
]
