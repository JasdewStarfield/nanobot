"""Tests for parallel tool execution in agent and subagent loops."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


class _DummyProvider:
    def get_default_model(self) -> str:
        return "test-model"

    async def chat_with_retry(self, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_calls_concurrently(tmp_path):
    bus = MessageBus()
    provider = _DummyProvider()
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)

    loop.provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(id="call_1", name="t1", arguments={}),
                ToolCallRequest(id="call_2", name="t2", arguments={}),
            ],
        ),
        LLMResponse(content="done", tool_calls=[]),
    ])

    async def delayed_execute(name, arguments):
        await asyncio.sleep(0.1)
        return f"result-{name}"

    loop.tools.execute = AsyncMock(side_effect=delayed_execute)

    start = time.perf_counter()
    final_content, _, messages = await loop._run_agent_loop([
        {"role": "user", "content": "run"},
    ])
    elapsed = time.perf_counter() - start

    assert final_content == "done"
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert elapsed < 0.18


@pytest.mark.asyncio
async def test_subagent_executes_tool_calls_concurrently(tmp_path, monkeypatch):
    bus = MessageBus()
    provider = _DummyProvider()
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)

    calls = {"n": 0}

    async def scripted_chat_with_retry(*, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="call_1", name="t1", arguments={}),
                    ToolCallRequest(id="call_2", name="t2", arguments={}),
                ],
            )
        return LLMResponse(content="done", tool_calls=[])

    provider.chat_with_retry = scripted_chat_with_retry

    async def delayed_execute(self, name, arguments):
        await asyncio.sleep(0.1)
        return f"result-{name}"

    monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", delayed_execute)

    mgr._announce_result = AsyncMock(return_value=None)

    start = time.perf_counter()
    await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})
    elapsed = time.perf_counter() - start

    assert elapsed < 0.18
