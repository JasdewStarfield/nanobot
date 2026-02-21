import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app

runner = CliRunner()


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.utils.helpers.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_gateway_heartbeat_uses_main_model_when_not_configured(monkeypatch):
    from types import SimpleNamespace

    from nanobot.config.schema import Config
    from nanobot.cli import commands

    class FakeAgentLoop:
        instances = []

        def __init__(self, **kwargs):
            self.calls = []
            FakeAgentLoop.instances.append(self)

        async def process_direct(self, prompt: str, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            return "ok"

        async def run(self):
            return None

        async def close_mcp(self):
            return None

        def stop(self):
            return None

    class FakeHeartbeatService:
        def __init__(self, workspace, on_heartbeat, interval_s, enabled):
            self.on_heartbeat = on_heartbeat

        async def start(self):
            await self.on_heartbeat("heartbeat prompt")

        def stop(self):
            return None

    class FakeCronService:
        def __init__(self, _store_path):
            self.on_job = None

        def status(self):
            return {"jobs": 1}

        async def start(self):
            if self.on_job:
                job = SimpleNamespace(
                    id="job-1",
                    payload=SimpleNamespace(
                        message="cron ping",
                        channel="cli",
                        to="direct",
                        deliver=False,
                    ),
                )
                await self.on_job(job)

        def stop(self):
            return None

    class FakeChannelManager:
        def __init__(self, _config, _bus):
            self.enabled_channels = []

        async def start_all(self):
            return None

        async def stop_all(self):
            return None

    config = Config.model_validate(
        {
            "agents": {"defaults": {"model": "main-model"}},
            "heartbeat": {"model": "", "contextSessionKey": "shared-session"},
        }
    )

    provider_calls = []

    def fake_make_provider(_config, model=None):
        provider_calls.append(model or _config.agents.defaults.model)
        return object()

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: Path("/tmp"))
    monkeypatch.setattr("nanobot.cli.commands._make_provider", fake_make_provider)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", FakeAgentLoop)
    monkeypatch.setattr("nanobot.heartbeat.service.HeartbeatService", FakeHeartbeatService)
    monkeypatch.setattr("nanobot.cron.service.CronService", FakeCronService)
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", FakeChannelManager)
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.session.manager.SessionManager", lambda _workspace: object())

    commands.gateway(port=18790, verbose=False)

    assert provider_calls == ["main-model"]
    assert len(FakeAgentLoop.instances) == 1
    assert [call["session_key"] for call in FakeAgentLoop.instances[0].calls] == ["cron:job-1", "heartbeat"]


def test_gateway_heartbeat_uses_dedicated_model_when_configured(monkeypatch):
    from types import SimpleNamespace

    from nanobot.config.schema import Config
    from nanobot.cli import commands

    class FakeAgentLoop:
        instances = []

        def __init__(self, **kwargs):
            self.calls = []
            FakeAgentLoop.instances.append(self)

        async def process_direct(self, prompt: str, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            return "ok"

        async def run(self):
            return None

        async def close_mcp(self):
            return None

        def stop(self):
            return None

    class FakeHeartbeatService:
        def __init__(self, workspace, on_heartbeat, interval_s, enabled):
            self.on_heartbeat = on_heartbeat

        async def start(self):
            await self.on_heartbeat("heartbeat prompt")

        def stop(self):
            return None

    class FakeCronService:
        def __init__(self, _store_path):
            self.on_job = None

        def status(self):
            return {"jobs": 1}

        async def start(self):
            if self.on_job:
                job = SimpleNamespace(
                    id="job-1",
                    payload=SimpleNamespace(
                        message="cron ping",
                        channel="cli",
                        to="direct",
                        deliver=False,
                    ),
                )
                await self.on_job(job)

        def stop(self):
            return None

    class FakeChannelManager:
        def __init__(self, _config, _bus):
            self.enabled_channels = []

        async def start_all(self):
            return None

        async def stop_all(self):
            return None

    config = Config.model_validate(
        {
            "agents": {"defaults": {"model": "main-model"}},
            "heartbeat": {"model": "heartbeat-model", "contextSessionKey": "shared-session"},
        }
    )

    provider_calls = []

    def fake_make_provider(_config, model=None):
        provider_calls.append(model or _config.agents.defaults.model)
        return object()

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: config)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: Path("/tmp"))
    monkeypatch.setattr("nanobot.cli.commands._make_provider", fake_make_provider)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", FakeAgentLoop)
    monkeypatch.setattr("nanobot.heartbeat.service.HeartbeatService", FakeHeartbeatService)
    monkeypatch.setattr("nanobot.cron.service.CronService", FakeCronService)
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", FakeChannelManager)
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.session.manager.SessionManager", lambda _workspace: object())

    commands.gateway(port=18790, verbose=False)

    assert provider_calls == ["main-model", "heartbeat-model"]
    assert len(FakeAgentLoop.instances) == 2
    assert [call["session_key"] for call in FakeAgentLoop.instances[0].calls] == ["cron:job-1"]
    assert [call["session_key"] for call in FakeAgentLoop.instances[1].calls] == ["heartbeat"]
    assert FakeAgentLoop.instances[1].calls[0]["context_session_key"] == "shared-session"
