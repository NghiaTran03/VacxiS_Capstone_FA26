"""Verify whole-server sharing, failure fallback and tunnel ownership."""

import asyncio
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
import uvicorn

from vacxis_capstone_fa26.web import sharing


def test_share_starts_after_local_binding_and_targets_outer_server(monkeypatch, capsys):
    app = FastAPI()
    server = sharing._ShareServer(uvicorn.Config(app, port=8123))
    events = []

    async def startup(self, sockets=None):
        events.append("bound")

    def setup(**kwargs):
        assert events == ["bound"]
        assert kwargs["local_host"] == "127.0.0.1"
        assert kwargs["local_port"] == 8123
        assert kwargs["share_token"] == server.share_token
        return "https://demo.gradio.live"

    monkeypatch.setattr(uvicorn.Server, "startup", startup)
    monkeypatch.setattr(sharing.networking, "setup_tunnel", setup)
    asyncio.run(server.startup())
    assert app.state.share_url == "https://demo.gradio.live"
    assert "Running on public URL: https://demo.gradio.live" in capsys.readouterr().out


def test_failed_share_keeps_local_server_and_cleans_only_own_tunnel(monkeypatch, caplog):
    server = sharing._ShareServer(uvicorn.Config(FastAPI(), port=8123))
    own = Mock(share_token=server.share_token)
    other = Mock(share_token="another-app")
    tunnels = [own, other]
    monkeypatch.setattr(sharing, "CURRENT_TUNNELS", tunnels)
    monkeypatch.setattr(uvicorn.Server, "startup", AsyncMock())
    monkeypatch.setattr(sharing.networking, "setup_tunnel", Mock(side_effect=RuntimeError("unavailable")))
    asyncio.run(server.startup())
    own.kill.assert_called_once()
    other.kill.assert_not_called()
    assert tunnels == [other]
    assert not server.should_exit
    assert "Local UI remains available" in caplog.text


def test_unbound_server_does_not_publish_link(monkeypatch):
    server = sharing._ShareServer(uvicorn.Config(FastAPI(), port=8123))
    server.should_exit = True
    setup = Mock()
    monkeypatch.setattr(uvicorn.Server, "startup", AsyncMock())
    monkeypatch.setattr(sharing.networking, "setup_tunnel", setup)
    asyncio.run(server.startup())
    setup.assert_not_called()


def test_local_mode_does_not_open_tunnel(monkeypatch):
    run = Mock()
    setup = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(sharing.networking, "setup_tunnel", setup)
    app = FastAPI()
    sharing.run_server(app, port=8123)
    run.assert_called_once_with(app, host="127.0.0.1", port=8123)
    setup.assert_not_called()


def test_share_mode_closes_tunnel_when_interrupted(monkeypatch):
    server = Mock()
    server.run.side_effect = KeyboardInterrupt
    monkeypatch.setattr(sharing, "_ShareServer", Mock(return_value=server))
    sharing.run_server(FastAPI(), port=8123, share=True)
    server.close_tunnel.assert_called_once()
