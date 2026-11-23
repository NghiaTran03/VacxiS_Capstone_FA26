"""Share the complete FastAPI application through Gradio's tunnel."""

import asyncio
import logging
from uuid import uuid4

from gradio import networking
from gradio.tunneling import CURRENT_TUNNELS
import uvicorn

logger = logging.getLogger(__name__)


class _ShareServer(uvicorn.Server):
    """Create a tunnel after binding the local server and own its cleanup."""

    def __init__(self, config: uvicorn.Config):
        super().__init__(config)
        self.share_token = uuid4().hex

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.should_exit:
            return
        try:
            url = await asyncio.to_thread(
                networking.setup_tunnel,
                local_host="127.0.0.1",
                local_port=self.config.port,
                share_token=self.share_token,
                share_server_address=None,
                share_server_tls_certificate=None,
            )
        except Exception:
            self.close_tunnel()
            logger.warning(
                "Could not create a Gradio share link. Check your Internet connection "
                "and Gradio tunnel availability. Local UI remains available at http://127.0.0.1:%s",
                self.config.port,
            )
            return
        self.config.app.state.share_url = url
        print(f"* Running on public URL: {url}", flush=True)
        print("Keep this terminal and computer running while others use the link.", flush=True)

    def close_tunnel(self):
        """Close only tunnels created by this server, including failed attempts."""
        for tunnel in list(CURRENT_TUNNELS):
            if tunnel.share_token == self.share_token:
                tunnel.kill()
                CURRENT_TUNNELS.remove(tunnel)


def run_server(app, *, port: int, share: bool = False):
    """Keep the same local server with an optional Gradio public tunnel."""
    if not share:
        uvicorn.run(app, host="127.0.0.1", port=port)
        return
    server = _ShareServer(uvicorn.Config(app, host="127.0.0.1", port=port))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close_tunnel()
