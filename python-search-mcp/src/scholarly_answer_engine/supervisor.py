"""Ensures the router is reachable, starting it when it is not.

The router is a separate Node service, so "run the MCP server" alone leaves
synthesis broken. Rather than requiring the user to launch two processes by hand,
the engine probes the router's health endpoint and, if it is not listening,
spawns it as a child process and waits for it to come up.

The probe is cheap and the spawn is retried once; a router that cannot be started
is reported loudly at startup rather than surfacing as a 404 on the first request.
Graceful degradation is preserved: the ``papers`` tool needs no LLM at all, so a
missing router never makes the whole server unusable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


class RouterSupervisor:
    def __init__(
        self,
        router_url: str,
        router_dir: str | None = None,
        port: int | None = None,
        startup_timeout: float = 45.0,
    ) -> None:
        self.router_url = router_url.rstrip("/")
        self.health_url = f"{self.router_url}/healthz"
        self.router_dir = router_dir
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None

    # -- discovery -------------------------------------------------------

    def _default_router_dir(self) -> Path | None:
        """Look for the router next to this project, or in the sibling repo."""
        if self.router_dir:
            candidate = Path(self.router_dir)
            return candidate if (candidate / "dist" / "index.js").exists() else None
        # This project deliberately omits the parent, so look up from the repo root.
        here = Path(__file__).resolve().parents[2]
        for parent in [here, *here.parents[:3]]:
            candidate = parent / "scholarly-answer-router"
            if (candidate / "dist" / "index.js").exists():
                return candidate
        return None

    def _port_from_url(self) -> int | None:
        if self.port is not None:
            return self.port
        try:
            parsed = urllib.parse.urlparse(self.router_url)
            return int(parsed.port) if parsed.port else None
        except (AttributeError, ValueError):
            return None

    # -- lifecycle -------------------------------------------------------

    def is_reachable(self) -> bool:
        try:
            with urllib.request.urlopen(self.health_url, timeout=3) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def ensure(self) -> bool:
        """Make sure a router is reachable. Returns True when it is."""
        if self.is_reachable():
            logger.info("router already reachable at %s", self.router_url)
            return True

        if not self.start():
            return False

        return self._wait_for_health()

    def start(self) -> bool:
        router_dir = self._default_router_dir()
        if router_dir is None:
            logger.warning(
                "router is not reachable at %s and no built router was found on disk; "
                "set ROUTER_URL to a running router, or build scholarly-answer-router",
                self.router_url,
            )
            return False

        port = self._port_from_url() or 8787
        logger.info("starting router from %s on port %d", router_dir, port)
        try:
            self.process = subprocess.Popen(
                ["node", "dist/index.js"],
                cwd=str(router_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "ROUTER_PORT": str(port)},
            )
        except (OSError, ValueError) as exc:
            logger.warning("could not start the router: %s", exc)
            return False
        return True

    def _wait_for_health(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.is_reachable():
                logger.info("router is up at %s", self.router_url)
                return True
            if self.process is not None and self.process.poll() is not None:
                stderr = ""
                try:
                    stderr = (self.process.stderr.read() if self.process.stderr else "")[:600]
                except (OSError, ValueError) as exc:  # pragma: no cover - diagnostics only
                    logger.debug("could not read router stderr: %s", exc)
                logger.warning("router exited during startup (code=%s): %s", self.process.returncode, stderr)
                return False
            time.sleep(0.5)
        logger.warning("router did not become healthy within %.0fs", self.startup_timeout)
        return False

    def stop(self) -> None:
        """Terminate a router this process spawned.

        Only ever called on a process we started ourselves, so a router the user
        launched by hand is left running.
        """
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.debug("router did not exit in time; killing it")
            try:
                self.process.kill()
            except OSError as exc:  # pragma: no cover - best effort
                logger.debug("could not kill router: %s", exc)
