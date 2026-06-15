#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║           VTES CARD SCANNER — Terminal Control           ║
║           cyberpunk unified operations TUI               ║
╚══════════════════════════════════════════════════════════╝

Requires: pip install rich
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Event

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ─── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(r"C:\Users\JackSuicide\VTES\vtes_obs_detect")
SERVER_SOURCE = Path(r"C:\Users\JackSuicide\VTES\vtes-standalone-server")
SERVER_DEPLOY = Path(r"C:\VTES\vtes-server")
OBS_LOG_DIR = Path(os.environ.get("APPDATA", r"C:\Users\JackSuicide\AppData\Roaming")) / "obs-studio" / "logs"
BUILD_SCRIPT = PROJECT_ROOT / "build.ps1"
DEPLOY_SCRIPT = PROJECT_ROOT / "deploy-windows.ps1"

# ─── Cyberpunk Palette ──────────────────────────────────────────────────────

class Palette:
    bg = "#0a0e27"
    primary = "#00f0ff"     # electric cyan
    accent = "#ff00aa"      # hot pink
    gold = "#ffd700"
    green = "#00ff88"
    red = "#ff3355"
    yellow = "#ffaa00"
    dim = "#445588"
    white = "#c8d6e5"
    surface = "#111833"
    surface2 = "#1a2255"


# ─── Subprocess Manager ─────────────────────────────────────────────────────

class ServerProcess:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.logs: list[str] = []
        self.running = False
        self._stop_event = Event()

    def start(self) -> str | None:
        if self.running:
            return "Server already running"

        exe = SERVER_DEPLOY / "vtes-server.exe"
        if not exe.exists():
            exe = SERVER_SOURCE / "build_x64" / "Release" / "vtes-server.exe"
        if not exe.exists():
            return f"vtes-server.exe not found at {exe}"

        try:
            self.process = subprocess.Popen(
                [str(exe)],
                cwd=str(exe.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.running = True
            self._stop_event.clear()
            Thread(target=self._reader, daemon=True).start()
            self.logs.append(f"[{now()}] Server started (PID: {self.process.pid})")
            return None
        except Exception as e:
            return str(e)

    def stop(self) -> str | None:
        if not self.running or not self.process:
            return "Server not running"
        self._stop_event.set()
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.running = False
        self.logs.append(f"[{now()}] Server stopped")
        return None

    def _reader(self):
        for line in self.process.stdout:
            if self._stop_event.is_set():
                break
            line = line.rstrip("\n\r")
            if line:
                self.logs.append(f"[{now()}] {line}")
                if len(self.logs) > 500:
                    self.logs = self.logs[-500:]
        self.running = False

    @property
    def recent_logs(self, n: int = 20):
        return self.logs[-n:]


class BuildManager:
    def __init__(self):
        self.busy = False
        self.last_result = ""
        self.logs: list[str] = []

    def run(self, script: Path, label: str, *args: str):
        if self.busy:
            return "Already running a build/deploy task"

        self.busy = True
        self.logs.clear()
        Thread(target=self._runner, args=(script, label, args), daemon=True).start()
        return None

    def _runner(self, script: Path, label: str, args: tuple):
        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script), *args]
        self.logs.append(f"[{now()}] {label}: starting...")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
            )
            for line in proc.stdout:
                line = line.rstrip("\n\r")
                if line:
                    self.logs.append(f"[{now()}] {line}")
                    if len(self.logs) > 200:
                        self.logs = self.logs[-200:]
            proc.wait()
            if proc.returncode == 0:
                self.logs.append(f"[{now()}] {label}: SUCCESS")
                self.last_result = "success"
            else:
                self.logs.append(f"[{now()}] {label}: FAILED (exit code {proc.returncode})")
                self.last_result = "failed"
        except Exception as e:
            self.logs.append(f"[{now()}] {label}: ERROR — {e}")
            self.last_result = "error"
        self.busy = False

    @property
    def recent_logs(self, n: int = 15):
        return self.logs[-n:]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─── OBS Helpers ────────────────────────────────────────────────────────────

def obs_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq obs64.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return "obs64.exe" in result.stdout
    except Exception:
        return False


def latest_obs_log() -> str:
    if not OBS_LOG_DIR.exists():
        return "(no log directory)"
    logs = sorted(OBS_LOG_DIR.glob("*.txt"), key=os.path.getmtime, reverse=True)
    if not logs:
        return "(no logs found)"
    try:
        with open(logs[0], "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # Return last 30 lines
        return "".join(lines[-30:])
    except Exception as e:
        return f"(error reading log: {e})"


# ─── TUI Rendering ──────────────────────────────────────────────────────────

def make_status_bar(server: ServerProcess, build: BuildManager) -> Panel:
    obs = "● RUNNING" if obs_running() else "○ STOPPED"
    obs_style = "bold green" if obs_running() else "bold red"

    if server.running:
        srv = f"● RUNNING  (PID: {server.process.pid if server.process else '?'})"
        srv_style = "bold green"
    else:
        srv = "○ STOPPED"
        srv_style = "bold red"

    if build.busy:
        bld = Spinner("dots", text=" WORKING...", style="bold yellow")
    elif build.last_result == "success":
        bld = Text("● READY", style="bold green")
    elif build.last_result == "failed":
        bld = Text("● FAILED", style="bold red")
    else:
        bld = Text("○ IDLE", style="bold dim")

    grid = Table.grid(padding=(0, 4))
    grid.add_column(style="bold", justify="right")
    grid.add_column()
    grid.add_row("🖥  OBS", Text(obs, style=obs_style))
    grid.add_row("⚡ Server", Text(srv, style=srv_style))
    grid.add_row("🔧 Build", bld)
    grid.add_row(
        "📁 Project",
        Text(f"{PROJECT_ROOT.name}\\  [{SERVER_DEPLOY.name}]", style="dim"),
    )

    return Panel(
        grid,
        title="  ◆  STATUS  ",
        border_style=Palette.primary,
        padding=(1, 2),
    )


def make_action_panel() -> Panel:
    actions = Table.grid(padding=(0, 3))
    actions.add_column(justify="right", style=f"bold {Palette.accent}")
    actions.add_column(style=Palette.white)
    actions.add_column(justify="right", style=f"bold {Palette.accent}")
    actions.add_column(style=Palette.white)

    actions.add_row(" [1] ", "Build Plugin     ", " [2] ", "Deploy to OBS")
    actions.add_row(" [3] ", "Restart OBS      ", " [4] ", "Toggle Server")
    actions.add_row(" [5] ", "Server Logs      ", " [6] ", "OBS Logs")
    actions.add_row(" [7] ", "Build + Deploy   ", " [Q] ", "Quit")

    return Panel(
        Align.center(actions),
        title="  ◆  ACTIONS  ",
        border_style=Palette.accent,
        padding=(1, 1),
    )


def make_log_panel(logs: list[str], title: str = "LOG", border_style: str | None = None) -> Panel:
    if not logs:
        content = Text("(no output yet)", style="dim")
    else:
        lines = []
        for line in logs:
            # Colorize log lines
            ts_match = re.match(r"^\[(\d{2}:\d{2}:\d{2})\](.*)", line)
            if ts_match:
                ts = Text(f"[{ts_match.group(1)}]", style=f"bold {Palette.dim}")
                rest = ts_match.group(2)
                styled = Text()
                if "error" in rest.lower() or "fail" in rest.lower():
                    styled.append(rest, style=f"bold {Palette.red}")
                elif "success" in rest.lower() or "ok" in rest.lower():
                    styled.append(rest, style=f"bold {Palette.green}")
                elif "card" in rest.lower() or "identif" in rest.lower():
                    styled.append(rest, style=Palette.gold)
                elif "warn" in rest.lower():
                    styled.append(rest, style=Palette.yellow)
                else:
                    styled.append(rest, style=Palette.white)
                lines.append(Text.assemble(ts, styled))
            else:
                lines.append(Text(line, style=Palette.white))
        content = Group(*lines)

    return Panel(
        content,
        title=f"  ◆  {title}  ",
        border_style=border_style or Palette.primary,
        padding=(1, 2),
        height=18,
    )


def make_header() -> Panel:
    title = Text()
    title.append("⚡ VTES CARD SCANNER", style=f"bold {Palette.accent}")
    title.append("  ──  ", style=f"bold {Palette.dim}")
    title.append("Terminal Control", style=f"bold {Palette.primary}")

    subtitle = Text()
    subtitle.append(f"v1.0  |  {PROJECT_ROOT}", style=f"dim {Palette.white}")

    inner = Group(
        Align.center(title),
        Align.center(subtitle),
    )
    return Panel(
        inner,
        border_style=Palette.accent,
        padding=(1, 0),
    )


def make_footer(msg: str = "") -> Panel:
    text = Text(f"  {msg or 'PRESS [1-7] OR [Q]  •  Ctrl+C to quit'}", style=f"bold {Palette.dim}")
    return Panel(
        Align.center(text),
        border_style=Palette.surface2,
        padding=(0, 0),
    )


# ─── Main Loop ──────────────────────────────────────────────────────────────

async def main():
    console = Console()
    server = ServerProcess()
    build = BuildManager()

    footer_msg = ""
    active_panel = "server"  # 'server' or 'obs'
    server_log_scroll = 20
    build_log_scroll = 15

    def build_layout():
        status = make_status_bar(server, build)
        actions = make_action_panel()

        if active_panel == "server":
            log_title = "SERVER LOGS"
            logs = server.recent_logs(server_log_scroll)
            bstyle = Palette.primary
        else:
            log_title = "OBS LOGS"
            logs_text = latest_obs_log()
            logs = logs_text.split("\n") if logs_text else ["(no log)"]
            bstyle = Palette.gold

        log_panel = make_log_panel(logs, title=log_title, border_style=bstyle)

        # Build/deploy logs in a secondary small panel
        bl = build.recent_logs(build_log_scroll)
        build_panel = make_log_panel(bl, title="BUILD LOG", border_style=Palette.green)

        top_row = Columns([status, actions], equal=True, expand=True)
        main_group = Group(top_row, log_panel, build_panel)

        return Panel(
            main_group,
            border_style=Palette.surface,
            padding=(0, 0),
        )

    # ── Keyboard input reader ───────────────────────────────────────────
    def read_keys():
        nonlocal footer_msg, active_panel, server_log_scroll, build_log_scroll
        try:
            import msvcrt
            while True:
                ch = msvcrt.getch().decode("ascii", errors="replace").lower()
                if ch == "1":
                    footer_msg = "Building plugin..."
                    build.run(BUILD_SCRIPT, "Build")
                elif ch == "2":
                    footer_msg = "Deploying to OBS..."
                    build.run(DEPLOY_SCRIPT, "Deploy")
                elif ch == "3":
                    footer_msg = "Restarting OBS..."
                    os.system("taskkill /f /im obs64.exe 2>nul & timeout /t 2 >nul & start obs64.exe")
                    footer_msg = "OBS restarted"
                elif ch == "4":
                    if server.running:
                        err = server.stop()
                        footer_msg = err or "Server stopped"
                    else:
                        err = server.start()
                        footer_msg = err or "Server started"
                elif ch == "5":
                    active_panel = "server" if active_panel == "obs" else "obs"
                    footer_msg = f"Showing {active_panel.upper()} logs"
                elif ch == "6":
                    footer_msg = latest_obs_log()[:80]
                elif ch == "7":
                    footer_msg = "Building + Deploying..."
                    build.run(BUILD_SCRIPT, "Build")
                    time.sleep(0.5)
                    while build.busy:
                        time.sleep(0.5)
                    build.run(DEPLOY_SCRIPT, "Deploy")
                elif ch == "q":
                    os._exit(0)
        except Exception:
            pass

    Thread(target=read_keys, daemon=True).start()

    # ── Live render loop ────────────────────────────────────────────────
    try:
        with Live(build_layout(), console=console, refresh_per_second=8, screen=True) as live:
            while True:
                live.update(build_layout())
                await asyncio.sleep(0.125)
    except KeyboardInterrupt:
        pass
    finally:
        if server.running:
            server.stop()
        console.print("[bold red]TUI exited[/bold red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye.")
