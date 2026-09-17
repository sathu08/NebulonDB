# ndb_host/tui/app.py
import sys
import time

import pyfiglet

from textual.binding import Binding
from textual.reactive import reactive
from textual.app import App, ComposeResult

from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Static

from textual import work

from .processes import _is_server_running, is_server_starting_up, _probe_host
from .context import cfg, logger, NEBULONDB_PID_FILE, enable_tui_mode, setup_nebulondb_paths
from .server_ops import start_server, stop_server, restart_server, create_user, is_initialized

from .screens import CreateUserScreen


NEBULONDB_BANNER = pyfiglet.figlet_format("NEBULONDB", font="smslant")

class NebulonDBApp(App):

    TITLE = "NebulonDB"

    CSS = """

    Screen {
        background: #0b0f14;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #sidebar {
        width: 38%;
        height: 100%;
        border: solid #00bcd4;
        padding: 2;
    }

    #content {
        width: 62%;
        height: 100%;
        border: solid #263238;
        padding: 3;
    }

    #logo {
        height: 7;
        content-align: center middle;
        text-style: bold;
        color: #00e5ff;
    }

    #menu-title {
        height: 3;
        content-align: left middle;
        text-style: bold;
    }

    .menu-button {
        width: 100%;
        margin: 1 0;
    }

    #status {
        height: 14;
        border: solid #263238;
        padding: 2;
    }

    #dashboard-title {
        height: 5;
        content-align: center middle;
        text-style: bold;
        color: #00e5ff;
    }

    #create-user-container {
        width: 70%;
        height: auto;
        margin: 4 15;
        padding: 3;
        border: solid #00bcd4;
    }

    #create-user-title {
        height: 5;
        content-align: center middle;
        text-style: bold;
        color: #00e5ff;
    }

    #create-user-buttons {
        height: 5;
        align: center middle;
    }

    Input {
        margin: 1 0;
    }

    Select {
        margin: 1 0;
    }

    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("up", "focus_previous", "Up"),
        Binding("down", "focus_next", "Down"),
        Binding("left", "focus_previous", "Left"),
        Binding("right", "focus_next", "Right"),
    ]

    server_status = reactive(False)

    def action_focus_next(self):
        self.screen.focus_next()

    def action_focus_previous(self):
        self.screen.focus_previous()

    def on_mount(self):
        self._refresh_sidebar()
        self.update_status()
        self._start_status_timer()

    def _start_status_timer(self):
        """Poll the server status every few seconds so the dashboard
        auto-updates without pressing the refresh button."""
        self.set_interval(3, self.update_status)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():

            # --------------------------------------------------
            # Sidebar
            # --------------------------------------------------

            with Vertical(id="sidebar"):
                yield Static("✦\n" + NEBULONDB_BANNER, id="logo")
                yield Static("Server", id="menu-title")
                yield Button("▶  Start Server", id="start", classes="menu-button")
                yield Button("■  Stop Server", id="stop", classes="menu-button")
                yield Button("↻  Restart Server", id="restart", classes="menu-button")
                yield Button("♙  Create User", id="create-user", classes="menu-button")
                yield Button("●  Refresh Status", id="status-button", classes="menu-button")
                yield Button("×  Exit", id="exit", classes="menu-button", variant="error")

            # --------------------------------------------------
            # Main content
            # --------------------------------------------------

            with Vertical(id="content"):
                yield Static("NebulonDB Dashboard", id="dashboard-title")
                yield Static("Checking server status...", id="status")

        yield Footer()

    # ======================================================
    # Sidebar visibility
    # ======================================================

    def _refresh_sidebar(self):
        """Show Start/Stop only once the system is initialized (corpora
        present); show Create User only on a fresh, uninitialized install."""
        initialized = is_initialized(cfg)
        for button in self.query(Button):
            if button.id in ("start", "stop", "restart", "create-user"):
                button.display = (button.id == "create-user") == (not initialized)
        for button_id in ("start", "stop", "restart", "create-user"):
            try:
                button = self.query_one(f"#{button_id}", Button)
            except Exception:
                continue
            if button.display:
                button.focus()
                break

    # ======================================================
    # Button Events
    # ======================================================

    def on_button_pressed(self, event: Button.Pressed):
        button_id = event.button.id
        if button_id == "start":
            self.start_server_action()
        elif button_id == "stop":
            self.stop_server_action()
        elif button_id == "restart":
            self.restart_server_action()
        elif button_id == "create-user":
            if is_initialized(cfg):
                self.notify(
                    "System already initialized. Use the running server to manage users.",
                    severity="warning",
                )
            else:
                self.push_screen(CreateUserScreen())
        elif button_id == "status-button":
            self.update_status()
        elif button_id == "exit":
            self.exit()

    # ======================================================
    # Server Actions (Start / Stop)
    # ======================================================

    def _server_action(self, label, run, success_msg, failure_msg, wait_running):
        """Shared worker body for start/stop: notify, run, then poll the
        port until it reflects the desired state."""
        self.notify(f"{label}ing NebulonDB...", severity="information")
        try:
            ok = run()
            if ok:
                self.notify(success_msg, severity="success")
            else:
                self.notify(failure_msg, severity="warning")
        except Exception as exc:
            logger.exception(f"Failed to {label.lower()} NebulonDB")
            self.notify(f"{label} failed: {exc}", severity="error")
            return

        max_attempts = 12 if wait_running else 6
        for _ in range(max_attempts):
            time.sleep(1)
            self.call_from_thread(self.update_status)
            if _is_server_running(cfg.HOST, cfg.PORT) == wait_running:
                break

    @work(thread=True)
    def start_server_action(self):
        self._server_action(
            "Start",
            lambda: start_server(cfg, foreground=False),
            "NebulonDB started successfully.",
            "NebulonDB was not started.",
            wait_running=True,
        )

    @work(thread=True)
    def restart_server_action(self):
        self._server_action(
            "Restart",
            lambda: restart_server(cfg, foreground=False, force=False),
            "NebulonDB restarted successfully.",
            "NebulonDB was not restarted.",
            wait_running=True,
        )

    @work(thread=True)
    def stop_server_action(self):
        self._server_action(
            "Stop",
            lambda: stop_server(cfg, force=False),
            "NebulonDB stopped successfully.",
            "NebulonDB could not be stopped.",
            wait_running=False,
        )

    # ======================================================
    # Status
    # ======================================================

    def update_status(self):
        try:
            running = _is_server_running(cfg.HOST, cfg.PORT)
        except Exception:
            running = False
        starting_up = not running and is_server_starting_up(NEBULONDB_PID_FILE)
        self.server_status = running
        status_widget = self.query_one("#status", Static)
        web_host = _probe_host(cfg.HOST)
        pid_label = "Present" if NEBULONDB_PID_FILE.exists() else "Not Found"
        if running:
            status_widget.update(
                f"""
                [bold green]● SERVER RUNNING[/bold green]

                Host       : {cfg.HOST}
                Port       : {cfg.PORT}
                Workers    : {cfg.WORKERS}

                Web UI     : [bold cyan]http://{web_host}:{cfg.PORT}/api/NebulonDB/dashboard/[/bold cyan]
                PID File   : {pid_label}
                """
                            )
        elif starting_up:
            status_widget.update(
                f"""
                [bold yellow]● SERVER STARTING...[/bold yellow]

                NebulonDB is booting up. Please wait.

                Host       : {cfg.HOST}
                Port       : {cfg.PORT}
                Workers    : {cfg.WORKERS}

                PID File   : Present
                """
                            )
        else:
            status_widget.update(
                f"""
                [bold red]● SERVER STOPPED[/bold red]

                Host       : {cfg.HOST}
                Port       : {cfg.PORT}

                PID File   : {pid_label}
                """
            )


# ==========================================================
#         Main Entry Point (CLI + TUI dispatcher)
# ==========================================================

def main():
    if len(sys.argv) < 2:
        # No command -> launch the interactive TUI.
        enable_tui_mode()
        setup_nebulondb_paths()
        app = NebulonDBApp()
        app.run()
        return

    command = sys.argv[1].lower()

    if command in ("--help", "-h", "help"):
        print("Usage: nebulondb {start|stop|restart|--create-user} [--foreground|-f] [--force|-F]")
        print("       nebulondb                          # launch the interactive management TUI")
        sys.exit(0)

    setup_nebulondb_paths()

    # Check for foreground flag
    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    # Check for force flag
    force = "--force" in sys.argv or "-F" in sys.argv

    if command == "start":
        start_server(cfg, foreground=foreground)
    elif command == "stop":
        stop_server(cfg, force=force)
    elif command == "restart":
        restart_server(cfg, foreground=foreground, force=force)
    elif command == "--create-user":
        create_user(cfg)
    else:
        logger.error("Invalid command. Usage: nebulondb {start|stop|restart|--create-user} [--foreground|-f] [--force|-F]")
        sys.exit(1)