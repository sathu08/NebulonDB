# ndb_host/tui/app.py
import sys

import pyfiglet

from textual.binding import Binding
from textual.reactive import reactive
from textual.app import App, ComposeResult

from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Static

from textual import work

from .processes import _is_server_running
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
        width: 28%;
        height: 100%;
        border: solid #00bcd4;
        padding: 2;
    }

    #content {
        width: 72%;
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
        height: 10;
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
    ]

    server_status = reactive(False)

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
                yield Button("♙  Create User", id="create-user", classes="menu-button", disabled=is_initialized(cfg))
                yield Button("●  Refresh Status", id="status-button", classes="menu-button")
                yield Button("×  Exit", id="exit", classes="menu-button", variant="error")

            # --------------------------------------------------
            # Main content
            # --------------------------------------------------

            with Vertical(id="content"):
                yield Static("NebulonDB Dashboard", id="dashboard-title")
                yield Static("Checking server status...", id="status")

        yield Footer()

    def on_mount(self):
        self.update_status()
        # First button receives focus
        self.query_one("#start", Button).focus()

    # ======================================================
    # Button Events
    # ======================================================

    def on_button_pressed(self, event: Button.Pressed):
        button_id = event.button.id
        if button_id == "start":
            self.start_server_action()
        elif button_id == "stop":
            self.stop_server_action()
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
    # Start
    # ======================================================

    @work(thread=True)
    def start_server_action(self):
        self.notify("Starting NebulonDB...", severity="information")
        try:
            success = start_server(cfg, foreground=False)
            if success:
                self.notify("NebulonDB started successfully.", severity="success")
            else:
                self.notify("NebulonDB was not started.", severity="warning")
        except Exception as exc:
            logger.exception("Failed to start NebulonDB")
            self.notify(f"Start failed: {exc}", severity="error")

        self.call_after_refresh(self.update_status)

    # ======================================================
    # Stop
    # ======================================================

    @work(thread=True)
    def stop_server_action(self):
        self.notify("Stopping NebulonDB...", severity="information")
        try:
            success = stop_server(cfg, force=False)
            if success:
                self.notify("NebulonDB stopped successfully.", severity="success")
            else:
                self.notify("NebulonDB could not be stopped.", severity="warning")
        except Exception as exc:
            logger.exception("Failed to stop NebulonDB")
            self.notify(f"Stop failed: {exc}", severity="error")

        self.call_after_refresh(self.update_status)

    # ======================================================
    # Status
    # ======================================================

    def update_status(self):
        try:
            running = _is_server_running(cfg.HOST, cfg.PORT)
        except Exception:
            running = False
        self.server_status = running
        status_widget = self.query_one("#status", Static)
        if running:
            status_widget.update(
                f"""
                    [bold green]● SERVER RUNNING[/bold green]
                    Host       : {cfg.HOST}
                    Port       : {cfg.PORT}
                    Workers    : {cfg.WORKERS}

                    PID File   : {
                        "Present"
                        if NEBULONDB_PID_FILE.exists()
                        else "Not Found"
                    }
                    Dashboard  : http://{cfg.HOST}:{cfg.PORT}/api/NebulonDB/dashboard/
                    """
                                )
        else:
            status_widget.update(
                f"""
[bold red]● SERVER STOPPED[/bold red]

Host       : {cfg.HOST}
Port       : {cfg.PORT}

PID File   : {
    "Present"
    if NEBULONDB_PID_FILE.exists()
    else "Not Found"
}
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