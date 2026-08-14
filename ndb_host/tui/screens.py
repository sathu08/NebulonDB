# ndb_host/tui/screens.py
from textual.screen import Screen
from textual.binding import Binding
from textual.app import ComposeResult

from textual.containers import Horizontal, Container
from textual.widgets import Header, Footer, Button, Static, Input, Select

from .context import cfg, logger
from .server_ops import create_user

# ==========================================================
# TUI SCREENS
# ==========================================================

class CreateUserScreen(Screen):

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
    ]

    def compose(self) -> ComposeResult:

        yield Header()

        with Container(id="create-user-container"):
            yield Static("Create NebulonDB User", id="create-user-title")
            yield Input(placeholder="Username", id="username")
            yield Input(placeholder="Password", password=True, id="password")
            yield Input(placeholder="Confirm Password", password=True, id="confirm-password")
            yield Select(
                [("user", "user"), ("admin_user", "admin_user"), ("super_user", "super_user")],
                value="user",
                id="role",
            )

            with Horizontal(id="create-user-buttons"):
                yield Button("Create User", variant="success", id="create")
                yield Button("Cancel", variant="error", id="cancel")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):

        if event.button.id == "cancel":
            self.app.pop_screen()
            return

        if event.button.id == "create":
            username = self.query_one("#username", Input).value.strip()
            password = self.query_one("#password", Input).value
            confirm = self.query_one("#confirm-password", Input).value
            role = self.query_one("#role", Select).value

            if not username:
                self.app.notify("Username cannot be empty.", severity="error")
                return

            if len(password) < 8:
                self.app.notify("Password must be at least 8 characters.", severity="error")
                return

            if password != confirm:
                self.app.notify("Passwords do not match.", severity="error")
                return

            self.app.run_worker(self._create_user(username, password, role), exclusive=True)

    async def _create_user(self, username, password, role):

        self.app.notify("Creating user...", severity="information")

        try:
            success = create_user(cfg, username, password, role)
            if success:
                self.app.notify(f"User '{username}' created successfully.", severity="success")
                self.app.pop_screen()
            else:
                self.app.notify("User creation failed.", severity="error")
        except Exception as exc:
            logger.exception("User creation failed")
            self.app.notify(f"Error: {exc}", severity="error")

    def action_cancel(self):
        self.app.pop_screen()