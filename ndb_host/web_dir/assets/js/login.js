const alertEl = document.getElementById("alert");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const loginBtn = document.getElementById("loginBtn");
const registerBtn = document.getElementById("registerBtn");
const serverModal = document.getElementById("serverModal");
const apiBaseInput = document.getElementById("apiBase");

function showAlert(message, type) {
  alertEl.className = "alert show alert-" + (type || "error");
  alertEl.textContent = message;
}

function hideAlert() {
  alertEl.className = "alert";
}

function setBusy(btn, busy) {
  btn.disabled = busy;
  btn.innerHTML = busy
    ? '<span class="spinner"></span>&nbsp;Please wait...'
    : btn.dataset.origLabel || btn.textContent;
}

loginBtn.dataset.origLabel = "Sign in";
registerBtn.dataset.origLabel = "Create account";

document.getElementById("registerToggle").addEventListener("click", () => {
  loginForm.style.display = "none";
  registerForm.style.display = "block";
  hideAlert();
});

document.getElementById("loginToggle").addEventListener("click", () => {
  registerForm.style.display = "none";
  loginForm.style.display = "block";
  hideAlert();
});

document.getElementById("serverLink").addEventListener("click", (e) => {
  e.preventDefault();
  apiBaseInput.value = API_BASE;
  serverModal.classList.add("show");
});

document.getElementById("serverModalClose").addEventListener("click", () => serverModal.classList.remove("show"));
document.getElementById("serverModalCancel").addEventListener("click", () => serverModal.classList.remove("show"));
document.getElementById("serverModalSave").addEventListener("click", async () => {
  const base = await applyConfigToBase(apiBaseInput.value.trim());
  setApiBase(base);
  serverModal.classList.remove("show");
});

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const remember = document.getElementById("remember").checked;

  if (!username || !password) {
    showAlert("Please enter username and password.");
    return;
  }

  hideAlert();
  setBusy(loginBtn, true);
  try {
    saveCredentials(username, password, remember);
    const resp = await AuthAPI.verify();
    const user = resp.user || {};
    if (!user.is_authenticated) {
      throw new Error(user.message || "Invalid username or password");
    }
    saveUser(user);
    window.location.href = "dashboard.html";
  } catch (err) {
    clearCredentials();
    if (err instanceof ApiError && err.status === 401) {
      showAlert("Invalid username or password.");
    } else {
      showAlert(err.message || "Unable to reach the server.");
    }
    setBusy(loginBtn, false);
  }
});

registerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("regUsername").value.trim();
  const password = document.getElementById("regPassword").value;
  const role = document.getElementById("regRole").value;

  if (username.length < 3) {
    showAlert("Username must be at least 3 characters.");
    return;
  }
  if (password.length < 6) {
    showAlert("Password must be at least 6 characters.");
    return;
  }

  hideAlert();
  setBusy(registerBtn, true);
  try {
    const resp = await AuthAPI.register(username, password, role);
    if (!resp.success) {
      throw new Error(resp.message || "Registration failed");
    }
    showAlert("Account created. You can now sign in.", "success");
    registerForm.style.display = "none";
    loginForm.style.display = "block";
    document.getElementById("username").value = username;
    document.getElementById("password").value = "";
    setBusy(registerBtn, false);
  } catch (err) {
    showAlert(err.message || "Registration failed");
    setBusy(registerBtn, false);
  }
});

(async () => {
  try {
    const user = getSavedUser();
    if (user && user.is_authenticated !== false) {
      await AuthAPI.verify();
      window.location.href = "dashboard.html";
      return;
    }
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) {
      clearCredentials();
    }
  }
})();
