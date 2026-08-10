const CONSOLE_BASE = "/api/NebulonDB/dashboard";
const CONFIG_PATH = CONSOLE_BASE + "/config";

const DEFAULT_API_BASE =
  window.location.origin || ("http://" + resolveApiHost() + ":6969");

let API_BASE = localStorage.getItem("ndb_api_base") || DEFAULT_API_BASE;

const _configReady = initApiConfig();

function resolveApiHost() {
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return window.location.hostname;
  }
  return "127.0.0.1";
}

async function initApiConfig() {
  try {
    const resp = await fetch(API_BASE + CONFIG_PATH);
    if (!resp.ok) return;
    const data = await resp.json();
    const server = data.server || {};
    let host = server.host || "";
    if (!host || host === "0.0.0.0" || host === "::") host = resolveApiHost();
    const port = server.port || new URL(API_BASE).port || "";
    let hostname = host;
    const savedBase = localStorage.getItem("ndb_api_base");
    if (savedBase) {
      try {
        hostname = new URL(savedBase).hostname || host;
      } catch (e) {}
    }
    API_BASE = "http://" + hostname + (port ? ":" + port : "");
  } catch (e) {}
}

async function applyConfigToBase(base) {
  const candidate = String(base || "").replace(/\/+$/, "");
  if (!candidate) return API_BASE;
  try {
    const resp = await fetch(candidate + CONFIG_PATH);
    if (!resp.ok) return candidate;
    const data = await resp.json();
    let hostname = "127.0.0.1";
    try {
      hostname = new URL(candidate).hostname || hostname;
    } catch (e) {}
    const port = (data.server || {}).port || "";
    return "http://" + hostname + (port ? ":" + port : "");
  } catch (e) {
    return candidate;
  }
}

const AUTH_KEY = "ndb_auth";
const USER_KEY = "ndb_user";

function setApiBase(url) {
  localStorage.setItem("ndb_api_base", url.replace(/\/+$/, ""));
}

function getAuthHeader() {
  const token = sessionStorage.getItem(AUTH_KEY) || localStorage.getItem(AUTH_KEY);
  return token ? { Authorization: "Basic " + token } : {};
}

function saveCredentials(username, password, remember) {
  const token = btoa(username + ":" + password);
  const store = remember ? localStorage : sessionStorage;
  store.setItem(AUTH_KEY, token);
  if (remember) sessionStorage.removeItem(AUTH_KEY);
  else localStorage.removeItem(AUTH_KEY);
  return token;
}

function getSavedUser() {
  try {
    return JSON.parse(sessionStorage.getItem(USER_KEY) || localStorage.getItem(USER_KEY) || "null");
  } catch (e) {
    return null;
  }
}

function saveUser(user) {
  const record = JSON.stringify(user);
  const remember = Boolean(localStorage.getItem(AUTH_KEY));
  const store = remember ? localStorage : sessionStorage;
  store.setItem(USER_KEY, record);
  if (remember) sessionStorage.removeItem(USER_KEY);
  else localStorage.removeItem(USER_KEY);
}

function clearCredentials() {
  localStorage.removeItem(AUTH_KEY);
  sessionStorage.removeItem(AUTH_KEY);
  localStorage.removeItem(USER_KEY);
  sessionStorage.removeItem(USER_KEY);
}

function isAdmin(user) {
  if (!user) return false;
  const role = String(user.role || "").toLowerCase();
  return role === "admin_user" || role === "super_user" || role === "system";
}

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

async function api(path, { method = "GET", body, params } = {}) {
  await _configReady;
  let url = API_BASE + path;
  if (params && Object.keys(params).length) {
    url += "?" + new URLSearchParams(params).toString();
  }
  const options = {
    method,
    headers: { "Content-Type": "application/json", ...getAuthHeader() },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  };
  const resp = await fetch(url, options);
  if (resp.status === 401) {
    throw new ApiError("Authentication failed. Please log in again.", 401);
  }
  const contentType = resp.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await resp.json() : await resp.text();
  if (!resp.ok) {
    throw new ApiError(data.detail || `Request failed (HTTP ${resp.status})`, resp.status, data);
  }
  return data;
}

const AuthAPI = {
  verify() {
    return api("/api/NebulonDB/auth/verify");
  },
  register(username, password, role) {
    return api("/api/NebulonDB/auth/register", {
      method: "POST",
      body: { username, password, user_role: role },
    });
  },
  changePassword(current_password, new_password) {
    return api("/api/NebulonDB/auth/change_password", {
      method: "POST",
      body: { current_password, new_password },
    });
  },
};

const CorpusAPI = {
  list() {
    return api("/api/NebulonDB/corpus/list_corpus");
  },
  create(corpus_name, ndb_type) {
    return api("/api/NebulonDB/corpus/create_corpus", {
      method: "POST",
      body: { corpus_name, ndb_type },
    });
  },
  delete(corpus_name) {
    return api("/api/NebulonDB/corpus/delete_corpus", {
      method: "POST",
      body: { corpus_name },
    });
  },
  activate(corpus_name) {
    return api("/api/NebulonDB/corpus/activate_corpus", {
      method: "POST",
      body: { corpus_name },
    });
  },
  deactivate(corpus_name) {
    return api("/api/NebulonDB/corpus/deactivate_corpus", {
      method: "POST",
      body: { corpus_name },
    });
  },
};

const SegmentAPI = {
  list(corpus_name) {
    return api("/api/NebulonDB/segment/list_segment", { params: { corpus_name } });
  },
  stats(corpus_name, segment_name) {
    return api("/api/NebulonDB/segment/segment_stats", {
      method: "POST",
      body: { corpus_name, segment_name },
    });
  },
  search(corpus_name, segment_name, search_item, top_matches) {
    return api("/api/NebulonDB/segment/search_segment", {
      method: "POST",
      body: {
        corpus_name,
        segment_name,
        search_item,
        top_matches: top_matches || 10,
        mode: "auto",
      },
    });
  },
  getRecord(corpus_name, segment_name, record_id) {
    return api("/api/NebulonDB/segment/get_record", {
      method: "POST",
      body: { corpus_name, segment_name, record_id },
    });
  },
  getData(corpus_name, segment_name, ndb_type, limit) {
    return api("/api/NebulonDB/segment/get_data", {
      method: "POST",
      body: { corpus_name, segment_name, ndb_type, limit: limit || 10 },
    });
  },
  meshVisualization(corpus_name, segment_name, ndb_type) {
    return api("/api/NebulonDB/segment/mesh_visualization", {
      method: "POST",
      body: { corpus_name, segment_name, ndb_type },
    });
  },
};

const ConfigAPI = {
  get() {
    return api(CONFIG_PATH);
  },
  update(config) {
    return api(CONFIG_PATH, {
      method: "PUT",
      body: { config },
    });
  },
};

function fmt(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value, null, 2);
    } catch (e) {
      return String(value);
    }
  }
  return String(value);
}

function fmtDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  return isNaN(d.getTime()) ? value : d.toLocaleString();
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
