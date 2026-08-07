let API_BASE = localStorage.getItem("ndb_api_base") || "http://127.0.0.1:8000";

const _configReady = initApiConfig();

function resolveApiHost() {
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return window.location.hostname;
  }
  return "127.0.0.1";
}

async function initApiConfig() {
  try {
    const resp = await fetch(API_BASE + "/api/NebulonDB/system/config");
    if (!resp.ok) return;
    const data = await resp.json();
    let host = data.host || "";
    if (!host || host === "0.0.0.0" || host === "::") host = resolveApiHost();
    const port = data.port || 8000;
    let hostname = host;
    const savedBase = localStorage.getItem("ndb_api_base");
    if (savedBase) {
      try {
        hostname = new URL(savedBase).hostname || host;
      } catch (e) {}
    }
    API_BASE = "http://" + hostname + ":" + port;
  } catch (e) {}
}

async function applyConfigToBase(base) {
  const candidate = String(base || "").replace(/\/+$/, "");
  if (!candidate) return API_BASE;
  try {
    const resp = await fetch(candidate + "/api/NebulonDB/system/config");
    if (!resp.ok) return candidate;
    const data = await resp.json();
    let hostname = "127.0.0.1";
    try {
      hostname = new URL(candidate).hostname || hostname;
    } catch (e) {}
    return "http://" + hostname + ":" + (data.port || 8000);
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
  return user && String(user.role || "").toLowerCase().includes("admin");
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
