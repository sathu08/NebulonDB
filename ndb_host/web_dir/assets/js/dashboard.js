const state = {
  user: null,
  corpora: [],
  selectedCorpus: null,
  selectedSegment: null,
  statusMap: {},
  segmentCounts: {},
};

const els = {
  corpusList: document.getElementById("corpusList"),
  corpusSearch: document.getElementById("corpusSearch"),
  content: document.getElementById("content"),
  emptyState: document.getElementById("emptyState"),
  corpusView: document.getElementById("corpusView"),
  corpusTitle: document.getElementById("corpusTitle"),
  corpusBadge: document.getElementById("corpusBadge"),
  corpusMeta: document.getElementById("corpusMeta"),
  corpusActions: document.getElementById("corpusActions"),
  segmentTableBody: document.getElementById("segmentTableBody"),
  segmentEmpty: document.getElementById("segmentEmpty"),
  segmentView: document.getElementById("segmentView"),
  segmentName: document.getElementById("segmentName"),
  segmentStatsNote: document.getElementById("segmentStatsNote"),
  loadSegmentBtn: document.getElementById("loadSegmentBtn"),
  statsGrid: document.getElementById("statsGrid"),
  topbarTitle: document.getElementById("topbarTitle"),
  userName: document.getElementById("userName"),
  userRole: document.getElementById("userRole"),
  userAvatar: document.getElementById("userAvatar"),
  modalOverlay: document.getElementById("modalOverlay"),
  modalTitle: document.getElementById("modalTitle"),
  modalBody: document.getElementById("modalBody"),
  modalFoot: document.getElementById("modalFoot"),
};

const TOAST_ICONS = { error: "✕", success: "✓", warning: "⚠", info: "ℹ" };

function showToast(message, type, target) {
  const container = document.getElementById("toastContainer");
  if (container && message != null && message !== "") {
    const kind = type === "error" ? "error" : type === "success" ? "success" : type === "warning" ? "warning" : "info";
    const toast = document.createElement("div");
    toast.className = "toast toast-" + kind;
    toast.innerHTML = `<span class="toast-icon">${TOAST_ICONS[kind]}</span><span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
      toast.classList.add("hide");
      setTimeout(() => toast.remove(), 250);
    }, 3500);
    return;
  }
  const el = target || document.getElementById("dataMessage");
  if (!el) return;
  el.textContent = message;
  el.className = "mt-16 small " + (type === "error" ? "alert-error" : type === "success" ? "alert-success" : "muted");
}

function logout() {
  clearCredentials();
  window.location.href = CONSOLE_BASE + "/";
}

document.getElementById("logoutBtn").addEventListener("click", logout);

async function loadUser() {
  try {
    const resp = await AuthAPI.verify();
    state.user = resp.user || {};
    if (!state.user.is_authenticated) {
      throw new Error("Not authenticated");
    }
    saveUser(state.user);
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      logout();
      return;
    }
    state.user = getSavedUser();
  }
  const name = state.user.username || "user";
  els.userName.textContent = name;
  els.userRole.textContent = state.user.role || "user";
  els.userAvatar.textContent = name.charAt(0).toUpperCase();
  applyAccess();
}

function setBtnAccess(id, enabled, tip) {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.disabled = !enabled;
  btn.classList.toggle("disabled", !enabled);
  btn.title = enabled ? "" : (tip || "Access restricted");
}

function applyAccess() {
  const admin = isAdmin(state.user);

  setBtnAccess("newCorpusBtn", admin, "Admin access required");
  setBtnAccess("loadSegmentBtn", false, "Not available in this version. Load This manually");
  setBtnAccess("getDataBtn", true, "");

  renderCorpusActions();
}

function statusBadge(status) {
  const label = status || "unknown";
  const cls = {
    active: "badge-active",
    deactivate: "badge-deactivate",
    system: "badge-system",
  }[label] || "badge-default";
  return `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
}

async function loadCorpora() {
  els.corpusList.innerHTML = '<div class="muted small" style="padding:12px;">Loading corpora...</div>';
  try {
    const resp = await CorpusAPI.list();
    if (!resp.success) throw new Error(resp.message || "Failed to load corpora");
    const data = resp.data || {};
    state.corpora = (data.corpus_list || []).map((c) =>
      typeof c === "string"
        ? { name: c }
        : { name: c.name || "", type: c.ndb_type, status: c.status, created_at: c.created_at }
    );
    state.statusMap = {};
    state.corpora.forEach((c) => {
      if (c.status) state.statusMap[c.name] = c.status;
    });
    const counts = await Promise.allSettled(
      state.corpora.map(async (c) => {
        const r = await SegmentAPI.list(c.name);
        return { name: c.name, segments: (r.data || {}).segment_list || [] };
      })
    );
    counts.forEach((res, i) => {
      if (res.status === "fulfilled") {
        state.segmentCounts[state.corpora[i].name] = res.value.segments.length;
      }
    });
    renderCorpusList();
    if (state.corpora.length === 0) {
      els.corpusList.innerHTML = '<div class="muted small" style="padding:12px;">No corpora found.</div>';
    }
  } catch (err) {
    els.corpusList.innerHTML = `<div class="alert alert-error show small">${escapeHtml(err.message || "Error")}</div>`;
  }
}

function renderCorpusList() {
  const q = els.corpusSearch.value.trim().toLowerCase();
  els.corpusList.innerHTML = "";
  state.corpora
    .filter((c) => !q || c.name.toLowerCase().includes(q))
    .forEach((c) => {
      const count = state.segmentCounts[c.name] ?? 0;
      const selected = state.selectedCorpus === c.name;
      const card = document.createElement("div");
      card.className = "corpus-card" + (selected ? " selected" : "");
      card.innerHTML = `
        <div class="corpus-card-top">
          <span class="corpus-name">${escapeHtml(c.name)}</span>
        </div>
        <div class="corpus-meta">
          ${statusBadge(state.statusMap[c.name])}
          <span class="badge badge-default">${escapeHtml(c.type || "orbit")}</span>
          <span>📄 ${count} segment${count === 1 ? "" : "s"}</span>
        </div>
      `;
      card.addEventListener("click", () => selectCorpus(c.name));
      els.corpusList.appendChild(card);
    });
}

function selectCorpus(name) {
  state.selectedCorpus = name;
  state.selectedSegment = null;
  renderCorpusList();
  loadSegments(name);
  els.emptyState.style.display = "none";
  els.corpusView.style.display = "block";
  els.segmentView.style.display = "none";
  els.topbarTitle.textContent = name;
  els.corpusTitle.textContent = name;
  const corpus = state.corpora.find((c) => c.name === name) || {};
  els.corpusBadge.innerHTML = statusBadge(state.statusMap[name]);
  els.corpusMeta.innerHTML = `<span class="badge badge-default">Type: ${escapeHtml(corpus.type || "orbit")}</span> • ${state.segmentCounts[name] ?? 0} segments`;
  renderCorpusActions();
}

function renderCorpusActions() {
  els.corpusActions.innerHTML = "";
  const name = state.selectedCorpus;
  if (!name) return;
  const status = state.statusMap[name] || "active";
  const isSystem = status === "system";
  const isDeactivated = status === "deactivate";

  const actBtn = document.createElement("button");
  actBtn.className = "btn btn-sm btn-warning";
  actBtn.textContent = isDeactivated ? "Activate" : "Deactivate";
  if (isSystem) {
    actBtn.classList.add("disabled");
    actBtn.disabled = true;
    actBtn.title = "System corpora cannot be activated or deactivated";
  }
  actBtn.addEventListener("click", async () => {
    if (isSystem) return;
    try {
      const resp = isDeactivated
        ? await CorpusAPI.activate(name)
        : await CorpusAPI.deactivate(name);
      if (!resp.success) throw new Error(resp.message || "Action failed");
      state.statusMap[name] = isDeactivated ? "active" : "deactivate";
      showToast(resp.message || (isDeactivated ? "Corpus activated." : "Corpus deactivated."), "success");
      renderCorpusList();
      els.corpusBadge.innerHTML = statusBadge(state.statusMap[name]);
      renderCorpusActions();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  const delBtn = document.createElement("button");
  delBtn.className = "btn btn-sm btn-danger";
  delBtn.textContent = "Delete";
  if (isSystem) {
    delBtn.classList.add("disabled");
    delBtn.disabled = true;
    delBtn.title = "System corpora cannot be deleted";
  }
  delBtn.addEventListener("click", async () => {
    if (isSystem) return;
    if (!confirm(`Delete corpus "${name}"? This cannot be undone.`)) return;
    try {
      const resp = await CorpusAPI.delete(name);
      if (!resp.success) throw new Error(resp.message || "Delete failed");
      state.corpora = state.corpora.filter((c) => c.name !== name);
      state.selectedCorpus = null;
      state.selectedSegment = null;
      delete state.segmentCounts[name];
      renderCorpusList();
      els.corpusView.style.display = "none";
      els.emptyState.style.display = "flex";
      els.topbarTitle.textContent = "Overview";
      showToast(resp.message, "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  const admin = isAdmin(state.user);
  if (!admin) {
    actBtn.classList.add("disabled");
    actBtn.disabled = true;
    actBtn.title = "Admin access required";
    delBtn.classList.add("disabled");
    delBtn.disabled = true;
    delBtn.title = "Admin access required";
  }
  els.corpusActions.append(actBtn, delBtn);
}

async function loadSegments(corpusName) {
  els.segmentTableBody.innerHTML = '<tr><td colspan="4" class="muted">Loading segments...</td></tr>';
  try {
    const resp = await SegmentAPI.list(corpusName);
    if (!resp.success) throw new Error(resp.message || "Failed to load segments");
    const segments = (resp.data || {}).segment_list || [];
    state.segmentCounts[corpusName] = segments.length;
    renderCorpusList();

    if (!segments.length) {
      els.segmentTableBody.innerHTML = '<tr><td colspan="4" class="muted">No segments in this corpus.</td></tr>';
    } else {
      const corpus = state.corpora.find((c) => c.name === corpusName) || {};
      const isOrbit = (corpus.type || "").toLowerCase() === "orbit";
      const meshAttr = isOrbit
        ? 'title="View mesh graph visualization"'
        : 'disabled title="Mesh visualization is only available for Orbit corpora"';
      els.segmentTableBody.innerHTML = segments
        .map(
          (s) => `
          <tr class="row-click" data-segment="${escapeHtml(s.name || "")}">
            <td>
              <div class="segment-cell-name">
                <span class="segment-icon">📄</span>
                <span>${escapeHtml(s.name || "—")}</span>
              </div>
            </td>
            <td>${fmt(s.inserted)}</td>
            <td class="muted">${fmtDate(s.created_at)}</td>
            <td>
              <div class="flex" style="gap:8px;">
                <button class="btn btn-ghost btn-sm view-btn" data-segment="${escapeHtml(s.name || "")}">View data</button>
                <button class="btn btn-ghost btn-sm mesh-btn" data-segment="${escapeHtml(s.name || "")}" ${meshAttr}>View mesh visual</button>
              </div>
            </td>
          </tr>`
        )
        .join("");
      els.segmentTableBody.querySelectorAll("[data-segment]").forEach((el) => {
        el.addEventListener("click", (e) => {
          if (e.target.closest(".mesh-btn")) {
            const btn = e.target.closest(".mesh-btn");
            openMeshVisual(btn.dataset.segment, btn);
          } else if (e.target.closest(".view-btn")) {
            selectSegment(e.target.dataset.segment);
          } else {
            const seg = el.dataset.segment;
            if (seg) selectSegment(seg);
          }
        });
      });
    }
  } catch (err) {
    els.segmentTableBody.innerHTML = `<tr><td colspan="4" class="alert alert-error show">${escapeHtml(err.message || "Error")}</td></tr>`;
  }
}

async function selectSegment(segmentName) {
  state.selectedSegment = segmentName;
  els.segmentView.style.display = "block";
  els.segmentName.textContent = segmentName;
  els.loadSegmentBtn.style.display = "block";
  els.statsGrid.innerHTML = '<div class="muted small">Loading stats...</div>';
  els.segmentStatsNote.textContent = "";
  loadStats(state.selectedCorpus, segmentName);
}

async function loadStats(corpusName, segmentName) {
  try {
    const resp = await SegmentAPI.stats(corpusName, segmentName);
    if (!resp.success) throw new Error(resp.message || "Failed to load stats");
    const stats = resp.data || {};
    const items = [
      { label: "Vectors", value: stats.vector_count },
      { label: "Nodes", value: stats.node_count },
      { label: "Edges", value: stats.edge_count },
      { label: "Dimension", value: stats.dimension },
      { label: "Space", value: stats.space },
      { label: "Deleted", value: stats.deleted_ratio },
    ];
    els.statsGrid.innerHTML = items
      .map((it) => `
        <div class="stat-card">
          <div class="stat-label">${it.label}</div>
          <div class="stat-value">${it.value === null || it.value === undefined ? "—" : escapeHtml(String(it.value))}</div>
        </div>`)
      .join("");
    els.segmentStatsNote.textContent = "Last sequence: " + (stats.lsn ?? "—");
  } catch (err) {
    els.statsGrid.innerHTML = `<div class="alert alert-error show">${escapeHtml(err.message)}</div>`;
  }
}

function renderSearchMeta(meta) {
  if (!meta || typeof meta !== "object") return "—";
  const ranks = ["text", "content", "title", "doc", "label"];
  const keys = Object.keys(meta).sort((a, b) => {
    const ia = ranks.indexOf(a);
    const ib = ranks.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
  if (!keys.length) return "—";

  let html = '<div class="meta-list">';
  keys.forEach((k) => {
    let val = fmt(meta[k]);
    if (typeof val === "string" && val.length > 200) val = val.slice(0, 200) + "…";
    html += `<div class="meta-row"><span class="meta-key">${escapeHtml(k)}</span><span class="meta-value">${escapeHtml(val)}</span></div>`;
  });
  html += "</div>";
  return html;
}

async function loadData() {
  const corpusName = state.selectedCorpus;
  const segmentName = state.selectedSegment;
  if (!corpusName || !segmentName) {
    showToast("Select a corpus and segment first.", "error", document.getElementById("dataMessage"));
    return;
  }
  const corpus = state.corpora.find((c) => c.name === corpusName) || {};
  const type = corpus.type || "orbit";
  const msg = document.getElementById("dataMessage");
  const wrap = document.getElementById("dataTableWrap");
  const body = document.getElementById("dataTableBody");
  msg.textContent = "Loading records...";
  wrap.style.display = "none";
  try {
    const resp = await SegmentAPI.getData(corpusName, segmentName, type, 10);
    if (!resp.success) throw new Error(resp.message || "Failed to load data");
    const records = (resp.data || {}).records || [];
    if (!records.length) {
      msg.textContent = "No records found in this segment.";
      return;
    }
    body.innerHTML = records
      .map((r) => `
        <tr>
          <td>${escapeHtml(fmt(r.id))}</td>
          <td>${escapeHtml(String(r.text || r.metadata?.text || "")).slice(0, 300) || "—"}</td>
          <td>
            <div class="meta-row"><span class="meta-key">lang</span><span class="meta-value">${escapeHtml(r.lang || "—")}</span></div>
            <div class="meta-row"><span class="meta-key">type</span><span class="meta-value">${escapeHtml(r.type || "—")}</span></div>
            ${renderSearchMeta(r.metadata)}
          </td>
        </tr>`)
      .join("");
    wrap.style.display = "block";
    msg.textContent = `Showing ${records.length} of ${resp.data.total_count ?? records.length} records (${escapeHtml(type)}).`;
  } catch (err) {
    wrap.style.display = "none";
    msg.textContent = err.message;
  }
}

document.getElementById("getDataBtn").addEventListener("click", loadData);
document.getElementById("corpusSearch").addEventListener("input", renderCorpusList);

async function openMeshVisual(segmentName, btn) {
  const corpusName = state.selectedCorpus;
  const corpus = state.corpora.find((c) => c.name === corpusName) || {};
  const type = corpus.type || "orbit";
  try {
    const resp = await SegmentAPI.meshVisualization(corpusName, segmentName, type);
    if (!resp.success) throw new Error(resp.message || "Mesh visualization not available");
    const html = (resp.data || {}).html;
    if (!html) throw new Error("No mesh visualization content returned.");
    showMeshModal(segmentName, html);
  } catch (err) {
    if (btn) {
      btn.classList.add("disabled");
      btn.disabled = true;
      btn.title = "Mesh visualization is not available";
    }
    showToast(err.message || "Mesh visualization unavailable", "error");
  }
}

function showMeshModal(segmentName, html) {
  els.modalTitle.textContent = "Mesh visualization — " + segmentName;
  els.modalOverlay.querySelector(".modal").classList.add("modal-wide");
  els.modalBody.innerHTML = `
    <iframe class="mesh-frame" src="about:blank"></iframe>`;
  els.modalFoot.innerHTML = `
    <button class="btn btn-ghost" id="mCancel">Close</button>`;
  els.modalOverlay.classList.add("show");
  document.getElementById("mCancel").addEventListener("click", closeModal);
  const frame = els.modalBody.querySelector(".mesh-frame");
  const doc = frame.contentDocument;
  doc.open();
  doc.write(html);
  doc.close();
}

document.getElementById("loadSegmentBtn").addEventListener("click", () => {
  if (!isAdmin(state.user)) {
    showToast("Only admins can load segments.", "error");
    return;
  }
  openLoadSegmentModal();
});

function openCreateCorpusModal() {
  els.modalTitle.textContent = "New corpus";
  els.modalBody.innerHTML = `
    <div class="form-grid">
      <div class="field" style="grid-column:1/-1;">
        <label for="mCorpusName">Corpus name</label>
        <input id="mCorpusName" class="input" type="text" placeholder="e.g. product_docs">
      </div>
      <div class="field">
        <label for="mNdbType">Database type</label>
        <select id="mNdbType" class="input">
          <option value="cosmos">Cosmos</option>
          <option value="orbit">Orbit</option>
        </select>
      </div>
    </div>`;
  els.modalFoot.innerHTML = `
    <button class="btn btn-ghost" id="mCancel">Cancel</button>
    <button class="btn btn-primary" id="mCreate">Create</button>`;
  els.modalOverlay.classList.add("show");
  document.getElementById("mCancel").addEventListener("click", closeModal);
  document.getElementById("mCreate").addEventListener("click", async () => {
    const name = document.getElementById("mCorpusName").value.trim();
    const type = document.getElementById("mNdbType").value;
    if (!name) {
      showToast("Corpus name is required.", "error", document.getElementById("mCorpusName"));
      return;
    }
    const btn = document.getElementById("mCreate");
    btn.disabled = true;
    try {
      const resp = await CorpusAPI.create(name, type);
      if (!resp.success) throw new Error(resp.message || "Creation failed");
      closeModal();
      await loadCorpora();
      selectCorpus(name);
      showToast(resp.message, "success");
    } catch (err) {
      showToast(err.message, "error", document.getElementById("mCorpusName"));
      btn.disabled = false;
    }
  });
}

function openLoadSegmentModal() {
  els.modalTitle.textContent = "Load segment into " + state.selectedCorpus;
  els.modalBody.innerHTML = `
    <div class="form-grid">
      <div class="field">
        <label for="mSegName">Segment name</label>
        <input id="mSegName" class="input" type="text" placeholder="e.g. my_data">
      </div>
      <div class="field">
        <label for="mDocType">Document type</label>
        <input id="mDocType" class="input" type="text" placeholder="optional">
      </div>
      <div class="field">
        <label for="mLang">Language</label>
        <input id="mLang" class="input" type="text" placeholder="optional">
      </div>
      <div class="field" style="grid-column:1/-1;">
        <label for="mData">Dataset (JSON)</label>
        <textarea id="mData" class="input" rows="8" placeholder='[{"text": "sample row"}]'></textarea>
        <p class="small muted mt-16">Array of objects or a JSON object with column arrays.</p>
      </div>
    </div>`;
  els.modalFoot.innerHTML = `
    <button class="btn btn-ghost" id="mCancel">Cancel</button>
    <button class="btn btn-primary" id="mLoad">Load segment</button>`;
  els.modalOverlay.classList.add("show");
  document.getElementById("mCancel").addEventListener("click", closeModal);
  document.getElementById("mLoad").addEventListener("click", async () => {
    const name = document.getElementById("mSegName").value.trim();
    const dataRaw = document.getElementById("mData").value.trim();
    if (!name || !dataRaw) {
      showToast("Segment name and dataset are required.", "error");
      return;
    }
    let dataset;
    try {
      dataset = JSON.parse(dataRaw);
    } catch (e) {
      showToast("Dataset is not valid JSON.", "error");
      return;
    }
    const btn = document.getElementById("mLoad");
    btn.disabled = true;
    try {
      const resp = await api("/api/NebulonDB/segment/load_segment", {
        method: "POST",
        body: {
          corpus_name: state.selectedCorpus,
          segment_name: name,
          doc_type: document.getElementById("mDocType").value.trim() || null,
          lang_type: document.getElementById("mLang").value.trim() || null,
          segment_dataset: dataset,
          set_columns: "all",
        },
      });
      if (!resp.success) throw new Error(resp.message || "Load failed");
      closeModal();
      await loadSegments(state.selectedCorpus);
      showToast(resp.message, "success");
    } catch (err) {
      showToast(err.message, "error");
      btn.disabled = false;
    }
  });
}

function openUserModal() {
  els.modalOverlay.querySelector(".modal").classList.remove("modal-wide");
  els.modalTitle.textContent = "Account";
  const name = (state.user && state.user.username) || "—";
  const role = (state.user && state.user.role) || "—";
  const initial = name.charAt(0).toUpperCase();
  const admin = isAdmin(state.user);

  els.modalBody.innerHTML = `
    <div class="account-section">
      <div class="account-profile">
        <div class="account-avatar">${escapeHtml(initial)}</div>
        <div>
          <div class="account-name">${escapeHtml(name)}</div>
          <div class="account-role">${escapeHtml(role)}</div>
        </div>
      </div>
    </div>

    <form autocomplete="off" action="#" onsubmit="return false;" novalidate>
      <div class="account-section">
        <div class="section-label">Change password</div>
        <input type="text" name="ndb_username" autocomplete="username" value="${escapeHtml(name)}" class="autofill-honeypot" aria-hidden="true" tabindex="-1">
        <div class="field">
          <label for="uCurrentPwd">Current password</label>
          <div class="pwd-wrap">
            <input id="uCurrentPwd" class="input" type="password" autocomplete="current-password" placeholder="Locked" value="" disabled>
            <span class="pwd-lock" aria-hidden="true">🔒</span>
          </div>
          <p class="small muted mt-16" style="margin-top:6px;">Locked — the current password is not required.</p>
        </div>
        <div class="field">
          <label for="uNewPwd">New password</label>
          <input id="uNewPwd" class="input" type="password" autocomplete="new-password" placeholder="At least 8 characters">
        </div>
        <div class="field">
          <label for="uConfirmPwd">Confirm new password</label>
          <input id="uConfirmPwd" class="input" type="password" autocomplete="new-password" placeholder="Repeat new password">
        </div>
        <button type="submit" class="btn btn-primary" id="uChangePwdBtn">Change password</button>
      </div>
    </form>

    <div class="divider"></div>

    <div class="account-section">
      <div class="flex-between">
        <div class="section-label" style="margin:0;">Create new user</div>
        ${admin ? `<button class="btn btn-sm btn-ghost" id="uToggleCreate">Add user</button>` : `<span class="small muted">Admin access required</span>`}
      </div>
      <div id="uCreateForm" style="display:none;" class="mt-16">
        <div class="field">
          <label for="uNewUsername">Username</label>
          <input id="uNewUsername" class="input" type="text" placeholder="Min 3 characters">
        </div>
        <div class="field">
          <label for="uNewUserPwd">Password</label>
          <input id="uNewUserPwd" class="input" type="password" autocomplete="new-password" placeholder="Min 6 characters">
        </div>
        <div class="field">
          <label for="uNewUserRole">User type</label>
          <select id="uNewUserRole" class="input">
            <option value="user">User</option>
            <option value="admin_user">Admin</option>
            <option value="super_user">Super user</option>
          </select>
        </div>
        <button class="btn btn-primary" id="uCreateUserBtn">Create user</button>
      </div>
    </div>`;

  els.modalFoot.innerHTML = `
    <button class="btn btn-ghost" id="mCancel">Close</button>`;
  els.modalOverlay.classList.add("show");

  document.getElementById("mCancel").addEventListener("click", closeModal);

  document.getElementById("uChangePwdBtn").addEventListener("click", async () => {
    const current = document.getElementById("uCurrentPwd").value;
    const next = document.getElementById("uNewPwd").value;
    const confirmVal = document.getElementById("uConfirmPwd").value;
    if (!next) {
      showToast("New password is required.", "error");
      return;
    }
    if (next.length < 8) {
      showToast("New password must be at least 8 characters.", "error");
      return;
    }
    if (next !== confirmVal) {
      showToast("New passwords do not match.", "error");
      return;
    }
    const btn = document.getElementById("uChangePwdBtn");
    btn.disabled = true;
    try {
      const resp = await AuthAPI.changePassword(current || "", next);
      if (!resp.success) throw new Error(resp.message || "Password change failed");
      document.getElementById("uCurrentPwd").value = "";
      document.getElementById("uNewPwd").value = "";
      document.getElementById("uConfirmPwd").value = "";
      showToast(resp.message || "Password changed successfully.", "success");
    } catch (err) {
      showToast(err.message || "Password change failed.", "error");
    } finally {
      btn.disabled = false;
    }
  });

  const toggleBtn = document.getElementById("uToggleCreate");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const form = document.getElementById("uCreateForm");
      form.style.display = form.style.display === "none" ? "block" : "none";
    });
  }

  const createBtn = document.getElementById("uCreateUserBtn");
  if (createBtn) {
    createBtn.addEventListener("click", async () => {
      const username = document.getElementById("uNewUsername").value.trim();
      const password = document.getElementById("uNewUserPwd").value;
      const role = document.getElementById("uNewUserRole").value;
      if (!username || !password) {
        showToast("Username and password are required.", "error");
        return;
      }
      if (username.length < 3) {
        showToast("Username must be at least 3 characters.", "error");
        return;
      }
      if (password.length < 6) {
        showToast("Password must be at least 6 characters.", "error");
        return;
      }
      const btn = document.getElementById("uCreateUserBtn");
      btn.disabled = true;
      try {
        const resp = await AuthAPI.register(username, password, role);
        if (!resp.success) throw new Error(resp.message || "User creation failed");
        document.getElementById("uNewUsername").value = "";
        document.getElementById("uNewUserPwd").value = "";
        showToast(resp.message || `User '${username}' created.`, "success");
      } catch (err) {
        showToast(err.message || "User creation failed.", "error");
      } finally {
        btn.disabled = false;
      }
    });
  }
}

const userChipEl = document.getElementById("userChip");
userChipEl.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (document.activeElement && document.activeElement.blur) {
    document.activeElement.blur();
  }
  openUserModal();
});
userChipEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    openUserModal();
  }
});

const SETTINGS_LABELS = {
  app_name: "App name", host: "Host", port: "Port", workers: "Workers",
  timeout: "Timeout (s)", keep_alive: "Keep alive (s)", graceful_timeout: "Graceful timeout (s)",
  access_logfile: "Access logfile", error_logfile: "Error logfile", log_level: "Log level",
  wal_auto_flush: "WAL auto flush", compress_segments: "Compress segments",
  bloom_filter_enabled: "Bloom filter enabled", max_open_segments: "Max open segments",
  compaction_interval: "Compaction interval", max_segments_before_compact: "Max segments before compact",
  flush_interval: "Flush interval", flush_record_threshold: "Flush record threshold",
  enabled: "Enabled", bits_per_key: "Bits per key", hash_count: "Hash count",
  dimension: "Dimension", space: "Space", top_matches: "Top matches", min_score: "Min score",
  save_every_n: "Save every n", compaction_threshold: "Compaction threshold",
  m: "M", ef_construction: "EF construction", ef_search: "EF search",
  rank_topk: "Rank top-k", weight_vector: "Weight vector", weight_bm25: "Weight BM25",
  weight_metadata: "Weight metadata", weight_importance: "Weight importance",
  weight_freshness: "Weight freshness",
};

function openSettingsModal() {
  els.modalTitle.textContent = "Settings";
  els.modalBody.innerHTML = '<div class="muted small">Loading configuration...</div>';
  els.modalFoot.innerHTML = `
    <button class="btn btn-ghost" id="mCancel">Cancel</button>
    <button class="btn btn-primary" id="mSave">Save</button>`;
  els.modalOverlay.querySelector(".modal").classList.add("modal-wide");
  els.modalOverlay.classList.add("show");
  document.getElementById("mCancel").addEventListener("click", closeModal);
  document.getElementById("mSave").addEventListener("click", saveSettings);
  ConfigAPI.get()
    .then((cfg) => renderSettingsForm(cfg))
    .catch((err) => {
      els.modalBody.innerHTML = `<div class="alert alert-error show">${escapeHtml(err.message || "Failed to load configuration")}</div>`;
    });
}

function renderSettingsForm(cfg) {
  const sections = ["server", "segments", "bloom", "vector", "hnsw", "rank"];
  let html = "";
  for (const sec of sections) {
    const vals = cfg[sec] || {};
    const keys = Object.keys(vals).filter((k) => k !== "url");
    if (!keys.length) continue;
    html += `
      <div class="settings-section">
        <div class="section-label">${escapeHtml(sec)}</div>
        <div class="form-grid">`;
    for (const key of keys) {
      const val = vals[key];
      const label = SETTINGS_LABELS[key] || key;
      const inputId = "cfg_" + sec + "_" + key;
      if (typeof val === "boolean") {
        html += `
          <div class="field">
            <label for="${inputId}">${escapeHtml(label)}</label>
            <select id="${inputId}" class="input" data-section="${sec}" data-key="${key}">
              <option value="true" ${val ? "selected" : ""}>Enabled</option>
              <option value="false" ${val ? "" : "selected"}>Disabled</option>
            </select>
          </div>`;
      } else {
        html += `
          <div class="field">
            <label for="${inputId}">${escapeHtml(label)}</label>
            <input id="${inputId}" class="input" type="text" data-section="${sec}" data-key="${key}" value="${escapeHtml(String(val))}">
          </div>`;
      }
    }
    html += `</div></div>`;
  }
  els.modalBody.innerHTML = html;
}

async function saveSettings() {
  const config = {};
  document.querySelectorAll("#modalBody [data-section][data-key]").forEach((el) => {
    const sec = el.dataset.section;
    const key = el.dataset.key;
    if (!config[sec]) config[sec] = {};
    config[sec][key] = el.value;
  });
  const btn = document.getElementById("mSave");
  if (btn) btn.disabled = true;
  try {
    const resp = await ConfigAPI.update(config);
    showToast(resp.message || "Settings saved.", "success");
    closeModal();
  } catch (err) {
    showToast(err.message || "Failed to save settings.", "error");
    if (btn) btn.disabled = false;
  }
}

document.getElementById("settingsBtn").addEventListener("click", () => {
  if (!state.user || state.user.is_authenticated === false) return;
  openSettingsModal();
});

document.getElementById("newCorpusBtn").addEventListener("click", () => {
  if (!isAdmin(state.user)) {
    showToast("Only admins can create corpora.", "error");
    return;
  }
  openCreateCorpusModal();
});

const sidebarEl = document.getElementById("sidebar");
const sidebarToggleEl = document.getElementById("sidebarIconToggle");
const sidebarDragEl = document.getElementById("sidebarDrag");

function setSidebarIcon() {
  sidebarToggleEl.textContent = sidebarEl.classList.contains("collapsed") ? "»" : "«";
}

if (localStorage.getItem("ndb_sidebar_collapsed") === "1") {
  sidebarEl.classList.add("collapsed");
}
setSidebarIcon();

sidebarToggleEl.addEventListener("click", () => {
  sidebarEl.classList.toggle("collapsed");
  setSidebarIcon();
  localStorage.setItem("ndb_sidebar_collapsed", sidebarEl.classList.contains("collapsed") ? "1" : "0");
});

document.getElementById("brandExpand").addEventListener("click", () => {
  if (sidebarEl.classList.contains("collapsed")) {
    sidebarEl.classList.remove("collapsed");
    setSidebarIcon();
    localStorage.setItem("ndb_sidebar_collapsed", "0");
  }
});

let sidebarDragging = false;
let sidebarStartX = 0;
let sidebarStartW = 0;

sidebarDragEl.addEventListener("pointerdown", (e) => {
  sidebarDragging = true;
  sidebarStartX = e.clientX;
  sidebarStartW = sidebarEl.offsetWidth;
  sidebarEl.classList.add("resizing");
  sidebarEl.classList.remove("collapsed");
  sidebarEl.style.width = sidebarStartW + "px";
  sidebarEl.style.minWidth = sidebarStartW + "px";
  sidebarDragEl.setPointerCapture(e.pointerId);
  e.preventDefault();
});

sidebarDragEl.addEventListener("pointermove", (e) => {
  if (!sidebarDragging) return;
  const w = Math.min(360, Math.max(80, sidebarStartW + (e.clientX - sidebarStartX)));
  sidebarEl.style.width = w + "px";
  sidebarEl.style.minWidth = w + "px";
});

function sidebarDragEnd() {
  if (!sidebarDragging) return;
  sidebarDragging = false;
  sidebarEl.classList.remove("resizing");
  sidebarEl.style.width = "";
  sidebarEl.style.minWidth = "";
  if (sidebarEl.offsetWidth < 170) {
    sidebarEl.classList.add("collapsed");
    localStorage.setItem("ndb_sidebar_collapsed", "1");
  } else {
    localStorage.setItem("ndb_sidebar_collapsed", "0");
  }
  setSidebarIcon();
}

sidebarDragEl.addEventListener("pointerup", sidebarDragEnd);
sidebarDragEl.addEventListener("pointercancel", sidebarDragEnd);

function closeModal() {
  els.modalOverlay.querySelector(".modal").classList.remove("modal-wide");
  els.modalOverlay.classList.remove("show");
}
document.getElementById("modalClose").addEventListener("click", closeModal);
els.modalOverlay.addEventListener("click", (e) => {
  if (e.target === els.modalOverlay) closeModal();
});

(async () => {
  await loadUser();
  if (!state.user || state.user.is_authenticated === false) return;
  applyAccess();
  await loadCorpora();
})();
