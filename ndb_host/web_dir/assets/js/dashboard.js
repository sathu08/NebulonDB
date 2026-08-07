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
  searchResultsWrap: document.getElementById("searchResultsWrap"),
  searchResultsBody: document.getElementById("searchResultsBody"),
  searchMessage: document.getElementById("searchMessage"),
  recordView: document.getElementById("recordView"),
  topbarTitle: document.getElementById("topbarTitle"),
  userName: document.getElementById("userName"),
  userRole: document.getElementById("userRole"),
  userAvatar: document.getElementById("userAvatar"),
  modalOverlay: document.getElementById("modalOverlay"),
  modalTitle: document.getElementById("modalTitle"),
  modalBody: document.getElementById("modalBody"),
  modalFoot: document.getElementById("modalFoot"),
};

function showToast(message, type, target) {
  const el = target || document.getElementById("searchMessage");
  if (!el) return;
  el.textContent = message;
  el.className = "mt-16 small " + (type === "error" ? "alert-error" : type === "success" ? "alert-success" : "muted");
}

function logout() {
  clearCredentials();
  window.location.href = "index.html";
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
    state.corpora = (data.corpus_list || []).map((name) => ({ name }));
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
  els.corpusBadge.innerHTML = statusBadge(state.statusMap[name]);
  els.corpusMeta.textContent = ` • ${state.segmentCounts[name] ?? 0} segments`;
  renderCorpusActions();
}

function renderCorpusActions() {
  els.corpusActions.innerHTML = "";
  const name = state.selectedCorpus;
  if (!name) return;
  const status = state.statusMap[name];

  const actBtn = document.createElement("button");
  actBtn.className = "btn btn-sm btn-warning";
  actBtn.textContent = status === "active" ? "Deactivate" : "Activate";
  actBtn.addEventListener("click", async () => {
    try {
      const resp = status === "active"
        ? await CorpusAPI.deactivate(name)
        : await CorpusAPI.activate(name);
      if (!resp.success) throw new Error(resp.message || "Action failed");
      state.statusMap[name] = status === "active" ? "deactivate" : "active";
      showToast(resp.message, "success", document.getElementById("searchMessage"));
      renderCorpusList();
      els.corpusBadge.innerHTML = statusBadge(state.statusMap[name]);
    } catch (err) {
      showToast(err.message, "error", document.getElementById("searchMessage"));
    }
  });

  const delBtn = document.createElement("button");
  delBtn.className = "btn btn-sm btn-danger";
  delBtn.textContent = "Delete";
  delBtn.addEventListener("click", async () => {
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
      showToast(resp.message, "success", document.getElementById("searchMessage"));
    } catch (err) {
      showToast(err.message, "error", document.getElementById("searchMessage"));
    }
  });

  const admin = isAdmin(state.user);
  if (admin) els.corpusActions.append(actBtn, delBtn);
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
            <td><button class="btn btn-ghost btn-sm view-btn" data-segment="${escapeHtml(s.name || "")}">View data</button></td>
          </tr>`
        )
        .join("");
      els.segmentTableBody.querySelectorAll("[data-segment]").forEach((el) => {
        el.addEventListener("click", (e) => {
          if (e.target.closest(".view-btn")) {
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
  els.searchResultsWrap.style.display = "none";
  els.recordView.style.display = "none";
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

async function runSearch() {
  const query = document.getElementById("searchInput").value.trim();
  if (!query) {
    showToast("Enter a search term first.", "error");
    return;
  }
  const top = parseInt(document.getElementById("searchTop").value, 10) || 10;
  els.searchMessage.textContent = "Searching...";
  try {
    const resp = await SegmentAPI.search(state.selectedCorpus, state.selectedSegment, query, top);
    if (!resp.success) throw new Error(resp.message || "Search failed");
    const results = resp.data || [];
    els.searchResultsWrap.style.display = results.length ? "block" : "none";
    els.searchMessage.textContent = resp.message || (results.length ? "" : "No results found.");
    els.searchResultsBody.innerHTML = results
      .map((r) => `
        <tr>
          <td>${escapeHtml(fmt(r.id))}</td>
          <td>${typeof r.score === "number" ? r.score.toFixed(4) : escapeHtml(fmt(r.score))}</td>
          <td>${renderSearchMeta(r.metadata)}</td>
          <td><button class="btn btn-ghost btn-sm" data-id="${escapeHtml(fmt(r.id))}" data-record>View</button></td>
        </tr>`)
      .join("");
    els.searchResultsBody.querySelectorAll("[data-record]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const rid = parseInt(btn.dataset.id, 10);
        if (!isNaN(rid)) fetchRecord(rid);
      });
    });
  } catch (err) {
    els.searchResultsWrap.style.display = "none";
    showToast(err.message, "error");
  }
}

function renderSearchMeta(meta) {
  if (!meta || typeof meta !== "object") return "—";
  const text = meta.text || meta.content || meta.title || meta.doc || "";
  const keys = Object.keys(meta).filter((k) => !["text", "content", "title", "doc"].includes(k));
  let html = "";
  if (text) {
    html += `<div style="margin-bottom:4px;">${escapeHtml(text).slice(0, 400)}${escapeHtml(text).length > 400 ? "…" : ""}</div>`;
  }
  if (keys.length) {
    html += `<div class="muted small">${keys.slice(0, 6).map((k) => `${escapeHtml(k)}: ${escapeHtml(fmt(meta[k])).slice(0, 60)}`).join(" · ")}</div>`;
  }
  return html || "—";
}

function compactVectors(obj) {
  if (obj === null || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) {
    if (obj.length > 0 && obj.every((v) => typeof v === "number")) {
      return `[${obj.length} floats] ${obj.slice(0, 8).join(", ")}…`;
    }
    return obj.map(compactVectors);
  }
  const out = {};
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    if (Array.isArray(v) && v.length > 24 && v.every((n) => typeof n === "number")) {
      out[k] = `[${v.length} floats]`;
    } else {
      out[k] = compactVectors(v);
    }
  }
  return out;
}

function renderRecord(record) {
  els.recordView.style.display = "block";
  const compact = compactVectors(record);
  els.recordView.innerHTML = `
    <div class="section-label">Full record</div>
    <pre class="json-pre">${escapeHtml(JSON.stringify(compact, null, 2))}</pre>`;
}

async function fetchRecord(recordId) {
  try {
    const resp = await SegmentAPI.getRecord(state.selectedCorpus, state.selectedSegment, recordId);
    if (!resp.success) throw new Error(resp.message || "Record not found");
    renderRecord(resp.data);
  } catch (err) {
    els.recordView.style.display = "block";
    els.recordView.innerHTML = `<div class="alert alert-error show">${escapeHtml(err.message)}</div>`;
  }
}

document.getElementById("searchBtn").addEventListener("click", runSearch);
document.getElementById("searchInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});
document.getElementById("recordBtn").addEventListener("click", () => {
  const rid = parseInt(document.getElementById("recordIdInput").value, 10);
  if (isNaN(rid)) {
    showToast("Enter a valid record ID.", "error");
    return;
  }
  fetchRecord(rid);
});
document.getElementById("corpusSearch").addEventListener("input", renderCorpusList);

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

document.getElementById("newCorpusBtn").addEventListener("click", () => {
  if (!isAdmin(state.user)) {
    showToast("Only admins can create corpora.", "error");
    return;
  }
  openCreateCorpusModal();
});

function closeModal() {
  els.modalOverlay.classList.remove("show");
}
document.getElementById("modalClose").addEventListener("click", closeModal);
els.modalOverlay.addEventListener("click", (e) => {
  if (e.target === els.modalOverlay) closeModal();
});

(async () => {
  await loadUser();
  if (!state.user || state.user.is_authenticated === false) return;
  await loadCorpora();
})();
