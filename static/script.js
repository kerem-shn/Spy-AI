/* === SPY AI — Client-Side Logic (v4) === */
(function () {
    "use strict";
    const $ = id => document.getElementById(id);

    const siteHeader = $("site-header"), settingsToggle = $("settings-toggle"), settingsPanel = $("settings-panel");
    const deeplKeyInput = $("deepl-key-input");
    const dirBtns = document.querySelectorAll(".direction-btn");
    const uploadZone = $("upload-zone"), fileInput = $("file-input");
    const fileInfo = $("file-info"), fileName = $("file-name"), fileSize = $("file-size");
    const analyzeBtn = $("analyze-btn"), clearBtn = $("clear-btn");
    const uploadSection = $("upload-section"), loadingSection = $("loading-section");
    const loadingStatus = $("loading-status"), loadingBarFill = $("loading-bar-fill");
    const resultsSection = $("results-section");
    const statTerms = $("stat-terms"), statEntities = $("stat-entities"), statEngine = $("stat-engine");
    const tabTerms = $("tab-terms"), tabEntities = $("tab-entities");
    const termsView = $("terms-view"), entitiesContainer = $("entities-container");
    const sourceTextView = $("source-text-view"), detailAnchor = $("detail-panel-anchor");
    const resultsLayout = $("results-layout");
    const themePicker = $("theme-picker"), accentPicker = $("accent-picker"), positionPicker = $("position-picker");
    const toastContainer = $("toast-container");

    let selectedFile = null, currentDirection = "en-tr", analysisData = null, panelPos = "bottom";

    function init() {
        const sk = localStorage.getItem("spyai_deepl_key"); if (sk) deeplKeyInput.value = sk;
        const sd = localStorage.getItem("spyai_direction"); if (sd) setDirection(sd);
        applyTheme(localStorage.getItem("spyai_theme") || "light");
        applyAccent(localStorage.getItem("spyai_accent") || "default");
        applyPanelPos(localStorage.getItem("spyai_panelpos") || "bottom");
        bindEvents();
    }

    function bindEvents() {
        // Settings toggle
        settingsToggle.addEventListener("click", () => {
            settingsPanel.classList.toggle("settings-panel--open");
            settingsPanel.setAttribute("aria-hidden", !settingsPanel.classList.contains("settings-panel--open"));
        });
        deeplKeyInput.addEventListener("change", () => localStorage.setItem("spyai_deepl_key", deeplKeyInput.value.trim()));
        dirBtns.forEach(b => b.addEventListener("click", () => setDirection(b.dataset.dir)));

        // Upload
        uploadZone.addEventListener("click", () => fileInput.click());
        uploadZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click() } });
        uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("upload-zone--dragover") });
        uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("upload-zone--dragover"));
        uploadZone.addEventListener("drop", e => { e.preventDefault(); uploadZone.classList.remove("upload-zone--dragover"); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]) });
        fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]) });
        analyzeBtn.addEventListener("click", runAnalysis);
        clearBtn.addEventListener("click", clearFile);

        // Tabs
        tabTerms.addEventListener("click", () => switchTab("terms"));
        tabEntities.addEventListener("click", () => switchTab("entities"));

        // Theme, accent, position pickers
        themePicker.querySelectorAll(".theme-swatch").forEach(s => s.addEventListener("click", () => applyTheme(s.dataset.theme)));
        accentPicker.querySelectorAll(".theme-swatch").forEach(s => s.addEventListener("click", () => applyAccent(s.dataset.accent)));
        positionPicker.querySelectorAll(".pos-swatch").forEach(s => s.addEventListener("click", () => applyPanelPos(s.dataset.pos)));

        // Header shrink on scroll — debounced via rAF to prevent glitch
        let scrollTicking = false;
        window.addEventListener("scroll", () => {
            if (!scrollTicking) {
                requestAnimationFrame(() => {
                    siteHeader.classList.toggle("header--compact", window.scrollY > 60);
                    scrollTicking = false;
                });
                scrollTicking = true;
            }
        }, { passive: true });
    }

    // --- Theme / Accent / Position ---
    function applyTheme(t) {
        document.documentElement.setAttribute("data-theme", t);
        localStorage.setItem("spyai_theme", t);
        themePicker.querySelectorAll(".theme-swatch").forEach(s => s.classList.toggle("theme-swatch--active", s.dataset.theme === t));
    }
    function applyAccent(a) {
        document.documentElement.setAttribute("data-accent", a);
        localStorage.setItem("spyai_accent", a);
        accentPicker.querySelectorAll(".theme-swatch").forEach(s => s.classList.toggle("theme-swatch--active", s.dataset.accent === a));
    }
    function applyPanelPos(p) {
        panelPos = p;
        localStorage.setItem("spyai_panelpos", p);
        if (resultsLayout) resultsLayout.setAttribute("data-panel-pos", p);
        positionPicker.querySelectorAll(".pos-swatch").forEach(s => s.classList.toggle("pos-swatch--active", s.dataset.pos === p));
    }
    function setDirection(d) {
        currentDirection = d; localStorage.setItem("spyai_direction", d);
        dirBtns.forEach(b => b.classList.toggle("direction-btn--active", b.dataset.dir === d));
    }

    // --- File ---
    function handleFile(f) {
        const ext = f.name.split(".").pop().toLowerCase();
        if (!["pdf", "docx", "doc", "txt"].includes(ext)) { showToast("Unsupported file type.", "error"); return }
        selectedFile = f; fileName.textContent = f.name; fileSize.textContent = fmtSize(f.size); fileInfo.hidden = false;
    }
    function clearFile() { selectedFile = null; fileInput.value = ""; fileInfo.hidden = true; resultsSection.hidden = true }
    function fmtSize(b) { if (b < 1024) return b + " B"; if (b < 1048576) return (b / 1024).toFixed(1) + " KB"; return (b / 1048576).toFixed(1) + " MB" }

    // --- Analysis ---
    // --- Analysis (Streaming) ---
    async function runAnalysis() {
        if (!selectedFile) { showToast("Please select a file.", "error"); return }
        uploadSection.style.display = "none"; loadingSection.hidden = false; resultsSection.hidden = true;
        loadingStatus.textContent = "Connecting...";
        loadingBarFill.style.width = "5%";

        const fd = new FormData(); fd.append("file", selectedFile); fd.append("direction", currentDirection);
        const key = deeplKeyInput.value.trim(); if (key) fd.append("deepl_key", key);

        analysisData = { source_text: "", terms: {}, entities: {}, stats: {} };

        try {
            const response = await fetch("/upload", { method: "POST", body: fd });
            if (!response.ok) throw new Error("Server error");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const raw = line.substring(6);
                        try {
                            const { type, payload } = JSON.parse(raw);
                            handleStreamEvent(type, payload);
                        } catch (e) { console.warn("JSON Parse Error", e); }
                    }
                }
            }
        } catch (err) {
            showToast(err.message || "Analysis failed.", "error");
            uploadSection.style.display = "";
        } finally {
            loadingSection.hidden = true;
        }
    }

    function handleStreamEvent(type, payload) {
        switch (type) {
            case "status":
                loadingStatus.textContent = payload;
                if (payload.includes("Processing terms")) loadingBarFill.style.width = "40%";
                if (payload.includes("Researching entities")) loadingBarFill.style.width = "75%";
                break;
            case "meta":
                analysisData.source_text = payload.source_text;
                analysisData.stats.total_terms = payload.total_terms;
                analysisData.stats.total_entities = payload.total_entities;
                analysisData.stats.translation_engine = payload.engine;
                initResultsUI();
                break;
            case "term":
                analysisData.terms[payload.lemma] = payload;
                updateStats();
                incrementalRender();
                break;
            case "entity":
                analysisData.entities[payload.name] = payload.summary;
                updateStats();
                renderEntities(analysisData.entities);
                incrementalRender();
                break;
            case "done":
                loadingBarFill.style.width = "100%";
                showToast("Analysis complete!", "success");
                break;
            case "error":
                showToast(payload, "error");
                break;
        }
    }

    function initResultsUI() {
        resultsSection.hidden = false;
        statTerms.textContent = analysisData.stats.total_terms;
        statEntities.textContent = analysisData.stats.total_entities;
        statEngine.textContent = analysisData.stats.translation_engine || "Google Translate";
        sourceTextView.innerHTML = `<p>${analysisData.source_text.replace(/\n/g, "<br>")}</p>`;
        entitiesContainer.innerHTML = "";
        switchTab("terms");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function updateStats() {
        const termsFound = Object.keys(analysisData.terms).length;
        const entsFound = Object.keys(analysisData.entities).length;
        statTerms.textContent = `${termsFound} / ${analysisData.stats.total_terms}`;
        statEntities.textContent = `${entsFound} / ${analysisData.stats.total_entities}`;
    }

    function incrementalRender() {
        renderSourceText(analysisData.source_text, analysisData.terms, analysisData.entities);
    }

    // === SOURCE TEXT with paragraph preservation ===
    function renderSourceText(text, terms, entities) {
        // Build term map: surface form -> lemma
        const termMap = {};
        const textLower = text.toLowerCase();
        for (const [lemma, info] of Object.entries(terms)) {
            const forms = info.originals && info.originals.length ? info.originals : [lemma];
            let foundContiguous = false;
            for (const f of forms) {
                // Check if this surface form actually exists contiguously in text
                if (textLower.includes(f.toLowerCase())) {
                    termMap[f.toLowerCase()] = lemma;
                    foundContiguous = true;
                }
            }
            // Also try the lemma itself
            if (textLower.includes(lemma.toLowerCase())) {
                termMap[lemma.toLowerCase()] = lemma;
                foundContiguous = true;
            }
            // For multi-word terms that DON'T appear contiguously
            // (e.g. "mother patch" when text says "mother" or "herald" patch),
            // register each component word as a clickable link to the compound term.
            if (!foundContiguous && lemma.includes(" ")) {
                const words = lemma.split(/\s+/);
                for (const w of words) {
                    // Only add if this word isn't already mapped to something else
                    if (!termMap[w.toLowerCase()]) {
                        termMap[w.toLowerCase()] = lemma;
                    }
                }
            }
        }
        const entityNames = Object.keys(entities).sort((a, b) => b.length - a.length);

        // Split text into paragraphs (preserve original formatting)
        const paragraphs = text.split(/\n\s*\n|\r\n\s*\r\n/);

        let fullHTML = "";
        for (const para of paragraphs) {
            const trimmed = para.trim();
            if (!trimmed) continue;
            const paraHTML = highlightParagraph(trimmed, termMap, entityNames);
            fullHTML += `<p>${paraHTML}</p>`;
        }
        // If no paragraph breaks, treat each line as a paragraph
        if (paragraphs.length <= 1) {
            const lines = text.split(/\n|\r\n/);
            if (lines.length > 1) {
                fullHTML = "";
                for (const line of lines) {
                    const t = line.trim();
                    if (!t) continue;
                    fullHTML += `<p>${highlightParagraph(t, termMap, entityNames)}</p>`;
                }
            }
        }

        sourceTextView.innerHTML = fullHTML;
        detailAnchor.innerHTML = "";

        // Bind clicks
        sourceTextView.querySelectorAll(".hl-term").forEach(el => {
            el.addEventListener("click", () => {
                sourceTextView.querySelectorAll(".hl-term--active,.hl-entity--active").forEach(a => a.classList.remove("hl-term--active", "hl-entity--active"));
                el.classList.add("hl-term--active");
                showTermDetail(el.dataset.lemma);
            });
        });
        sourceTextView.querySelectorAll(".hl-entity").forEach(el => {
            el.addEventListener("click", () => {
                sourceTextView.querySelectorAll(".hl-term--active,.hl-entity--active").forEach(a => a.classList.remove("hl-term--active", "hl-entity--active"));
                el.classList.add("hl-entity--active");
                showEntityDetail(el.dataset.entity);
            });
        });
    }

    function highlightParagraph(text, termMap, entityNames) {
        // Find entity positions
        const entityPos = [];
        for (const name of entityNames) {
            const re = new RegExp(escRe(name), "gi");
            let m; while ((m = re.exec(text)) !== null) {
                const s = m.index, e = s + m[0].length;
                if (!entityPos.some(p => !(e <= p.start || s >= p.end)))
                    entityPos.push({ start: s, end: e, name, original: m[0] });
            }
        }
        // Find term positions (whole words) — sort entries longest-first
        // so "christmas tree rash" is matched before "rash"
        const termEntries = Object.entries(termMap).sort((a, b) => b[0].length - a[0].length);
        const termPos = [];
        for (const [surface, lemma] of termEntries) {
            const re = new RegExp("\\b" + escRe(surface) + "\\b", "gi");
            let m; while ((m = re.exec(text)) !== null) {
                const s = m.index, e = s + m[0].length;
                if (!entityPos.some(p => !(e <= p.start || s >= p.end)) && !termPos.some(p => !(e <= p.start || s >= p.end)))
                    termPos.push({ start: s, end: e, lemma, original: m[0] });
            }
        }
        const all = [...entityPos.map(p => ({ ...p, type: "entity" })), ...termPos.map(p => ({ ...p, type: "term" }))].sort((a, b) => a.start - b.start);
        let html = "", cursor = 0;
        for (const pos of all) {
            if (pos.start < cursor) continue;
            html += esc(text.slice(cursor, pos.start));
            if (pos.type === "term")
                html += `<span class="hl-term" data-lemma="${esc(pos.lemma)}" title="Click for translation">${esc(pos.original)}</span>`;
            else
                html += `<span class="hl-entity" data-entity="${esc(pos.name)}" title="Click for research">${esc(pos.original)}</span>`;
            cursor = pos.end;
        }
        html += esc(text.slice(cursor));
        return html;
    }

    // === DETAIL PANELS ===
    function showTermDetail(lemma) {
        const info = analysisData.terms[lemma]; if (!info) return;
        const chips = (info.translations || []).map((t, i) => `<span class="translation-chip ${i === 0 ? "translation-chip--primary" : ""}">${esc(t)}</span>`).join("");
        const meanEN = (info.meanings_en || []).map(m => `<li class="meaning-item"><span class="meaning-badge ${m.is_primary ? "meaning-badge--primary" : "meaning-badge--alt"}">${m.is_primary ? "context" : "alt"}</span><span>${esc(m.definition)}</span></li>`).join("");
        const meanTR = (info.meanings_tr || []).map(m => `<li class="meaning-item"><span class="meaning-badge ${m.is_primary ? "meaning-badge--primary" : "meaning-badge--alt"}">${m.is_primary ? "context" : "alt"}</span><span>${esc(m.definition)}</span></li>`).join("");
        const url = `https://www.google.com/search?q=${encodeURIComponent(lemma + " definition")}`;

        detailAnchor.innerHTML = `
        <div class="detail-panel">
            <button class="detail-panel__close" id="detail-close" title="Close">&times;</button>
            <div class="detail-panel__header">
                <span class="detail-panel__word">${esc(lemma)}</span>
                <div class="detail-panel__actions">
                    <div class="meaning-lang-toggle">
                        <button class="meaning-lang-btn meaning-lang-btn--active" data-lang="en" type="button">EN</button>
                        <button class="meaning-lang-btn" data-lang="tr" type="button">TR</button>
                    </div>
                    <a href="${url}" target="_blank" rel="noopener" class="btn--icon" title="Research on Google">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </a>
                </div>
            </div>
            <div class="detail-panel__section"><div class="detail-panel__section-title">Translations</div><div class="translation-chips">${chips}</div></div>
            <div class="detail-panel__section"><div class="detail-panel__section-title">Meanings</div>
                <ul class="meaning-list" id="meanings-en">${meanEN}</ul>
                <ul class="meaning-list" id="meanings-tr" style="display:none">${meanTR}</ul>
            </div>
        </div>`;
        bindPanelEvents();
    }

    function showEntityDetail(name) {
        const info = analysisData.entities[name]; if (!info) return;
        const bc = { "Person": "person", "Organization": "organization", "Place": "place", "Event": "event", "Work of Art": "work", "Group/Nationality": "group" }[info.label_display] || "person";
        const url = `https://www.google.com/search?q=${encodeURIComponent(name)}`;

        detailAnchor.innerHTML = `
        <div class="detail-panel">
            <button class="detail-panel__close" id="detail-close" title="Close">&times;</button>
            <div class="detail-panel__header">
                <span class="detail-panel__word">${esc(name)}</span>
                <span class="entity-type-badge entity-type-badge--${bc}">${esc(info.label_display)}</span>
                <div class="detail-panel__actions">
                    <a href="${url}" target="_blank" rel="noopener" class="btn--icon" title="Research on Google">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    </a>
                </div>
            </div>
            <div class="detail-panel__section"><div class="detail-panel__section-title">Summary</div>
                <p style="font-size:.9rem;color:var(--text-secondary);line-height:1.7">${esc(info.summary)}</p>
                <div style="margin-top:8px;font-size:.72rem;color:var(--text-muted)">Source: ${esc(info.source)}</div>
            </div>
        </div>`;
        bindPanelEvents();
    }

    function bindPanelEvents() {
        $("detail-close").addEventListener("click", () => {
            detailAnchor.innerHTML = "";
            sourceTextView.querySelectorAll(".hl-term--active,.hl-entity--active").forEach(a => a.classList.remove("hl-term--active", "hl-entity--active"));
        });
        detailAnchor.querySelectorAll(".meaning-lang-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const lang = btn.dataset.lang;
                detailAnchor.querySelectorAll(".meaning-lang-btn").forEach(b => b.classList.toggle("meaning-lang-btn--active", b.dataset.lang === lang));
                const en = $("meanings-en"), tr = $("meanings-tr");
                if (en) en.style.display = lang === "en" ? "" : "none";
                if (tr) tr.style.display = lang === "tr" ? "" : "none";
            });
        });
        detailAnchor.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // === ENTITIES TAB ===
    function renderEntities(entities) {
        entitiesContainer.innerHTML = "";
        const entries = Object.entries(entities);
        if (!entries.length) { entitiesContainer.innerHTML = '<div style="text-align:center;padding:48px;color:var(--text-muted)">🔍 No named entities found.</div>'; return }
        entries.forEach(([name, info], idx) => {
            if (!info) return; // guard against null entity summary
            const card = document.createElement("div"); card.classList.add("entity-card");
            card.style.animationDelay = `${Math.min(idx * 0.05, 0.5)}s`;
            const bc = { "Person": "person", "Organization": "organization", "Place": "place", "Event": "event", "Work of Art": "work", "Group/Nationality": "group" }[info.label_display] || "person";
            const url = `https://www.google.com/search?q=${encodeURIComponent(name)}`;
            card.innerHTML = `
                <div class="entity-card__header">
                    <span class="entity-card__name">${esc(name)}</span>
                    <span class="entity-type-badge entity-type-badge--${bc}">${esc(info.label_display)}</span>
                    <div class="entity-card__actions"><a href="${url}" target="_blank" rel="noopener" class="btn--icon" title="Research on Google"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></a></div>
                </div>
                <p class="entity-card__summary">${esc(info.summary)}</p>
                <div class="entity-card__source"><span class="source-dot"></span>Source: ${esc(info.source)}</div>`;
            entitiesContainer.appendChild(card);
        });
    }

    function switchTab(tab) {
        tabTerms.classList.toggle("tab--active", tab === "terms");
        tabEntities.classList.toggle("tab--active", tab === "entities");
        termsView.hidden = tab !== "terms";
        entitiesContainer.hidden = tab !== "entities";
    }

    function showToast(msg, type = "error") {
        const t = document.createElement("div"); t.className = `toast toast--${type}`;
        t.innerHTML = (type === "error" ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-rose)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>' : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-green)" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>') + `<span>${esc(msg)}</span>`;
        toastContainer.appendChild(t); setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 300) }, 5000);
    }

    function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML }
    function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") }

    // ══════════════════════════════════════════════════════════════
    // EXERCISE MODULE
    // ══════════════════════════════════════════════════════════════

    const QUIZ_DATA = [
        {
            id: "dermatology-translation-1",
            title: "Pityriasis Rosea — Translation Exercise",
            description: "Test your knowledge of dermatological terminology translation. Covers key terms from the Pityriasis Rosea source text including Christmas tree rash, herald/mother/daughter patches, and translation problem types.",
            icon: "🩺",
            questionCount: 5,
            questions: [
                {
                    type: "multiple-select",
                    text: "How can <strong>\"Christmas Tree Rash\"</strong> be translated in this particular text?<br><em>(You may select more than one option)</em>",
                    options: [
                        "Gül Hastalığı",
                        "Madalyon Hastalığı",
                        "Noel Ağacı Döküntüsü",
                        "Yılbaşı Ağacı Döküntüsü"
                    ],
                    correct: [0, 1]
                },
                {
                    type: "multiple-choice",
                    text: "\"The first stage will cause a single, large '<strong>mother</strong>' or '<strong>herald</strong>' patch to appear.\"<br>Considering the target audience, which of the following is the most appropriate translation of this sentence?",
                    options: [
                        "İlk etapta '<strong>herald</strong>' veya '<strong>primer</strong>' de denilen, büyük bir tane döküntü oluşur.",
                        "İlk aşamada '<strong>anne</strong>' veya '<strong>müjdeci plak</strong>' da denilen büyük, tek bir döküntü görülür.",
                        "İlk evrede '<strong>birincil lezyon</strong>' veya '<strong>haberci plak</strong>' da denilen büyük, tek bir döküntü ortaya çıkar.",
                        "İlk evrede '<strong>primer</strong>' veya '<strong>haberci plak</strong>' da denilen büyük, tek bir döküntü ortaya çıkar."
                    ],
                    correct: [2]
                },
                {
                    type: "multiple-choice",
                    text: "\"The second stage will include the formation of '<strong>daughter</strong>' patches.\"<br>Which translation problem does the word <strong>\"daughter\"</strong> in this sentence relate to?",
                    options: [
                        "Linguistic Problem",
                        "Pragmatic Problem",
                        "Text-Specific Problem",
                        "Convention-Related Problem"
                    ],
                    correct: [1]
                },
                {
                    type: "multiple-select",
                    text: "\"The '<strong>\patch\</strong>' can be found throughout the body, other than the face, soles of the feet, scalp, and palms.\"<br>What are the most appropriate possible translations of the word <strong>\"patch\"</strong> in the context of this text?<br><em>(You can select more than one option)</em>",
                    options: [
                        "Plak",
                        "Yara",
                        "Kızarıklık",
                        "Lezyon"
                    ],
                    correct: [0, 3]
                },
                {
                    type: "true-false",
                    text: "\"<strong><u>In the USA</u></strong>, about 50 percent of people with this skin condition experience itchiness, according to <strong><u>the American Academy of Dermatology (AAD)</u></strong>.\"<br>Is it correct to translate the underlined parts as they are in this context?",
                    options: [
                        "Yes",
                        "No"
                    ],
                    correct: [1]
                }
            ]
        }
    ];

    // Exercise state
    let exCurrentTestId = null;
    let exCurrentQIndex = 0;
    let exStudentAnswers = [];

    // DOM refs for exercise
    const exerciseToggle = $("exercise-toggle");
    const exerciseSelect = $("exercise-select"), exerciseQuiz = $("exercise-quiz"), exerciseResults = $("exercise-results");
    const exerciseGrid = $("exercise-grid");
    const exerciseBackHome = $("exercise-back-home");
    const quizBackSelect = $("quiz-back-select");
    const quizTitle = $("quiz-title"), quizProgressLabel = $("quiz-progress-label"), quizProgressFill = $("quiz-progress-fill");
    const quizQType = $("quiz-q-type"), quizQText = $("quiz-q-text"), quizOptions = $("quiz-options");
    const quizNextBtn = $("quiz-next-btn"), quizPrevBtn = $("quiz-prev-btn");
    const resultsScoreCircle = $("results-score-circle");
    const resultsScorePct = $("results-score-pct");
    const resultsScoreDetail = $("results-score-detail");
    const resultsReview = $("results-review");
    const resultsBackTests = $("results-back-tests");
    const resultsRetry = $("results-retry");

    function hideAllSections() {
        uploadSection.style.display = "none";
        loadingSection.hidden = true;
        resultsSection.hidden = true;
        exerciseSelect.hidden = true;
        exerciseQuiz.hidden = true;
        exerciseResults.hidden = true;
    }

    function showHomePage() {
        hideAllSections();
        uploadSection.style.display = "";
    }

    function showExerciseSelect() {
        hideAllSections();
        exerciseSelect.hidden = false;
        renderTestCards();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function renderTestCards() {
        exerciseGrid.innerHTML = "";
        QUIZ_DATA.forEach(test => {
            const card = document.createElement("div");
            card.className = "exercise-card";
            card.innerHTML = `
                <div class="exercise-card__icon">${test.icon}</div>
                <div class="exercise-card__title">${esc(test.title)}</div>
                <div class="exercise-card__desc">${esc(test.description)}</div>
                <div class="exercise-card__meta">
                    <span class="exercise-card__badge exercise-card__badge--questions">${test.questionCount} Questions</span>
                    <span class="exercise-card__badge exercise-card__badge--type">Mixed Types</span>
                    <span class="exercise-card__badge exercise-card__badge--time">~5 min</span>
                </div>
                <div class="exercise-card__start">
                    <button class="btn btn--primary" data-test-id="${esc(test.id)}" type="button">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        Start Test
                    </button>
                </div>`;

            const btn = card.querySelector(".btn--primary");

            fetch(`/api/has_taken/${test.id}`)
                .then(r => r.json())
                .then(data => {
                    if (data.taken) {
                        btn.disabled = true;
                        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Completed`;
                        btn.style.background = "var(--accent-green, #22c55e)";
                        btn.style.opacity = "0.8";
                        btn.style.cursor = "not-allowed";
                        card.style.opacity = "0.75";
                    }
                }).catch(() => {});

            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                startQuiz(test.id);
            });
            exerciseGrid.appendChild(card);
        });
    }

    async function startQuiz(testId) {
        const test = QUIZ_DATA.find(t => t.id === testId);
        if (!test) return;

        try {
            const res = await fetch(`/api/has_taken/${testId}`);
            const data = await res.json();
            if (data.taken) {
                showToast("You have already completed this test. Each test can only be taken once.", "error");
                return;
            }
        } catch (e) { }

        exCurrentTestId = testId;
        exCurrentQIndex = 0;
        exStudentAnswers = test.questions.map(() => []);
        hideAllSections();
        exerciseQuiz.hidden = false;
        quizTitle.textContent = test.title;
        renderQuestion();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function getCurrentTest() {
        return QUIZ_DATA.find(t => t.id === exCurrentTestId);
    }

    function renderQuestion() {
        const test = getCurrentTest();
        if (!test) return;
        const q = test.questions[exCurrentQIndex];
        const total = test.questions.length;
        const num = exCurrentQIndex + 1;

        quizProgressLabel.textContent = `${num} / ${total}`;
        quizProgressFill.style.width = `${(num / total) * 100}%`;

        const typeLabels = {
            "multiple-choice": "Multiple Choice",
            "multiple-select": "Multiple Select",
            "true-false": "True or False"
        };
        quizQType.textContent = typeLabels[q.type] || q.type;
        quizQText.innerHTML = q.text;

        quizOptions.innerHTML = "";
        const isMulti = q.type === "multiple-select";
        q.options.forEach((opt, idx) => {
            const div = document.createElement("div");
            div.className = "quiz-option" + (isMulti ? " quiz-option--checkbox" : "");
            div.innerHTML = opt;
            if (exStudentAnswers[exCurrentQIndex].includes(idx)) {
                div.classList.add("quiz-option--selected");
            }
            div.addEventListener("click", () => {
                if (isMulti) {
                    div.classList.toggle("quiz-option--selected");
                    const sel = exStudentAnswers[exCurrentQIndex];
                    if (sel.includes(idx)) {
                        exStudentAnswers[exCurrentQIndex] = sel.filter(i => i !== idx);
                    } else {
                        sel.push(idx);
                    }
                } else {
                    quizOptions.querySelectorAll(".quiz-option").forEach(o => o.classList.remove("quiz-option--selected"));
                    div.classList.add("quiz-option--selected");
                    exStudentAnswers[exCurrentQIndex] = [idx];
                }
            });
            quizOptions.appendChild(div);
        });

        quizNextBtn.textContent = num === total ? "Submit" : "Next →";
        quizPrevBtn.style.display = exCurrentQIndex > 0 ? "" : "none";
    }

    function handleQuizPrev() {
        if (exCurrentQIndex <= 0) return;
        exCurrentQIndex--;
        renderQuestion();
        const card = $("quiz-question-card");
        if (card) {
            card.style.animation = "none";
            card.offsetHeight;
            card.style.animation = "panelIn .3s ease";
        }
    }

    function handleQuizNext() {
        const test = getCurrentTest();
        if (!test) return;
        if (exStudentAnswers[exCurrentQIndex].length === 0) {
            showToast("Please select an answer before continuing.", "error");
            return;
        }
        if (exCurrentQIndex < test.questions.length - 1) {
            exCurrentQIndex++;
            renderQuestion();
            fetch("/api/update_progress", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    test_id: exCurrentTestId,
                    question_index: exCurrentQIndex,
                    total: test.questions.length
                })
            }).catch(() => {});
            const card = $("quiz-question-card");
            if (card) {
                card.style.animation = "none";
                card.offsetHeight;
                card.style.animation = "panelIn .3s ease";
            }
        } else {
            showResults();
        }
    }

    function showResults() {
        const test = getCurrentTest();
        if (!test) return;

        hideAllSections();
        exerciseResults.hidden = false;

        let correctCount = 0;
        test.questions.forEach((q, idx) => {
            const student = [...exStudentAnswers[idx]].sort().join(",");
            const correct = [...q.correct].sort().join(",");
            if (student === correct) correctCount++;
        });

        const pct = Math.round((correctCount / test.questions.length) * 100);

        const circumference = 2 * Math.PI * 52;
        const offset = circumference - (pct / 100) * circumference;
        resultsScoreCircle.style.strokeDasharray = circumference;
        resultsScoreCircle.style.strokeDashoffset = circumference;
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                resultsScoreCircle.style.strokeDashoffset = offset;
            });
        });

        resultsScorePct.textContent = pct + "%";
        resultsScoreDetail.textContent = `${correctCount} of ${test.questions.length} correct`;

        fetch("/api/save_result", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                test_id: test.id,
                score: correctCount,
                total: test.questions.length,
                answers: exStudentAnswers
            })
        }).catch(err => console.error("Failed to save quiz result", err));

        resultsReview.innerHTML = "";
        test.questions.forEach((q, idx) => {
            const studentAns = exStudentAnswers[idx];
            const studentSorted = [...studentAns].sort().join(",");
            const correctSorted = [...q.correct].sort().join(",");
            const isCorrect = studentSorted === correctSorted;

            const card = document.createElement("div");
            card.className = `review-card ${isCorrect ? "review-card--correct" : "review-card--incorrect"}`;
            card.style.animationDelay = `${idx * 0.08}s`;

            const yourAnswerText = studentAns.length > 0
                ? studentAns.map(i => q.options[i]).join(", ")
                : "No answer";
            const correctAnswerText = q.correct.map(i => q.options[i]).join(", ");

            const plainQ = q.text.replace(/<[^>]+>/g, '');

            let answersHTML = "";
            if (isCorrect) {
                answersHTML = `
                    <div class="review-card__answer review-card__answer--correct-answer">
                        <span class="review-card__answer-label review-card__answer-label--correct">✓ Correct</span>
                        <span>${correctAnswerText}</span>
                    </div>`;
            } else {
                answersHTML = `
                    <div class="review-card__answer review-card__answer--yours">
                        <span class="review-card__answer-label review-card__answer-label--yours">Your Answer</span>
                        <span>${yourAnswerText}</span>
                    </div>
                    <div class="review-card__answer review-card__answer--correct-answer">
                        <span class="review-card__answer-label review-card__answer-label--correct">Correct Answer</span>
                        <span>${correctAnswerText}</span>
                    </div>`;
            }

            card.innerHTML = `
                <div class="review-card__header">
                    <span class="review-card__num">Q${idx + 1}</span>
                    <span class="review-card__result ${isCorrect ? "review-card__result--correct" : "review-card__result--incorrect"}">
                        ${isCorrect ? "✓ Correct" : "✗ Incorrect"}
                    </span>
                </div>
                <div class="review-card__question">${esc(plainQ)}</div>
                <div class="review-card__answers">${answersHTML}</div>`;
            resultsReview.appendChild(card);
        });

        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function bindExerciseEvents() {
        exerciseToggle.addEventListener("click", () => {
            settingsPanel.classList.remove("settings-panel--open");
            settingsPanel.setAttribute("aria-hidden", "true");
            showExerciseSelect();
        });
        exerciseBackHome.addEventListener("click", showHomePage);
        quizBackSelect.addEventListener("click", () => {
            if (confirm("Are you sure you want to leave this test? Your progress will be lost.")) {
                showExerciseSelect();
            }
        });
        quizNextBtn.addEventListener("click", handleQuizNext);
        quizPrevBtn.addEventListener("click", handleQuizPrev);
        resultsBackTests.addEventListener("click", showExerciseSelect);
        resultsRetry.addEventListener("click", () => {
            if (exCurrentTestId) startQuiz(exCurrentTestId);
        });
    }

    bindExerciseEvents();

    init();
})();
