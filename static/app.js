(function() {
  let clips = [];
  let votes = { good: [], bad: [] };
  let selected = null;
  let sortOrder = null;   // null = default, or [{id, score}, ...]
  let sortMode = "text";  // "text" | "learned" | "load"
  let selectMode = "top"; // "top" | "hard"
  let threshold = null;    // threshold for Good/Bad boundary
  let sortTimer = null;
  let inclusion = 0;       // Inclusion setting: -10 to +10
  let favoriteDetectors = [];  // List of favorite detectors
  let loadedDetector = null; // Stores loaded detector model weights
  let datasetLoaded = false;
  let audioVolume = 1.0; // Persisted volume across clip loads
  let progressTimer = null;
  const clipList = document.getElementById("clip-list");
  const center = document.getElementById("center");
  const goodList = document.getElementById("good-list");
  const badList = document.getElementById("bad-list");
  const textSortInput = document.getElementById("text-sort");
  const textSortWrap = document.getElementById("text-sort-wrap");
  const loadSortWrap = document.getElementById("load-sort-wrap");
  const loadDetectorBtn = document.getElementById("load-detector-btn");
  const loadDetectorFile = document.getElementById("load-detector-file");
  const learnedRadio = document.getElementById("learned-radio");
  const loadRadio = document.getElementById("load-radio");
  const sortStatus = document.getElementById("sort-status");
  const sortProgress = document.getElementById("sort-progress");
  const sortProgressLabel = document.getElementById("sort-progress-label");
  const sortProgressFill = document.querySelector(".sort-progress-fill");
  let sortProgressTimer = null;

  function showSortProgress(label) {
    sortStatus.textContent = "";
    sortProgressLabel.textContent = label;
    sortProgressFill.style.width = "";
    sortProgressFill.classList.remove("determinate");
    sortProgress.classList.add("active");
  }

  function showSortProgressWithPolling(label) {
    showSortProgress(label);
    startSortProgressPolling();
  }

  function hideSortProgress() {
    stopSortProgressPolling();
    sortProgress.classList.remove("active");
  }

  async function pollSortProgress() {
    try {
      const res = await fetch("/api/sort/progress");
      const progress = await res.json();
      if (progress.status === "idle") return;
      if (progress.total > 0) {
        const pct = Math.round((progress.current / progress.total) * 100);
        sortProgressFill.classList.add("determinate");
        sortProgressFill.style.width = `${pct}%`;
      }
      if (progress.message) {
        sortProgressLabel.textContent = progress.message;
      }
    } catch (_) {
      // ignore polling errors
    }
  }

  function startSortProgressPolling() {
    if (sortProgressTimer) return;
    sortProgressTimer = setInterval(pollSortProgress, 200);
  }

  function stopSortProgressPolling() {
    if (sortProgressTimer) {
      clearInterval(sortProgressTimer);
      sortProgressTimer = null;
    }
  }
  const stripeOverview = document.getElementById("stripe-overview");
  const stripeContainer = document.getElementById("stripe-container");
  const inclusionSlider = document.getElementById("inclusion-slider");
  const inclusionValue = document.getElementById("inclusion-value");

  // Dataset management elements
  const datasetWelcome = document.getElementById("dataset-welcome");
  const datasetOptions = document.getElementById("dataset-options");
  const datasetProgress = document.getElementById("dataset-progress");
  const progressFill = document.getElementById("progress-fill");
  const progressText = document.getElementById("progress-text");
  const progressMessage = document.getElementById("progress-message");
  const demoDatasetsDiv = document.getElementById("demo-datasets");
  const extendedImporterForm = document.getElementById("extended-importer-form");
  const backButton = document.getElementById("back-button");
  const loadFileBtn = document.getElementById("load-file-btn");
  const fileInput = document.getElementById("file-input");
  const datasetBar = document.getElementById("dataset-bar");
  const datasetInfo = document.getElementById("dataset-info");
  const leftPanel = document.getElementById("left-panel");
  const sortBar = document.getElementById("sort-bar");

  // Burger menu elements
  const burgerBtn = document.getElementById("burger-btn");
  const burgerDropdown = document.getElementById("burger-dropdown");
  const menuDatasetExport = document.getElementById("menu-dataset-export");
  const menuDatasetChange = document.getElementById("menu-dataset-change");
  const menuLabelsExport = document.getElementById("menu-labels-export");
  const menuLabelsImport = document.getElementById("menu-labels-import");
  const menuLabelsStatus = document.getElementById("menu-labels-status");
  const menuDetectorImport = document.getElementById("menu-detector-import");
  const menuDetectorExport = document.getElementById("menu-detector-export");
  const menuDetectorStatus = document.getElementById("menu-detector-status");
  const labelImporterModal = document.getElementById("label-importer-modal");
  const labelImporterModalClose = document.getElementById("label-importer-modal-close");
  const labelImporterList = document.getElementById("label-importer-list");
  const labelImporterFormDiv = document.getElementById("label-importer-form");
  const labelImporterBack = document.getElementById("label-importer-back");
  const menuFavoritesManage = document.getElementById("menu-favorites-manage");
  const menuFavoritesSave = document.getElementById("menu-favorites-save");
  const menuFavoritesAutodetect = document.getElementById("menu-favorites-autodetect");
  const favoritesModal = document.getElementById("favorites-modal");
  const favoritesModalClose = document.getElementById("favorites-modal-close");
  const favoritesList = document.getElementById("favorites-list");
  const favAddName = document.getElementById("fav-add-name");
  const favAddStatus = document.getElementById("fav-add-status");
  const favAddFromVotesBtn = document.getElementById("fav-add-from-votes-btn");
  const favAddFromDetectorBtn = document.getElementById("fav-add-from-detector-btn");
  const favAddFromLabelsBtn = document.getElementById("fav-add-from-labels-btn");
  const favDetectorFileInput = document.getElementById("fav-detector-file-input");
  const favLabelsFileInput = document.getElementById("fav-labels-file-input");
  const autodetectModal = document.getElementById("autodetect-modal");
  const autodetectModalClose = document.getElementById("autodetect-modal-close");
  const autodetectSummary = document.getElementById("autodetect-summary");
  const autodetectResults = document.getElementById("autodetect-results");
  const copyResultsBtn = document.getElementById("copy-results-btn");
  const autodetectProgressModal = document.getElementById("autodetect-progress-modal");
  const autodetectProgressText = document.getElementById("autodetect-progress-text");
  const autodetectProgressBar = document.getElementById("autodetect-progress-bar");

  // ---- Dataset Management ----

  async function checkDatasetStatus() {
    const res = await fetch("/api/dataset/status");
    const status = await res.json();
    datasetLoaded = status.loaded;

    if (datasetLoaded) {
      showMainUI();
      const mediaHeaderConfig = {
        "audio":     { icon: "🔊", label: "audio" },
        "video":     { icon: "🎬", label: "video" },
        "image":     { icon: "🖼️", label: "image" },
        "paragraph": { icon: "📄", label: "text" },
      };
      const mhc = mediaHeaderConfig[status.media_type];
      datasetInfo.textContent = mhc
        ? `${mhc.icon} ${status.num_clips} ${mhc.label} clips loaded`
        : `${status.num_clips} clips loaded`;
    } else {
      showWelcomeScreen();
    }

    return status;
  }

  function showWelcomeScreen() {
    center.innerHTML = "";
    center.appendChild(datasetWelcome);
    datasetWelcome.classList.remove("wide");
    datasetWelcome.style.display = "flex";
    center.className = "panel-center";
    datasetOptions.style.display = "flex";
    datasetProgress.style.display = "none";
    demoDatasetsDiv.style.display = "none";
    extendedImporterForm.style.display = "none";
    backButton.style.display = "none";
    sortBar.style.display = "none";
    datasetBar.style.display = "none";
    clipList.innerHTML = "";
  }

  function showMainUI() {
    datasetWelcome.style.display = "none";
    sortBar.style.display = "block";
    datasetBar.style.display = "flex";
    if (!selected) {
      center.className = "panel-center empty";
      center.innerHTML = "<p>Select a clip from the left panel</p>";
    }
  }

  function showProgress() {
    datasetOptions.style.display = "none";
    demoDatasetsDiv.style.display = "none";
    extendedImporterForm.style.display = "none";
    datasetProgress.style.display = "block";
    backButton.style.display = "none";
    // Reset progress bar to indeterminate state
    progressFill.style.width = "0%";
    progressFill.classList.add("indeterminate");
    progressText.textContent = "";
    progressMessage.textContent = "Loading...";
    progressMessage.style.color = "#aaa";
  }

  async function pollProgress() {
    const res = await fetch("/api/dataset/progress");
    const progress = await res.json();

    if (progress.error) {
      progressMessage.textContent = `Error: ${progress.error}`;
      progressMessage.style.color = "#f44336";
      stopProgressPolling();
      setTimeout(() => {
        showWelcomeScreen();
      }, 3000);
      return;
    }

    if (progress.status === "idle") {
      stopProgressPolling();
      await checkDatasetStatus();
      if (datasetLoaded) {
        await fetchClips();
        await fetchVotes();

        // Auto-select first clip if none selected
        if (clips.length > 0 && !selected) {
          selectClip(clips[0].id);
        }

        // Check if auto-detect mode is enabled
        const autodetectCheckbox = document.getElementById("autodetect-mode-checkbox");
        if (autodetectCheckbox && autodetectCheckbox.checked) {
          // Wait a moment for UI to settle
          setTimeout(async () => {
            await runAutoDetectAfterLoad();
          }, 500);
        }
      }
      return;
    }

    // Update progress bar
    if (progress.total > 0) {
      const percentage = Math.round((progress.current / progress.total) * 100);
      progressFill.classList.remove("indeterminate");
      progressFill.style.width = `${percentage}%`;
      progressText.textContent = `${percentage}%`;
    } else {
      // Indeterminate state - no total known yet
      progressFill.classList.add("indeterminate");
      progressText.textContent = "";
    }
    progressMessage.textContent = progress.message || "Loading...";
    progressMessage.style.color = "#aaa";
  }

  function startProgressPolling() {
    if (progressTimer) return;
    showProgress();
    progressTimer = setInterval(pollProgress, 500);
  }

  function stopProgressPolling() {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    progressFill.classList.remove("indeterminate");
  }

  // Load from file
  loadFileBtn.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    startProgressPolling();

    try {
      const res = await fetch("/api/dataset/load-file", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        progressMessage.textContent = `Error: ${error.error}`;
        progressMessage.style.color = "#f44336";
        stopProgressPolling();
      }
    } catch (e) {
      progressMessage.textContent = `Error: ${e.message}`;
      progressMessage.style.color = "#f44336";
      stopProgressPolling();
    }

    fileInput.value = "";
  });

  async function loadDemo(name) {
    startProgressPolling();

    try {
      const res = await fetch("/api/dataset/load-demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });

      if (!res.ok) {
        const error = await res.json();
        progressMessage.textContent = `Error: ${error.error}`;
        progressMessage.style.color = "#f44336";
        stopProgressPolling();
      }
    } catch (e) {
      progressMessage.textContent = `Error: ${e.message}`;
      progressMessage.style.color = "#f44336";
      stopProgressPolling();
    }
  }

  // ---- Extended importers (auto-discovered via /api/dataset/importers) ----

  async function loadExtendedImporters() {
    try {
      const res = await fetch("/api/dataset/importers");
      if (!res.ok) return;
      const data = await res.json();
      for (const importer of data.importers) {
        const btn = document.createElement("button");
        btn.className = "dataset-option";
        btn.innerHTML = `<h3>${importer.icon || "🔌"} ${importer.display_name}</h3><p>${importer.description}</p>`;
        btn.addEventListener("click", () => showExtendedImporterForm(importer));
        datasetOptions.appendChild(btn);
      }
    } catch (_) {
      // Extended importers are optional – silently ignore failures.
    }

    // Append the Load Demo Dataset button after all dynamic importers
    const demoBtnEl = document.createElement("button");
    demoBtnEl.className = "dataset-option";
    demoBtnEl.id = "load-demo-btn";
    demoBtnEl.innerHTML = `<h3>🏆 Load Demo Dataset</h3><p>Choose from a selection of pre-configured demo datasets</p>`;
    demoBtnEl.addEventListener("click", async () => {
      datasetOptions.style.display = "none";
      demoDatasetsDiv.style.display = "grid";
      backButton.style.display = "block";
      datasetWelcome.classList.add("wide");

      // Fetch demo datasets
      try {
        const res = await fetch("/api/dataset/demo-list");
        if (!res.ok) {
          throw new Error(`Server returned ${res.status}`);
        }
        const data = await res.json();

        demoDatasetsDiv.innerHTML = "";

        // Group datasets by media type and display in fixed column order
        const mediaOrder = ["video", "image", "audio", "paragraph"];
        const mediaConfig = {
          video:     { title: "Videos",  icon: "🎬", fileLabel: "video clips" },
          image:     { title: "Images",  icon: "🖼",  fileLabel: "images" },
          audio:     { title: "Sounds",  icon: "🔊", fileLabel: "sound files" },
          paragraph: { title: "Texts",   icon: "📄", fileLabel: "text snippets" },
        };

        const grouped = {};
        data.datasets.forEach(ds => {
          const mt = ds.media_type || "audio";
          if (!grouped[mt]) grouped[mt] = [];
          grouped[mt].push(ds);
        });

        mediaOrder.forEach(mt => {
          const items = grouped[mt] || [];
          if (!items.length) return;
          const cfg = mediaConfig[mt];

          const col = document.createElement("div");
          col.className = "demo-column";
          col.innerHTML = `<div class="demo-column-header">${cfg.title}</div>`;

          items.forEach(dataset => {
            const div = document.createElement("div");
            div.className = "demo-dataset" + (dataset.ready ? " ready" : "");
            const sizeText = dataset.ready
              ? `${dataset.download_size_mb} MB (cached)`
              : `${dataset.download_size_mb} MB to download`;
            div.innerHTML = `
              <h4>${dataset.label}</h4>
              <p style="margin: 4px 0 8px; font-size: 0.75rem; color: #999; line-height: 1.45;">${dataset.description}</p>
              <p style="margin: 0; font-size: 0.72rem; color: #666;">${cfg.icon} ${dataset.num_files} ${cfg.fileLabel} &middot; ${sizeText}</p>
              ${dataset.ready ? '<span class="ready-badge">Ready</span>' : '<span style="font-size:0.7rem;color:#666;display:inline-block;margin-top:6px;">Needs download</span>'}
            `;
            div.onclick = () => loadDemo(dataset.name);
            col.appendChild(div);
          });

          demoDatasetsDiv.appendChild(col);
        });
      } catch (e) {
        demoDatasetsDiv.innerHTML = `<div style="color:#f44336; text-align:center;">Error loading demo datasets: ${e.message}</div>`;
      }
    });
    datasetOptions.appendChild(demoBtnEl);

    // Always render the autodetect toggle last, after all import options
    const autodetectDiv = document.createElement("div");
    autodetectDiv.id = "autodetect-toggle";
    autodetectDiv.style = "margin-top: 16px; padding: 12px; background: #2a2d3a; border-radius: 4px;";
    autodetectDiv.innerHTML = `
      <label style="display: flex; align-items: center; color: #e0e0e0; cursor: pointer;">
        <input type="checkbox" id="autodetect-mode-checkbox" style="margin-right: 8px;">
        <span>Run auto-detect after loading (skip manual labeling)</span>
      </label>
      <p style="font-size: 0.85rem; color: #aaa; margin: 8px 0 0 0;">When checked, automatically runs all favorite detectors and shows positive hits.</p>
    `;
    datasetOptions.appendChild(autodetectDiv);
  }

  function showExtendedImporterForm(importer) {
    datasetOptions.style.display = "none";
    backButton.style.display = "block";

    const inputStyle = "width:100%;padding:8px;background:#252940;border:1px solid #2a2d3a;border-radius:4px;color:#e0e0e0;box-sizing:border-box;";
    let html = `<div style="max-width:420px;width:100%;margin:0 auto;">`;
    html += `<h3 style="margin-bottom:16px;color:#e0e0e0;">${importer.display_name}</h3>`;
    html += `<form id="ext-imp-form">`;
    for (const field of importer.fields) {
      html += `<div style="margin-bottom:14px;">`;
      html += `<label style="display:block;margin-bottom:5px;color:#aaa;font-size:0.85rem;">${field.label}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${field.key}" accept="${field.accept}" style="color:#e0e0e0;width:100%;" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${field.key}" style="${inputStyle}">`;
        for (const opt of field.options) {
          html += `<option value="${opt}"${opt === field.default ? " selected" : ""}>${opt}</option>`;
        }
        html += `</select>`;
      } else if (field.field_type === "folder") {
        html += `<div style="display:flex;gap:8px;align-items:center;">`;
        html += `<input type="text" name="${field.key}" placeholder="${field.description}" style="${inputStyle}flex:1;" data-folder-input="true" ${field.required ? "required" : ""}>`;
        html += `<button type="button" data-browse-btn="true" style="padding:8px 14px;background:#252940;border:1px solid #2a2d3a;border-radius:4px;color:#aaa;cursor:pointer;white-space:nowrap;">Browse…</button>`;
        html += `</div>`;
        html += `<input type="file" data-folder-picker="true" webkitdirectory style="display:none;">`;
      } else {
        const itype = field.field_type === "url" ? "url" : "text";
        html += `<input type="${itype}" name="${field.key}" value="${field.default}" placeholder="${field.description}" style="${inputStyle}" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div style="margin-top:4px;font-size:0.75rem;color:#666;">${field.description}</div>`;
      }
      html += `</div>`;
    }
    html += `<button type="submit" style="width:100%;padding:10px;background:#7c8aff;border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.9rem;">Import</button>`;
    html += `</form></div>`;

    extendedImporterForm.innerHTML = html;
    extendedImporterForm.style.display = "block";

    // Wire up folder browse buttons
    const browseBtn = extendedImporterForm.querySelector("[data-browse-btn]");
    const folderPicker = extendedImporterForm.querySelector("[data-folder-picker]");
    const folderTextInput = extendedImporterForm.querySelector("[data-folder-input]");
    if (browseBtn && folderPicker && folderTextInput) {
      browseBtn.addEventListener("click", () => folderPicker.click());
      folderPicker.addEventListener("change", () => {
        if (folderPicker.files.length > 0) {
          // webkitRelativePath is "folderName/sub/file" — top segment is the folder name
          const topFolder = folderPicker.files[0].webkitRelativePath.split("/")[0];
          if (!folderTextInput.value) {
            folderTextInput.placeholder = `Selected: ${topFolder} — enter full path below`;
          }
        }
      });
    }

    document.getElementById("ext-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};
      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = {};
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }
      startProgressPolling();
      try {
        const res = await fetch(`/api/dataset/import/${importer.name}`, { method: "POST", headers, body });
        if (!res.ok) {
          const err = await res.json();
          progressMessage.textContent = `Error: ${err.error}`;
          progressMessage.style.color = "#f44336";
          stopProgressPolling();
        }
      } catch (err) {
        progressMessage.textContent = `Error: ${err.message}`;
        progressMessage.style.color = "#f44336";
        stopProgressPolling();
      }
    });
  }

  loadExtendedImporters();

  backButton.addEventListener("click", () => {
    showWelcomeScreen();
  });

  // ---- Burger Menu ----

  // Pause/resume looping media when focus leaves the labeling interface
  function pauseActiveMedia() {
    const audio = document.getElementById("clip-audio");
    const video = document.getElementById("clip-video");
    window._mediaPausedForUI = false;
    if (audio && !audio.paused) { audio.pause(); window._mediaPausedForUI = true; }
    if (video && !video.paused) { video.pause(); window._mediaPausedForUI = true; }
  }

  function resumeActiveMedia() {
    if (!window._mediaPausedForUI) return;
    const audio = document.getElementById("clip-audio");
    const video = document.getElementById("clip-video");
    if (audio) audio.play().catch(() => {});
    if (video) video.play().catch(() => {});
    window._mediaPausedForUI = false;
  }

  // Toggle burger menu
  if (burgerBtn && burgerDropdown) {
    burgerBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      burgerDropdown.classList.toggle("show");
      if (burgerDropdown.classList.contains("show")) {
        pauseActiveMedia();
      } else {
        resumeActiveMedia();
      }
    });

    // Close burger menu when clicking outside
    document.addEventListener("click", (e) => {
      if (!burgerDropdown.contains(e.target) && !burgerBtn.contains(e.target)) {
        if (burgerDropdown.classList.contains("show")) {
          burgerDropdown.classList.remove("show");
          resumeActiveMedia();
        }
      }
    });
  }

  // Dataset export
  if (menuDatasetExport && burgerDropdown) {
    menuDatasetExport.addEventListener("click", () => {
      window.location.href = "/api/dataset/export";
      burgerDropdown.classList.remove("show");
    });
  }

  // Dataset change
  if (menuDatasetChange && burgerDropdown) {
    menuDatasetChange.addEventListener("click", () => {
      if (confirm("Changing the dataset will erase your current dataset. Continue?")) {
        fetch("/api/dataset/clear", { method: "POST" })
          .then(() => {
            showWelcomeScreen();
            clips = [];
            votes = { good: [], bad: [] };
            selected = null;
            datasetLoaded = false;
            burgerDropdown.classList.remove("show");
          });
      } else {
        burgerDropdown.classList.remove("show");
      }
    });
  }

  // Labels export
  if (menuLabelsExport && menuLabelsStatus && burgerDropdown) {
    menuLabelsExport.addEventListener("click", async () => {
      menuLabelsStatus.textContent = "";
      const res = await fetch("/api/labels/export");
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "labels.json";
      a.click();
      URL.revokeObjectURL(url);
      menuLabelsStatus.textContent = `Exported ${data.labels.length} labels`;
      setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
      burgerDropdown.classList.remove("show");
    });
  }

  // Labels import – open the label importer picker modal
  if (menuLabelsImport && burgerDropdown) {
    menuLabelsImport.addEventListener("click", async () => {
      burgerDropdown.classList.remove("show");
      await openLabelImporterModal();
    });
  }

  // Detector import
  if (menuDetectorImport && loadDetectorFile && burgerDropdown) {
    menuDetectorImport.addEventListener("click", () => {
      loadDetectorFile.click();
      burgerDropdown.classList.remove("show");
    });
  }

  // Detector export
  if (menuDetectorExport && menuDetectorStatus && burgerDropdown) {
    menuDetectorExport.addEventListener("click", async () => {
      menuDetectorStatus.textContent = "";
      if (votes.good.length === 0 || votes.bad.length === 0) {
        menuDetectorStatus.textContent = "Vote good & bad clips first";
        setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
        return;
      }
      menuDetectorStatus.textContent = "Exporting detector\u2026";
      const res = await fetch("/api/detector/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        menuDetectorStatus.textContent = "Export failed";
        setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
        return;
      }
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "detector.json";
      a.click();
      URL.revokeObjectURL(url);
      menuDetectorStatus.textContent = "Detector exported";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
      burgerDropdown.classList.remove("show");
    });
  }

  // ---- Favorite Detectors ----

  async function loadFavoriteDetectors() {
    const res = await fetch("/api/favorite-detectors");
    const data = await res.json();
    favoriteDetectors = data.detectors || [];
    updateFavoritesList();
  }

  function updateFavoritesList() {
    if (favoriteDetectors.length === 0) {
      favoritesList.innerHTML = '<p style="color: #888;">No favorite detectors saved yet.</p>';
      return;
    }

    const mediaIcons = { audio: "🔊", image: "🖼️", video: "🎬", paragraph: "📄" };
    favoritesList.innerHTML = favoriteDetectors.map(detector => {
      const icon = mediaIcons[detector.media_type] || "🔍";
      const created = detector.created_at
        ? new Date(detector.created_at * 1000).toLocaleDateString()
        : "";
      return `
      <div style="background: #2a2d3a; padding: 12px; margin-bottom: 8px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; gap: 12px;">
        <div style="flex: 1; min-width: 0;">
          <div style="font-weight: bold; color: #7c8aff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(detector.name)}</div>
          <div style="font-size: 0.8rem; color: #888; margin-top: 3px; display: flex; align-items: center; gap: 10px;">
            <span style="background: #1a1d27; border: 1px solid #3a3d50; border-radius: 3px; padding: 1px 6px; white-space: nowrap;">${icon} ${detector.media_type}</span>
            <span>threshold&nbsp;${detector.threshold.toFixed(2)}</span>
            ${created ? `<span>${created}</span>` : ""}
          </div>
        </div>
        <div style="display: flex; gap: 6px; flex-shrink: 0;">
          <button onclick="renameDetector('${escapeHtml(detector.name)}')" style="padding: 4px 10px; background: #3a3d50; color: #ccc; border: 1px solid #4a4d60; border-radius: 4px; cursor: pointer; font-size: 0.78rem;">Rename</button>
          <button onclick="deleteDetector('${escapeHtml(detector.name)}')" style="padding: 4px 10px; background: #c0392b; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.78rem;">Delete</button>
        </div>
      </div>`;
    }).join('');
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  window.renameDetector = async function(oldName) {
    const newName = prompt(`Rename detector "${oldName}" to:`, oldName);
    if (!newName || newName === oldName) return;

    const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(oldName)}/rename`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_name: newName }),
    });

    if (res.ok) {
      await loadFavoriteDetectors();
    } else {
      alert("Failed to rename detector. Name may already exist.");
    }
  };

  window.deleteDetector = async function(name) {
    if (!confirm(`Are you sure you want to delete detector "${name}"?`)) return;

    const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });

    if (res.ok) {
      await loadFavoriteDetectors();
    } else {
      alert("Failed to delete detector.");
    }
  };

  if (menuFavoritesManage) {
    menuFavoritesManage.addEventListener("click", async () => {
      await loadFavoriteDetectors();
      favoritesModal.classList.add("show");
      burgerDropdown.classList.remove("show");
    });
  }

  if (favoritesModalClose) {
    favoritesModalClose.addEventListener("click", () => {
      favoritesModal.classList.remove("show");
    });
  }

  if (menuFavoritesSave) {
    menuFavoritesSave.addEventListener("click", async () => {
      if (votes.good.length === 0 || votes.bad.length === 0) {
        alert("You need to vote on at least one good and one bad clip before saving a detector.");
        return;
      }

      const name = prompt("Enter a name for this detector:");
      if (!name) return;

      // Export the detector first
      const res = await fetch("/api/detector/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        alert("Failed to export detector.");
        return;
      }

      const detectorData = await res.json();

      // Determine media type from current clips
      const mediaType = clips.length > 0 ? clips[0].type : "audio";

      // Save as favorite
      const saveRes = await fetch("/api/favorite-detectors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name,
          media_type: mediaType,
          weights: detectorData.weights,
          threshold: detectorData.threshold,
        }),
      });

      if (saveRes.ok) {
        alert(`Detector "${name}" saved successfully!`);
        await loadFavoriteDetectors();
      } else {
        alert("Failed to save detector.");
      }

      burgerDropdown.classList.remove("show");
    });
  }

  // ---- Add Detector panel inside Manage Favorites modal ----

  function setFavAddStatus(msg, color) {
    if (favAddStatus) {
      favAddStatus.textContent = msg;
      favAddStatus.style.color = color || "#aaa";
    }
  }

  // Add from current votes (train a new detector from labelled clips)
  if (favAddFromVotesBtn) {
    favAddFromVotesBtn.addEventListener("click", async () => {
      if (votes.good.length === 0 || votes.bad.length === 0) {
        setFavAddStatus("Need at least one good and one bad vote first.", "#f44336");
        return;
      }
      const name = favAddName ? favAddName.value.trim() : "";
      if (!name) {
        setFavAddStatus("Enter a name first.", "#f44336");
        return;
      }
      setFavAddStatus("Training detector\u2026", "#aaa");

      const exportRes = await fetch("/api/detector/export", { method: "POST" });
      if (!exportRes.ok) {
        setFavAddStatus("Failed to train detector.", "#f44336");
        return;
      }
      const detectorData = await exportRes.json();
      const mediaType = clips.length > 0 ? clips[0].type : "audio";

      const saveRes = await fetch("/api/favorite-detectors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          media_type: mediaType,
          weights: detectorData.weights,
          threshold: detectorData.threshold,
        }),
      });

      if (saveRes.ok) {
        setFavAddStatus(`Detector \u201c${name}\u201d saved (${mediaType}).`, "#4caf50");
        if (favAddName) favAddName.value = "";
        await loadFavoriteDetectors();
      } else {
        setFavAddStatus("Failed to save detector.", "#f44336");
      }
    });
  }

  // Add from a detector JSON file (same format as Load Sort / detector export)
  if (favAddFromDetectorBtn) {
    favAddFromDetectorBtn.addEventListener("click", () => {
      if (favDetectorFileInput) favDetectorFileInput.click();
    });
  }

  if (favDetectorFileInput) {
    favDetectorFileInput.addEventListener("change", async () => {
      const file = favDetectorFileInput.files[0];
      if (!file) return;
      const defaultName = file.name.replace(/\.[^/.]+$/, "");
      const detectorName = (favAddName && favAddName.value.trim()) || defaultName;

      setFavAddStatus("Importing detector\u2026", "#aaa");

      const formData = new FormData();
      formData.append("file", file);
      formData.append("name", detectorName);

      const res = await fetch("/api/favorite-detectors/import-pkl", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setFavAddStatus(`Imported \u201c${data.name}\u201d (${data.media_type}).`, "#4caf50");
        if (favAddName) favAddName.value = "";
        await loadFavoriteDetectors();
      } else {
        const err = await res.json().catch(() => ({}));
        setFavAddStatus(`Error: ${err.error || "Import failed"}`, "#f44336");
      }
      favDetectorFileInput.value = "";
    });
  }

  // Add from a label file (JSON with paths + labels; trains a new detector)
  if (favAddFromLabelsBtn) {
    favAddFromLabelsBtn.addEventListener("click", () => {
      if (favLabelsFileInput) favLabelsFileInput.click();
    });
  }

  if (favLabelsFileInput) {
    favLabelsFileInput.addEventListener("change", async () => {
      const file = favLabelsFileInput.files[0];
      if (!file) return;
      const defaultName = file.name.replace(/\.[^/.]+$/, "");
      const detectorName = (favAddName && favAddName.value.trim()) || defaultName;

      setFavAddStatus("Training from label file\u2026", "#aaa");

      const formData = new FormData();
      formData.append("file", file);
      formData.append("name", detectorName);

      const res = await fetch("/api/favorite-detectors/import-labels", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setFavAddStatus(
          `Trained \u201c${data.name}\u201d (${data.media_type}, ${data.loaded} files).`,
          "#4caf50"
        );
        if (favAddName) favAddName.value = "";
        await loadFavoriteDetectors();
      } else {
        const err = await res.json().catch(() => ({}));
        setFavAddStatus(`Error: ${err.error || "Training failed"}`, "#f44336");
      }
      favLabelsFileInput.value = "";
    });
  }

  if (menuFavoritesAutodetect) {
    menuFavoritesAutodetect.addEventListener("click", async () => {
      if (clips.length === 0) {
        alert("No dataset loaded. Please load a dataset first.");
        return;
      }

      burgerDropdown.classList.remove("show");

      // Show progress modal
      autodetectProgressModal.classList.add("show");
      autodetectProgressText.textContent = "Running auto-detect...";
      autodetectProgressBar.style.width = "0%";

      // Simulate progress (since we don't have real-time progress from backend)
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 5;
        if (progress > 90) progress = 90;
        autodetectProgressBar.style.width = `${progress}%`;
      }, 200);

      // Run auto-detect
      const res = await fetch("/api/auto-detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      clearInterval(progressInterval);
      autodetectProgressBar.style.width = "100%";

      setTimeout(() => {
        autodetectProgressModal.classList.remove("show");

        if (!res.ok) {
          alert("Auto-detect failed. Make sure you have saved some favorite detectors for this media type.");
          return;
        }

        res.json().then(data => {
          displayAutodetectResults(data);
        });
      }, 500);
    });
  }

  async function runAutoDetectAfterLoad() {
    if (clips.length === 0) {
      return;
    }

    // Show progress modal
    autodetectProgressModal.classList.add("show");
    autodetectProgressText.textContent = "Running auto-detect...";
    autodetectProgressBar.style.width = "0%";

    // Simulate progress (since we don't have real-time progress from backend)
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 5;
      if (progress > 90) progress = 90;
      autodetectProgressBar.style.width = `${progress}%`;
    }, 200);

    // Run auto-detect
    const res = await fetch("/api/auto-detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    clearInterval(progressInterval);
    autodetectProgressBar.style.width = "100%";

    setTimeout(() => {
      autodetectProgressModal.classList.remove("show");

      if (!res.ok) {
        alert("Auto-detect failed. Make sure you have saved some favorite detectors for this media type.");
        return;
      }

      res.json().then(data => {
        displayAutodetectResults(data);
      });
    }, 500);
  }

  function displayAutodetectResults(data) {
    // Display summary
    autodetectSummary.innerHTML = `
      <p style="color: #e0e0e0;">
        <strong>Media Type:</strong> ${data.media_type}<br>
        <strong>Detectors Run:</strong> ${data.detectors_run}<br>
        <strong>Total Positive Hits:</strong> ${Object.values(data.results).reduce((sum, r) => sum + r.total_hits, 0)}
      </p>
    `;

    // Display results by detector
    autodetectResults.innerHTML = Object.values(data.results).map(result => `
      <div style="background: #2a2d3a; padding: 16px; margin-bottom: 16px; border-radius: 4px;">
        <h3 style="color: #7c8aff; margin-top: 0;">${escapeHtml(result.detector_name)}</h3>
        <p style="color: #aaa; font-size: 0.9rem;">Threshold: ${result.threshold} | Positive Hits: ${result.total_hits}</p>
        ${result.hits.length === 0 ? '<p style="color: #888;">No positive hits found.</p>' : ''}
        ${result.hits.slice(0, 10).map(hit => `
          <div style="background: #1a1d27; padding: 8px; margin-bottom: 4px; border-radius: 4px; display: flex; justify-content: space-between;">
            <span style="color: #e0e0e0;">Clip #${hit.id}: ${hit.filename || 'N/A'}</span>
            <span style="color: #7c8aff; font-weight: bold;">Score: ${hit.score}</span>
          </div>
        `).join('')}
        ${result.hits.length > 10 ? `<p style="color: #888; font-size: 0.85rem;">...and ${result.hits.length - 10} more</p>` : ''}
      </div>
    `).join('');

    // Store results for copying
    window.autodetectResultsData = data;

    // Show modal
    autodetectModal.classList.add("show");
  }

  if (autodetectModalClose) {
    autodetectModalClose.addEventListener("click", () => {
      autodetectModal.classList.remove("show");
    });
  }

  if (copyResultsBtn) {
    copyResultsBtn.addEventListener("click", () => {
      if (!window.autodetectResultsData) return;

      const data = window.autodetectResultsData;
      let text = `Auto-Detect Results\n`;
      text += `Media Type: ${data.media_type}\n`;
      text += `Detectors Run: ${data.detectors_run}\n\n`;

      for (const [detectorName, result] of Object.entries(data.results)) {
        text += `\n=== ${detectorName} ===\n`;
        text += `Threshold: ${result.threshold} | Positive Hits: ${result.total_hits}\n`;
        if (result.hits.length > 0) {
          result.hits.forEach(hit => {
            text += `  Clip #${hit.id}: ${hit.filename || 'N/A'} (Score: ${hit.score})\n`;
          });
        } else {
          text += `  No positive hits found.\n`;
        }
      }

      navigator.clipboard.writeText(text).then(() => {
        copyResultsBtn.textContent = "Copied!";
        setTimeout(() => {
          copyResultsBtn.textContent = "Copy Results to Clipboard";
        }, 2000);
      });
    });
  }

  // ---- Sort mode switching ----

  function updateSortModeAvailability() {
    const hasGoodAndBad = votes.good.length > 0 && votes.bad.length > 0;
    learnedRadio.disabled = !hasGoodAndBad;
    learnedRadio.parentElement.style.opacity = hasGoodAndBad ? "1" : "0.5";
    learnedRadio.parentElement.style.cursor = hasGoodAndBad ? "pointer" : "not-allowed";

    // Load radio is always enabled - selecting it prompts for detector file
    loadRadio.disabled = false;
    loadRadio.parentElement.style.opacity = "1";
    loadRadio.parentElement.style.cursor = "pointer";
  }

  document.querySelectorAll('input[name="sort-mode"]').forEach(radio => {
    radio.addEventListener("change", () => {
      // Validate selection
      if (radio.value === "learned" && (votes.good.length === 0 || votes.bad.length === 0)) {
        sortStatus.textContent = "Vote good & bad clips first";
        // Revert to text mode
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        return;
      }
      if (radio.value === "load") {
        // Immediately prompt to select a detector file
        sortMode = radio.value;
        textSortWrap.style.display = "none";
        loadSortWrap.style.display = "";
        sortStatus.textContent = "";
        // Trigger file picker
        loadDetectorFile.click();
        return;
      }

      sortMode = radio.value;
      textSortWrap.style.display = sortMode === "text" ? "" : "none";
      loadSortWrap.style.display = sortMode === "load" ? "" : "none";
      sortStatus.textContent = "";

      if (sortMode === "text") {
        onTextSortInput();
      } else if (sortMode === "learned") {
        fetchLearnedSort(true);
      }
    });
  });

  // ---- Select mode switching ----

  document.querySelectorAll('input[name="select-mode"]').forEach(radio => {
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      selectMode = radio.value;
      const nextClip = findNextClip();
      if (nextClip) {
        selectClip(nextClip.id);
      }
    });
  });

  // ---- Inclusion slider ----

  async function updateInclusion(newInclusion) {
    inclusion = newInclusion;
    inclusionValue.textContent = inclusion;

    // Save to server
    await fetch("/api/inclusion", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inclusion }),
    });

    // Re-calculate threshold if in learned sort mode
    if (sortMode === "learned" && votes.good.length > 0 && votes.bad.length > 0) {
      await fetchLearnedSort();
    }
  }

  inclusionSlider.addEventListener("input", () => {
    updateInclusion(parseInt(inclusionSlider.value));
  });

  // ---- Text sort ----

  async function fetchTextSort(text) {
    showSortProgressWithPolling("Searching and sorting\u2026");
    try {
      const res = await fetch("/api/sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      sortOrder = data.results.map(e => ({ id: e.id, score: e.similarity }));
      threshold = data.threshold;
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderClipList();
      const nextClip = findNextClip();
      if (nextClip) selectClip(nextClip.id);
    } catch (error) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${error.message}`;
      console.error("Sort error:", error);
    }
  }

  function onTextSortInput() {
    clearTimeout(sortTimer);
    const text = textSortInput.value.trim();
    if (!text) {
      sortOrder = null;
      sortStatus.textContent = "";
      renderClipList();
      return;
    }
    sortTimer = setTimeout(() => fetchTextSort(text), 400);
  }

  textSortInput.addEventListener("input", onTextSortInput);

  // ---- Learned sort ----

  async function fetchLearnedSort(autoSelect = false) {
    showSortProgress("Training\u2026");
    try {
      const res = await fetch("/api/learned-sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        sortOrder = null;
        threshold = null;
        hideSortProgress();
        sortStatus.textContent = "Vote good & bad first";
        renderClipList();
        return;
      }
      const data = await res.json();
      sortOrder = data.results;  // [{id, score}, ...]
      threshold = data.threshold;
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderClipList();
      if (autoSelect) {
        const nextClip = findNextClip();
        if (nextClip) selectClip(nextClip.id);
      }
    } catch (error) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${error.message}`;
      console.error("Learned sort error:", error);
    }
  }

  // ---- Load detector sort ----

  async function fetchLoadedSort(autoSelect = false) {
    if (!loadedDetector) {
      sortStatus.textContent = "Load a detector first";
      return;
    }
    showSortProgress("Scoring with loaded detector\u2026");
    try {
      const res = await fetch("/api/detector-sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ detector: loadedDetector }),
      });
      if (!res.ok) {
        sortOrder = null;
        threshold = null;
        hideSortProgress();
        sortStatus.textContent = "Failed to score with detector";
        renderClipList();
        return;
      }
      const data = await res.json();
      sortOrder = data.results;  // [{id, score}, ...]
      threshold = data.threshold;
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderClipList();
      if (autoSelect) {
        const nextClip = findNextClip();
        if (nextClip) selectClip(nextClip.id);
      }
    } catch (error) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${error.message}`;
      console.error("Detector sort error:", error);
    }
  }

  // ---- Load detector file ----

  loadDetectorBtn.addEventListener("click", () => {
    loadDetectorFile.click();
  });

  loadDetectorFile.addEventListener("change", async () => {
    const file = loadDetectorFile.files[0];
    if (!file) {
      // User cancelled - revert to text mode if no detector loaded
      if (loadedDetector === null) {
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        sortMode = "text";
        textSortWrap.style.display = "";
        loadSortWrap.style.display = "none";
        sortStatus.textContent = "";
      }
      return;
    }
    sortStatus.textContent = "Loading detector\u2026";
    menuDetectorStatus.textContent = "Loading detector\u2026";
    const text = await file.text();
    try {
      loadedDetector = JSON.parse(text);
      sortStatus.textContent = "Detector loaded";
      menuDetectorStatus.textContent = "Detector loaded";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
      updateSortModeAvailability();
      // Ensure load mode is selected
      document.querySelector('input[name="sort-mode"][value="load"]').checked = true;
      sortMode = "load";
      loadSortWrap.style.display = "";
      textSortWrap.style.display = "none";
      fetchLoadedSort(true);
    } catch (e) {
      sortStatus.textContent = "Invalid detector file";
      menuDetectorStatus.textContent = "Invalid detector file";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
      loadedDetector = null;
      updateSortModeAvailability();
      // Revert to text mode on error
      document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
      sortMode = "text";
      textSortWrap.style.display = "";
      loadSortWrap.style.display = "none";
    }
    loadDetectorFile.value = "";
  });

  // ---- Next Clip Selection ----

  function findNextClip() {
    // Determine the ordered list to walk and effective threshold
    let ordered = sortOrder;
    let effectiveThreshold = threshold;
    if (!ordered || ordered.length === 0) {
      // Null sort: use clips in their current (arbitrary) order
      ordered = clips.map(c => ({ id: c.id, score: 0 }));
      // Treat threshold as the bottom for Hard mode
      effectiveThreshold = -Infinity;
    }

    if (ordered.length === 0) {
      return null;
    }

    // Get unlabeled clips (not voted on)
    const unlabeled = ordered.filter(item => {
      return !votes.good.includes(item.id) && !votes.bad.includes(item.id);
    });

    if (unlabeled.length === 0) {
      return null;
    }

    let nextClip;
    if (selectMode === "top") {
      // Select highest scoring unlabeled clip (or first in order for null sort)
      nextClip = unlabeled[0];
    } else {
      // Select unlabeled clip closest to threshold by list position,
      // breaking ties by score distance
      if (effectiveThreshold === null) {
        return null;
      }
      // Find threshold index: first position where score drops below threshold
      let thresholdIdx = ordered.length;
      for (let i = 0; i < ordered.length; i++) {
        if (ordered[i].score < effectiveThreshold) {
          thresholdIdx = i;
          break;
        }
      }
      // Map clip id to its ordered index
      const idToIdx = {};
      ordered.forEach((item, idx) => { idToIdx[item.id] = idx; });

      let minIdxDist = Infinity;
      let minDist = Infinity;
      for (const item of unlabeled) {
        const idxDist = Math.abs(idToIdx[item.id] - thresholdIdx);
        const dist = Math.abs(item.score - effectiveThreshold);
        if (idxDist < minIdxDist || (idxDist === minIdxDist && dist < minDist)) {
          minIdxDist = idxDist;
          minDist = dist;
          nextClip = item;
        }
      }
    }

    return nextClip;
  }

  // ---- Rendering ----

  async function fetchClips() {
    const res = await fetch("/api/clips");
    clips = await res.json();
    renderClipList();
  }

  async function fetchVotes() {
    const res = await fetch("/api/votes");
    votes = await res.json();
    renderVotes();
    renderStripe();
    updateSortModeAvailability();
    if (selected) renderCenter();
  }

  async function fetchInclusion() {
    const res = await fetch("/api/inclusion");
    const data = await res.json();
    inclusion = data.inclusion;
    inclusionSlider.value = inclusion;
    inclusionValue.textContent = inclusion;
  }

  function renderClipList() {
    clipList.innerHTML = "";
    const scoreMap = {};
    if (sortOrder) {
      sortOrder.forEach(s => { scoreMap[s.id] = s.score; });
    }

    let ordered;
    if (sortOrder) {
      ordered = sortOrder.map(s => clips.find(c => c.id === s.id)).filter(Boolean);
    } else {
      ordered = clips;
    }

    let thresholdLineInserted = false;
    ordered.forEach(c => {
      // Insert threshold line before first clip whose score falls below threshold
      if (threshold !== null && !thresholdLineInserted && scoreMap[c.id] !== undefined && scoreMap[c.id] < threshold) {
        const line = document.createElement("div");
        line.className = "clip-threshold-line";
        clipList.appendChild(line);
        thresholdLineInserted = true;
      }

      const div = document.createElement("div");
      const isGood = votes.good.includes(c.id);
      const isBad = votes.bad.includes(c.id);
      let className = "clip-item";
      if (selected === c.id) className += " active";
      if (isGood) className += " labeled-good";
      if (isBad) className += " labeled-bad";
      div.className = className;
      let html = `<div style="font-weight: 500;">${c.filename || 'Clip #' + c.id}</div>`;
      if (scoreMap[c.id] !== undefined) {
        html += `<span class="sim">${(scoreMap[c.id] * 100).toFixed(1)}%</span>`;
      }
      let subInfo = [];
      if (c.frequency) {
        subInfo.push(`${c.frequency} Hz`);
      }
      if (c.category && c.category !== "unknown") {
        subInfo.push(c.category);
      }
      if (c.type === "audio" || c.type === "video") {
        subInfo.push(`${c.duration.toFixed(1)}s`);
      } else if (c.type === "image" && c.width && c.height) {
        subInfo.push(`${c.width}×${c.height}`);
      } else if (c.type === "paragraph" && c.word_count) {
        subInfo.push(`${c.word_count} words`);
      }
      html += `<div class="sub">${subInfo.join(' &middot; ')}</div>`;
      div.innerHTML = html;
      div.onclick = () => selectClip(c.id);
      clipList.appendChild(div);
    });

    renderStripe();
  }

  function selectClip(id) {
    selected = id;
    renderClipList();
    renderCenter();

    const activeItem = clipList.querySelector(".clip-item.active");
    if (activeItem) {
      activeItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  function renderCenter() {
    const c = clips.find(x => x.id === selected);
    if (!c) return;
    const isGood = votes.good.includes(c.id);
    const isBad = votes.bad.includes(c.id);
    center.className = "panel-center";

    const mediaType = c.type || "audio";
    let metaInfo = [];
    if (c.frequency) {
      metaInfo.push(`${c.frequency} Hz`);
    }
    if (c.category && c.category !== "unknown") {
      metaInfo.push(c.category);
    }
    if (mediaType === "audio" || mediaType === "video") {
      metaInfo.push(`${c.duration.toFixed(1)}s`);
    }
    if (mediaType === "image" && c.width && c.height) {
      metaInfo.push(`${c.width}×${c.height}`);
    }
    if (mediaType === "paragraph" && c.word_count) {
      metaInfo.push(`${c.word_count} words`);
    }
    metaInfo.push(`${(c.file_size / 1024).toFixed(1)} KB`);

    // Render media player based on media type
    let playerHTML = '';
    if (mediaType === "video") {
      playerHTML = `<video controls loop autoplay src="/api/clips/${c.id}/video" id="clip-video" style="width: 600px; max-height: 400px; border: 1px solid #2a2d3a; border-radius: 8px; background: #1a1d27;"></video>`;
    } else if (mediaType === "image") {
      playerHTML = `<div style="flex: 1; min-height: 0; width: 100%; display: flex; align-items: center; justify-content: center;"><img src="/api/clips/${c.id}/image" id="clip-image" style="max-width: 100%; max-height: 100%; object-fit: contain; border: 1px solid #2a2d3a; border-radius: 8px; background: #1a1d27;"></div>`;
    } else if (mediaType === "paragraph") {
      playerHTML = `
        <div id="clip-paragraph" style="max-width: 600px; max-height: 400px; overflow-y: auto; padding: 16px; border: 1px solid #2a2d3a; border-radius: 8px; background: #1a1d27; white-space: pre-wrap; line-height: 1.6; text-align: left;">
          Loading...
        </div>`;
    } else {
      // Audio/Sound
      playerHTML = `
        <canvas id="waveform-canvas" width="600" height="120"></canvas>
        <audio controls loop autoplay src="/api/clips/${c.id}/audio" id="clip-audio"></audio>`;
    }

    center.innerHTML = `
      <div class="meta">
        <h2>${c.filename || 'Clip #' + c.id}</h2>
        <p>${metaInfo.join(' &middot; ')}</p>
      </div>
      ${playerHTML}
      <div class="metadata-grid">
        ${c.frequency ? `
        <div class="metadata-item">
          <span class="metadata-label">Frequency</span>
          <span class="metadata-value">${c.frequency} Hz</span>
        </div>` : ''}
        ${c.category && c.category !== 'unknown' ? `
        <div class="metadata-item">
          <span class="metadata-label">Category</span>
          <span class="metadata-value">${c.category}</span>
        </div>` : ''}
        <div class="metadata-item">
          <span class="metadata-label">Media Type</span>
          <span class="metadata-value">${mediaType}</span>
        </div>
        ${(mediaType === 'audio' || mediaType === 'video') ? `
        <div class="metadata-item">
          <span class="metadata-label">Duration</span>
          <span class="metadata-value">${c.duration.toFixed(1)}s</span>
        </div>` : ''}
        ${(mediaType === 'image' && c.width && c.height) ? `
        <div class="metadata-item">
          <span class="metadata-label">Dimensions</span>
          <span class="metadata-value">${c.width}×${c.height}</span>
        </div>` : ''}
        ${(mediaType === 'paragraph' && c.word_count) ? `
        <div class="metadata-item">
          <span class="metadata-label">Word Count</span>
          <span class="metadata-value">${c.word_count}</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Characters</span>
          <span class="metadata-value">${c.character_count}</span>
        </div>` : ''}
        <div class="metadata-item">
          <span class="metadata-label">File Size</span>
          <span class="metadata-value">${(c.file_size / 1024).toFixed(1)} KB</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Filename</span>
          <span class="metadata-value">${c.filename || 'clip_' + c.id + '.wav'}</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">MD5</span>
          <span class="metadata-value metadata-md5">${c.md5}</span>
        </div>
      </div>
      <div class="vote-buttons">
        <button class="btn-bad${isBad ? " voted" : ""}" id="vote-bad">Bad</button>
        <button class="btn-good${isGood ? " voted" : ""}" id="vote-good">Good</button>
      </div>`;
    document.getElementById("vote-good").onclick = () => castVote(c.id, "good");
    document.getElementById("vote-bad").onclick = () => castVote(c.id, "bad");

    // Draw waveform only for audio clips
    if (mediaType === "audio") {
      drawWaveform(c.id);
      const audioEl = document.getElementById("clip-audio");
      if (audioEl) {
        audioEl.volume = audioVolume;
        audioEl.addEventListener("volumechange", () => {
          audioVolume = audioEl.volume;
        });
      }
    }

    // Load paragraph text content
    if (mediaType === "paragraph") {
      fetch(`/api/clips/${c.id}/paragraph`)
        .then(res => res.json())
        .then(data => {
          const paragraphDiv = document.getElementById("clip-paragraph");
          if (paragraphDiv) {
            paragraphDiv.textContent = data.content;
          }
        })
        .catch(err => {
          console.error("Error loading paragraph:", err);
        });
    }
  }

  async function castVote(id, vote) {
    await fetch(`/api/clips/${id}/vote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vote }),
    });
    await fetchVotes();
    if (sortMode === "learned") await fetchLearnedSort();

    // Auto-advance to next clip
    const nextClip = findNextClip();
    if (nextClip && nextClip.id !== selected) {
      selectClip(nextClip.id);
    }
  }

  async function drawWaveform(clipId) {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = "#1a1d27";
    ctx.fillRect(0, 0, width, height);

    try {
      // Fetch audio data
      const response = await fetch(`/api/clips/${clipId}/audio`);
      const arrayBuffer = await response.arrayBuffer();

      // Decode audio data
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // Get audio data from first channel
      const channelData = audioBuffer.getChannelData(0);
      const step = Math.ceil(channelData.length / width);
      const amp = height / 2;

      // Draw waveform
      ctx.strokeStyle = "#7c8aff";
      ctx.lineWidth = 1;
      ctx.beginPath();

      for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;

        for (let j = 0; j < step; j++) {
          const datum = channelData[i * step + j];
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }

        const yMin = (1 + min) * amp;
        const yMax = (1 + max) * amp;

        if (i === 0) {
          ctx.moveTo(i, yMin);
        }
        ctx.lineTo(i, yMin);
        ctx.lineTo(i, yMax);
      }

      ctx.stroke();

      // Draw center line
      ctx.strokeStyle = "#2a2d3a";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

    } catch (error) {
      console.error("Error drawing waveform:", error);
      // Draw error message
      ctx.fillStyle = "#f44336";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Unable to load waveform", width / 2, height / 2);
    }
  }

  function renderVotes() {
    goodList.innerHTML = "";
    badList.innerHTML = "";

    const sortedGood = [...votes.good].sort((a, b) => a - b);

    sortedGood.forEach(id => {
      const div = document.createElement("div");
      div.className = "vote-entry";
      div.textContent = `Clip #${id}`;
      div.onclick = () => selectClip(id);
      goodList.appendChild(div);
    });

    const sortedBad = [...votes.bad].sort((a, b) => a - b);

    sortedBad.forEach(id => {
      const div = document.createElement("div");
      div.className = "vote-entry";
      div.textContent = `Clip #${id}`;
      div.onclick = () => selectClip(id);
      badList.appendChild(div);
    });
  }

  function renderStripe() {
    stripeContainer.innerHTML = "";

    // Only show stripe when sorted
    if (!sortOrder || sortOrder.length === 0) {
      stripeOverview.style.display = "none";
      return;
    }

    stripeOverview.style.display = "block";
    const totalClips = sortOrder.length;

    // Add highlight element
    const highlight = document.createElement("div");
    highlight.className = "stripe-highlight";
    stripeContainer.appendChild(highlight);

    // Render dots for each labeled clip
    sortOrder.forEach((item, index) => {
      const isGood = votes.good.includes(item.id);
      const isBad = votes.bad.includes(item.id);
      const isSelected = item.id === selected;

      if (isGood || isBad) {
        const dot = document.createElement("div");
        dot.className = `stripe-dot ${isGood ? "good" : "bad"}`;
        dot.style.top = `${(index / totalClips) * 100}%`;
        dot.setAttribute("data-clip-id", item.id);
        dot.setAttribute("data-index", index);
        stripeContainer.appendChild(dot);
      }

      if (isSelected) {
        const dot = document.createElement("div");
        dot.className = "stripe-dot selected";
        dot.style.top = `${(index / totalClips) * 100}%`;
        stripeContainer.appendChild(dot);
      }
    });

    // Render threshold line if available
    if (threshold !== null) {
      // Find the index where score crosses threshold
      let thresholdIndex = 0;
      for (let i = 0; i < sortOrder.length; i++) {
        if (sortOrder[i].score < threshold) {
          thresholdIndex = i;
          break;
        }
      }

      const thresholdLine = document.createElement("div");
      thresholdLine.className = "stripe-threshold";
      thresholdLine.style.top = `${(thresholdIndex / totalClips) * 100}%`;
      stripeContainer.appendChild(thresholdLine);
    }

    updateStripeHighlight();
  }

  // ---- Stripe click handler ----

  stripeOverview.addEventListener("click", (e) => {
    if (!sortOrder || sortOrder.length === 0) return;

    const rect = stripeOverview.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const percentage = y / rect.height;
    const index = Math.floor(percentage * sortOrder.length);
    const clampedIndex = Math.max(0, Math.min(index, sortOrder.length - 1));

    if (sortOrder[clampedIndex]) {
      const clipId = sortOrder[clampedIndex].id;
      selectClip(clipId);

      // Scroll the clip into view
      const clipItems = clipList.querySelectorAll(".clip-item");
      if (clipItems[clampedIndex]) {
        clipItems[clampedIndex].scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  });

  // ---- Stripe highlight update ----

  function updateStripeHighlight() {
    const highlight = stripeContainer.querySelector(".stripe-highlight");
    if (!highlight) return;

    const scrollHeight = clipList.scrollHeight;
    const clientHeight = clipList.clientHeight;
    const scrollTop = clipList.scrollTop;

    const topPercent = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
    const heightPercent = scrollHeight > 0 ? (clientHeight / scrollHeight) * 100 : 100;

    highlight.style.top = `${topPercent}%`;
    highlight.style.height = `${heightPercent}%`;
  }

  clipList.addEventListener("scroll", updateStripeHighlight);
  window.addEventListener("resize", updateStripeHighlight);

  // ---- Label importer modal ----

  async function openLabelImporterModal() {
    // Fetch available importers
    let importers = [];
    try {
      const res = await fetch("/api/label-importers");
      if (res.ok) importers = await res.json();
    } catch (_) { /* ignore */ }

    // Reset to picker view
    labelImporterFormDiv.style.display = "none";
    labelImporterFormDiv.innerHTML = "";
    labelImporterBack.style.display = "none";
    labelImporterList.style.display = "";

    if (importers.length === 0) {
      labelImporterList.innerHTML = '<p style="color:#888;">No label importers available.</p>';
    } else {
      labelImporterList.innerHTML = importers.map(imp => `
        <div class="label-importer-option" data-name="${escapeHtml(imp.name)}" style="
          background:#2a2d3a; border:1px solid #3a3d50; border-radius:6px;
          padding:12px 16px; margin-bottom:10px; cursor:pointer;
          display:flex; align-items:center; gap:12px;">
          <span style="font-size:1.5rem;">${escapeHtml(imp.icon || '🏷️')}</span>
          <div>
            <div style="font-weight:bold; color:#e0e0e0;">${escapeHtml(imp.display_name)}</div>
            <div style="font-size:0.8rem; color:#888; margin-top:2px;">${escapeHtml(imp.description)}</div>
          </div>
        </div>
      `).join("");

      labelImporterList.querySelectorAll(".label-importer-option").forEach(el => {
        const name = el.dataset.name;
        const imp = importers.find(i => i.name === name);
        el.addEventListener("click", () => showLabelImporterForm(imp));
      });
    }

    labelImporterModal.classList.add("show");
  }

  function showLabelImporterForm(importer) {
    labelImporterList.style.display = "none";
    labelImporterBack.style.display = "inline-block";

    const inputStyle = "width:100%;padding:8px;background:#252940;border:1px solid #2a2d3a;border-radius:4px;color:#e0e0e0;box-sizing:border-box;";
    let html = `<h3 style="margin-bottom:14px;color:#e0e0e0;">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="label-imp-form">`;
    for (const field of importer.fields) {
      html += `<div style="margin-bottom:14px;">`;
      html += `<label style="display:block;margin-bottom:5px;color:#aaa;font-size:0.85rem;">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" style="color:#e0e0e0;width:100%;" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" style="${inputStyle}">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        const placeholder = escapeHtml(field.placeholder || field.description);
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" style="${inputStyle}" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div style="margin-top:4px;font-size:0.75rem;color:#666;">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="label-imp-status" style="min-height:1.4em;font-size:0.85rem;color:#888;margin-bottom:10px;"></div>`;
    html += `<button type="submit" style="width:100%;padding:10px;background:#7c8aff;border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.9rem;">Import</button>`;
    html += `</form>`;

    labelImporterFormDiv.innerHTML = html;
    labelImporterFormDiv.style.display = "block";

    const statusEl = labelImporterFormDiv.querySelector("#label-imp-status");

    labelImporterFormDiv.querySelector("#label-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      statusEl.textContent = "Importing\u2026";
      statusEl.style.color = "#888";

      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};

      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = {};
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }

      try {
        const res = await fetch(`/api/label-importers/import/${encodeURIComponent(importer.name)}`, {
          method: "POST", headers, body,
        });
        const result = await res.json();
        if (res.ok) {
          statusEl.textContent = `Applied ${result.applied}, skipped ${result.skipped}.`;
          statusEl.style.color = "#4caf50";
          await fetchVotes();
          setTimeout(() => {
            labelImporterModal.classList.remove("show");
            menuLabelsStatus.textContent = `Applied ${result.applied} label(s)`;
            setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
          }, 1500);
        } else {
          statusEl.textContent = result.error || "Import failed";
          statusEl.style.color = "#f44336";
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "#f44336";
      }
    });
  }

  if (labelImporterModalClose) {
    labelImporterModalClose.addEventListener("click", () => {
      labelImporterModal.classList.remove("show");
    });
  }

  if (labelImporterBack) {
    labelImporterBack.addEventListener("click", () => {
      labelImporterFormDiv.style.display = "none";
      labelImporterFormDiv.innerHTML = "";
      labelImporterBack.style.display = "none";
      labelImporterList.style.display = "";
    });
  }

  // ---- Progress tracking ----

  const checkProgressBtn = document.getElementById("check-progress-btn");
  const progressModal = document.getElementById("progress-modal");
  const modalClose = document.getElementById("modal-close");
  const goodCountSpan = document.getElementById("good-count");
  const badCountSpan = document.getElementById("bad-count");
  const labelingAnalysisModal = document.getElementById("labeling-analysis-modal");
  const labelingAnalysisBar = document.getElementById("labeling-analysis-bar");
  const labelingAnalysisText = document.getElementById("labeling-analysis-text");
  const labelingAnalysisPct = document.getElementById("labeling-analysis-pct");

  // Update label counts and schedule an indicator refresh
  function updateLabelCounts() {
    goodCountSpan.textContent = `(${votes.good.length})`;
    badCountSpan.textContent = `(${votes.bad.length})`;
    scheduleLabelingStatusUpdate();
  }

  // ---- Labeling status indicator ----

  let _statusTimer = null;

  function scheduleLabelingStatusUpdate() {
    clearTimeout(_statusTimer);
    _statusTimer = setTimeout(fetchLabelingStatus, 1200);
  }

  async function fetchLabelingStatus() {
    try {
      const res = await fetch("/api/labeling-status");
      if (!res.ok) return;
      const data = await res.json();
      if (data.error) return;
      applyLabelingStatus(data);
    } catch (_) {
      // Silently ignore — the indicator will just stay in its last state
    }
  }

  function applyLabelingStatus(data) {
    const btn = checkProgressBtn;
    const subtext = document.getElementById("indicator-subtext");
    btn.dataset.status = data.status;
    if (data.status === "red") {
      subtext.textContent = `${data.good_count}g / ${data.bad_count}b`;
    } else if (data.status === "yellow") {
      subtext.textContent = "Continue";
    } else if (data.status === "green") {
      subtext.textContent = "ready to stop";
    } else {
      subtext.textContent = "";
    }
  }

  checkProgressBtn.addEventListener("click", async () => {
    if (votes.good.length === 0 || votes.bad.length === 0) {
      alert("Need at least one good and one bad vote to check progress");
      return;
    }

    // Pause any active audio or video playback
    pauseActiveMedia();

    // Show loading progress modal
    labelingAnalysisBar.style.width = "0%";
    labelingAnalysisPct.textContent = "0%";
    labelingAnalysisText.textContent = "Training models over label history…";
    labelingAnalysisModal.classList.add("show");

    // Simulate progress while waiting for the backend
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 3;
      if (progress > 90) progress = 90;
      labelingAnalysisBar.style.width = `${progress}%`;
      labelingAnalysisPct.textContent = `${progress}%`;
    }, 250);

    checkProgressBtn.disabled = true;
    checkProgressBtn.querySelector(".indicator-label").textContent = "Analyzing…";

    try {
      const res = await fetch("/api/labeling-progress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      clearInterval(progressInterval);
      labelingAnalysisBar.style.width = "100%";
      labelingAnalysisPct.textContent = "100%";
      labelingAnalysisText.textContent = "Done!";

      await new Promise(r => setTimeout(r, 350));
      labelingAnalysisModal.classList.remove("show");

      if (!res.ok) {
        const error = await res.json();
        alert(error.error || "Failed to analyze progress");
        return;
      }

      const data = await res.json();
      displayProgressResults(data);
      progressModal.classList.add("show");
    } catch (e) {
      clearInterval(progressInterval);
      labelingAnalysisModal.classList.remove("show");
      alert("Error analyzing progress: " + e.message);
    } finally {
      checkProgressBtn.disabled = false;
      checkProgressBtn.querySelector(".indicator-label").textContent = "Progress";
      fetchLabelingStatus();
      // If the progress modal didn't open (error path), resume media now
      if (!progressModal.classList.contains("show")) {
        resumeActiveMedia();
      }
    }
  });

  modalClose.addEventListener("click", () => {
    progressModal.classList.remove("show");
    resumeActiveMedia();
  });

  progressModal.addEventListener("click", (e) => {
    if (e.target === progressModal) {
      progressModal.classList.remove("show");
      resumeActiveMedia();
    }
  });

  function displayProgressResults(data) {
    // Update summary stats
    document.getElementById("stat-total-labels").textContent = data.total_labels;
    document.getElementById("stat-total-clips").textContent = data.total_clips;

    // Render error cost chart
    renderErrorCostChart(data.error_cost_over_time);

    // Render stability chart
    renderStabilityChart(data.stability_over_time);

    // Generate recommendation
    const recommendation = generateRecommendation(data);
    document.getElementById("recommendation-text").textContent = recommendation;
  }

  function renderErrorCostChart(errorCostData) {
    const canvas = document.getElementById("error-cost-chart");
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!errorCostData || errorCostData.length === 0) {
      ctx.fillStyle = "#888";
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    // Extract data
    const numLabels = errorCostData.map(d => d.num_labels);
    const errorCosts = errorCostData.map(d => d.error_cost);

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Scales
    const maxLabels = Math.max(...numLabels);
    const maxCost = Math.max(...errorCosts);
    const minCost = Math.min(...errorCosts);

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - ((val - minCost) / (maxCost - minCost || 1)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = "#2a2d3a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = "#1e2030";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = "#7c8aff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < errorCostData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(errorCosts[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = "#7c8aff";
    for (let i = 0; i < errorCostData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(errorCosts[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Error Cost", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxCost.toFixed(2), padding.left - 5, padding.top + 5);
    ctx.fillText(minCost.toFixed(2), padding.left - 5, padding.top + chartHeight + 5);
  }

  function renderStabilityChart(stabilityData) {
    const canvas = document.getElementById("stability-chart");
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!stabilityData || stabilityData.length === 0) {
      ctx.fillStyle = "#888";
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    // Extract data (skip first entry since it has no previous to compare)
    const dataToPlot = stabilityData.slice(1);
    if (dataToPlot.length === 0) {
      ctx.fillStyle = "#888";
      ctx.font = "14px sans-serif";
      ctx.fillText("Need more labels for stability analysis", 20, canvas.height / 2);
      return;
    }

    const numLabels = dataToPlot.map(d => d.num_labels);
    const numFlips = dataToPlot.map(d => d.num_flips);

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Scales
    const maxLabels = Math.max(...numLabels);
    const maxFlips = Math.max(...numFlips, 1);

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - (val / maxFlips) * chartHeight;

    // Draw axes
    ctx.strokeStyle = "#2a2d3a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = "#1e2030";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = "#4caf50";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < dataToPlot.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(numFlips[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = "#4caf50";
    for (let i = 0; i < dataToPlot.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(numFlips[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Prediction Flips", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxFlips.toString(), padding.left - 5, padding.top + 5);
    ctx.fillText("0", padding.left - 5, padding.top + chartHeight + 5);
  }

  function generateRecommendation(data) {
    const errorCostData = data.error_cost_over_time;
    const stabilityData = data.stability_over_time;

    if (errorCostData.length < 5) {
      return "Keep labeling! You need more labels to assess progress (at least 5 training steps).";
    }

    // Check if error cost has plateaued (last 30% of data)
    const last30PercentCount = Math.max(3, Math.floor(errorCostData.length * 0.3));
    const recentErrorCosts = errorCostData.slice(-last30PercentCount).map(d => d.error_cost);
    const errorCostRange = Math.max(...recentErrorCosts) - Math.min(...recentErrorCosts);
    const avgErrorCost = recentErrorCosts.reduce((a, b) => a + b, 0) / recentErrorCosts.length;
    const errorCostStability = errorCostRange / (avgErrorCost || 1);

    // Check if prediction flips have decreased
    const recentStability = stabilityData.slice(-last30PercentCount);
    const recentFlips = recentStability.map(d => d.num_flips);
    const avgRecentFlips = recentFlips.reduce((a, b) => a + b, 0) / (recentFlips.length || 1);

    let recommendation = "";

    if (errorCostStability < 0.1 && avgRecentFlips < 5) {
      recommendation = "🛑 STOP LABELING: Both error cost and predictions have stabilized. Additional labels are unlikely to improve the model significantly. You've reached a good stopping point!";
    } else if (errorCostStability < 0.15) {
      recommendation = "⚠️ CONSIDER STOPPING: Error cost has mostly plateaued. You may be approaching diminishing returns. Consider stopping or labeling a few more diverse examples.";
    } else if (avgRecentFlips < 3) {
      recommendation = "⚠️ PREDICTIONS STABLE: Predictions on unlabeled clips aren't changing much. The model's decisions are solidifying. Consider whether additional labels will help.";
    } else {
      recommendation = "✅ KEEP LABELING: The model is still learning! Both error cost and predictions are changing, indicating that new labels are improving the model.";
    }

    return recommendation;
  }

  // Modify fetchVotes to update label counts
  const originalFetchVotes = fetchVotes;
  fetchVotes = async function() {
    await originalFetchVotes();
    updateLabelCounts();
  };

  // Initialize
  checkDatasetStatus().then(async () => {
    if (datasetLoaded) {
      await fetchClips();
      await fetchVotes();
      if (clips.length > 0 && !selected) {
        selectClip(clips[0].id);
      }
    }
  });
  fetchInclusion();
  updateLabelCounts();
  loadFavoriteDetectors();
  fetchLabelingStatus();
})();
