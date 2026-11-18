document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("[data-review-form]");
  if (!form) return;
  const editors = [];
  form.querySelectorAll("[data-editor]").forEach(section => {
    const input = section.querySelector("[data-edit-input]");
    const view = section.querySelector("[data-edit-view]");
    const reason = section.querySelector("[data-edit-reason]");
    const reasonBox = section.querySelector("[data-reason-box]");
    const error = section.querySelector("[data-edit-error]");
    const edit = section.querySelector("[data-edit]");
    const save = section.querySelector("[data-edit-save]");
    const cancel = section.querySelector("[data-edit-cancel]");
    let savedValue = input.value;
    let savedReason = reason.value;
    const hasChanges = () => input.value.trim() !== section.dataset.original.trim();
    const valid = () => {
      const missing = !input.value.trim() || (hasChanges() && !reason.value.trim());
      error.hidden = !missing;
      reason.setAttribute("aria-invalid", String(missing));
      return !missing;
    };
    const setEditing = active => {
      input.hidden = !active;
      reasonBox.hidden = !active;
      view.hidden = active;
      if (edit) {
        edit.hidden = active;
        save.hidden = !active;
        cancel.hidden = !active;
      }
    };
    const finish = () => {
      savedValue = input.value;
      savedReason = reason.value;
      view.textContent = input.value;
      setEditing(false);
    };
    setEditing(section.dataset.editing === "yes" && section.dataset.locked !== "yes");
    if (edit) {
      edit.addEventListener("click", () => { setEditing(true); input.focus(); });
      save.addEventListener("click", () => {
        if (!valid()) { reason.focus(); return; }
        finish();
        edit.focus();
      });
      cancel.addEventListener("click", () => {
        input.value = savedValue;
        reason.value = savedReason;
        error.hidden = true;
        reason.removeAttribute("aria-invalid");
        finish();
        edit.focus();
        form.dispatchEvent(new Event("input", { bubbles: true }));
      });
    }
    editors.push({ valid, finish, open: () => { setEditing(true); reason.focus(); } });
  });
  form.addEventListener("submit", event => {
    if (event.submitter?.value === "confirm") {
      const invalid = editors.filter(editor => !editor.valid());
      if (invalid.length) {
        event.preventDefault();
        invalid[0].open();
        return;
      }
    }
    editors.forEach(editor => editor.finish());
  });

  const workspace = document.querySelector("[data-review-workspace]");
  const panel = document.querySelector("[data-documents-panel]");
  if (!panel) return;
  const body = panel.querySelector("[data-documents-body]");
  const close = panel.querySelector("[data-documents-close]");
  const expand = panel.querySelector("[data-documents-expand]");
  const search = panel.querySelector("[data-document-search]");
  const previous = panel.querySelector("[data-find-previous]");
  const next = panel.querySelector("[data-find-next]");
  const status = panel.querySelector("[data-search-status]");
  const narrow = window.matchMedia("(max-width: 1000px)");
  let opener;
  let savedScroll = 0;
  let loaded = false;
  let loading = false;
  let results = [];
  let current = -1;
  let timer;
  const fold = value => value.normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase();
  const showStatus = () => {
    previous.disabled = next.disabled = !results.length;
    if (!search.value.trim()) status.textContent = "Nhập từ khóa hoặc cụm từ, có thể gõ không dấu.";
    else if (!results.length) status.textContent = "Không tìm thấy kết quả.";
    else {
      const mark = results[current][0];
      const documentTitle = mark.closest(".full-document").querySelector("summary").textContent.trim();
      const category = mark.closest(".document-category");
      status.textContent = (current + 1) + " / " + results.length + " · " + documentTitle +
        (category ? " · " + category.querySelector("summary").textContent.trim() : "");
    }
  };
  const goTo = index => {
    if (!results.length) return;
    current = (index + results.length) % results.length;
    body.querySelectorAll("mark.is-current").forEach(mark => mark.classList.remove("is-current"));
    const matches = results[current];
    matches.forEach(mark => {
      mark.classList.add("is-current");
      let parent = mark.parentElement;
      while (parent && parent !== body) {
        if (parent.tagName === "DETAILS") parent.open = true;
        parent = parent.parentElement;
      }
    });
    requestAnimationFrame(() => {
      const mark = matches[0];
      if (!mark.isConnected) return;
      const position = mark.getBoundingClientRect();
      body.scrollTop += position.top - body.getBoundingClientRect().top - body.clientHeight / 3;
      const table = mark.closest(".source-table");
      if (table) table.scrollLeft += position.left - table.getBoundingClientRect().left - table.clientWidth / 3;
    });
    showStatus();
  };
  const runSearch = () => {
    clearTimeout(timer);
    timer = null;
    body.querySelectorAll("mark[data-match]").forEach(mark => mark.replaceWith(document.createTextNode(mark.textContent)));
    body.normalize();
    results = [];
    current = -1;
    const query = fold(search.value.trim());
    if (!query) { showStatus(); return; }
    const selector = "summary, h3, p, li, td, th, pre";
    const blocks = [...body.querySelectorAll(selector)].filter(block => !block.querySelector(selector));
    blocks.forEach(block => {
      const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      const nodes = [];
      let combined = "";
      let node;
      while ((node = walker.nextNode())) {
        const mapping = [];
        let normalized = "";
        let offset = 0;
        for (const character of node.data) {
          const value = fold(character);
          for (let i = 0; i < value.length; i++) mapping.push({ start: offset, end: offset + character.length });
          normalized += value;
          offset += character.length;
        }
        nodes.push({ node, mapping, start: combined.length, end: combined.length + normalized.length, segments: [] });
        combined += normalized;
      }
      let start = combined.indexOf(query);
      while (start !== -1) {
        const resultIndex = results.length;
        results.push([]);
        const end = start + query.length;
        nodes.forEach(entry => {
          const left = Math.max(start, entry.start);
          const right = Math.min(end, entry.end);
          if (left < right) entry.segments.push({
            start: entry.mapping[left - entry.start].start,
            end: entry.mapping[right - entry.start - 1].end,
            resultIndex,
          });
        });
        start = combined.indexOf(query, end);
      }
      nodes.forEach(entry => {
        if (!entry.segments.length) return;
        const fragment = document.createDocumentFragment();
        let offset = 0;
        entry.segments.forEach(segment => {
          fragment.append(document.createTextNode(entry.node.data.slice(offset, segment.start)));
          const mark = document.createElement("mark");
          mark.dataset.match = String(segment.resultIndex);
          mark.textContent = entry.node.data.slice(segment.start, segment.end);
          results[segment.resultIndex].push(mark);
          fragment.append(mark);
          offset = segment.end;
        });
        fragment.append(document.createTextNode(entry.node.data.slice(offset)));
        entry.node.replaceWith(fragment);
      });
    });
    if (results.length) goTo(0);
    else showStatus();
  };
  search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(runSearch, 180); });
  search.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (timer) runSearch();
      else goTo(current + (event.shiftKey ? -1 : 1));
    }
    if (event.key === "Escape") {
      event.stopPropagation();
      search.value = "";
      runSearch();
    }
  });
  previous.addEventListener("click", () => { if (timer) runSearch(); else goTo(current - 1); });
  next.addEventListener("click", () => { if (timer) runSearch(); else goTo(current + 1); });
  showStatus();
  const setModal = () => {
    if (!panel.hidden && narrow.matches) {
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-modal", "true");
      workspace.querySelector(".review-reading").inert = true;
    } else {
      panel.removeAttribute("role");
      panel.removeAttribute("aria-modal");
      workspace.querySelector(".review-reading").inert = false;
    }
  };
  const load = async () => {
    if (loaded || loading) return;
    loading = true;
    body.textContent = "Đang tải tài liệu…";
    try {
      const response = await fetch(panel.dataset.url, { credentials: "same-origin" });
      if (!response.ok || response.redirected) throw new Error("Document request failed");
      body.innerHTML = await response.text();
      loaded = true;
      body.querySelectorAll(".full-document").forEach(doc => { doc.open = doc.dataset.docId === opener?.dataset.docId; });
      if (search.value.trim()) runSearch();
    } catch {
      body.textContent = "Chưa tải được tài liệu. Hãy kiểm tra kết nối hoặc phiên đăng nhập.";
      const retry = document.createElement("button");
      retry.type = "button";
      retry.textContent = "Thử lại";
      retry.addEventListener("click", load);
      body.append(retry);
    } finally {
      loading = false;
    }
  };
  document.querySelectorAll("[data-document-open]").forEach(button => {
    button.hidden = false;
    button.addEventListener("click", () => {
      opener = button;
      savedScroll = window.scrollY;
      panel.hidden = false;
      workspace.classList.add("documents-open", "documents-wide");
      expand.textContent = "Thu hẹp";
      setModal();
      body.querySelectorAll(".full-document").forEach(doc => { doc.open = doc.dataset.docId === button.dataset.docId; });
      body.scrollTop = 0;
      if (loaded && search.value.trim()) runSearch();
      close.focus({ preventScroll: true });
      panel.scrollIntoView({ block: "start" });
      load();
    });
  });
  close.addEventListener("click", () => {
    panel.hidden = true;
    workspace.classList.remove("documents-open", "documents-wide");
    setModal();
    opener?.focus({ preventScroll: true });
    window.scrollTo({ top: savedScroll });
  });
  expand.addEventListener("click", () => {
    expand.textContent = workspace.classList.toggle("documents-wide") ? "Thu hẹp" : "Mở rộng";
    panel.scrollIntoView({ block: "start" });
  });
  narrow.addEventListener("change", setModal);
  panel.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
      event.preventDefault();
      search.focus();
      search.select();
    }
    if (event.key === "Escape") close.click();
    if (event.key === "Tab" && narrow.matches) {
      const controls = [...panel.querySelectorAll("button, input, summary, [tabindex='0']")]
        .filter(control => !control.disabled && control.getClientRects().length);
      if (event.shiftKey && document.activeElement === controls[0]) {
        event.preventDefault(); controls.at(-1)?.focus();
      } else if (!event.shiftKey && document.activeElement === controls.at(-1)) {
        event.preventDefault(); controls[0]?.focus();
      }
    }
  });
});
