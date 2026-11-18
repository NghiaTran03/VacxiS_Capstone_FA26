(() => {
  const preference = window.matchMedia("(prefers-color-scheme: dark)");
  let saved = null;
  try { saved = localStorage.getItem("vacxis-theme"); } catch {}
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.querySelector("[data-theme-toggle]");
    const current = () => document.documentElement.dataset.theme || (preference.matches ? "dark" : "light");
    const label = () => { toggle.textContent = current() === "dark" ? "Chế độ sáng" : "Chế độ tối"; };
    toggle.hidden = false;
    label();
    preference.addEventListener("change", label);
    toggle.addEventListener("click", () => {
      const theme = current() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = theme;
      try { localStorage.setItem("vacxis-theme", theme); } catch {}
      label();
    });

    const invitation = document.querySelector("[data-invite-form]");
    const token = location.hash.slice(1);
    if (invitation && /^[A-Za-z0-9_-]{43}$/.test(token)) {
      invitation.elements.token.value = token;
      invitation.elements.action.value = "open";
      history.replaceState(null, "", location.pathname + location.search);
      invitation.requestSubmit();
    }

    const passwordToggle = document.querySelector("[data-password-toggle]");
    if (passwordToggle) {
      const password = document.getElementById(passwordToggle.getAttribute("aria-controls"));
      passwordToggle.hidden = false;
      passwordToggle.addEventListener("click", () => {
        const show = password.type === "password";
        password.type = show ? "text" : "password";
        passwordToggle.textContent = show ? "Ẩn mật khẩu" : "Hiện mật khẩu";
      });
    }

    document.querySelector("[data-focus]")?.focus();
    document.querySelector("[data-logout]")?.addEventListener("click", event => {
      if (!window.confirm("Sau khi đăng xuất, hãy xin nhóm đồ án cấp liên kết mới để đăng nhập lại.")) {
        event.preventDefault();
      }
    });
    document.querySelectorAll("[data-copy]").forEach(copy => {
      copy.hidden = false;
      copy.addEventListener("click", async () => {
        const field = document.getElementById(copy.dataset.copy);
        const feedback = copy.closest(".actions")?.querySelector("[data-copy-feedback]") || document.querySelector("[data-copy-feedback]");
        try {
          await navigator.clipboard.writeText(field.value);
          feedback.textContent = "Đã sao chép.";
        } catch {
          const details = field.closest("details");
          if (details) details.open = true;
          field.focus();
          field.select();
          feedback.textContent = "Đã chọn nội dung. Hãy sao chép bằng bàn phím hoặc menu.";
        }
      });
    });

    const formatter = new Intl.DateTimeFormat("vi-VN", { dateStyle: "short", timeStyle: "short" });
    document.querySelectorAll("[data-local-time]").forEach(element => {
      element.textContent = formatter.format(new Date(element.dateTime));
    });

    document.querySelectorAll("[data-source-index] a").forEach(link => {
      link.addEventListener("click", event => {
        const summary = document.getElementById(link.hash.slice(1));
        if (!summary) return;
        event.preventDefault();
        summary.parentElement.open = true;
        summary.focus({ preventScroll: true });
        summary.scrollIntoView({ block: "start" });
      });
    });

    const review = document.querySelector("[data-review-form]");
    if (review) {
      let edited = false;
      review.addEventListener("input", () => { edited = true; });
      review.addEventListener("submit", event => {
        queueMicrotask(() => { if (!event.defaultPrevented) edited = false; });
      });
      window.addEventListener("beforeunload", event => {
        if (edited) event.preventDefault();
      });
    }
  });
})();
