(() => {
    if (window.vaxisCitationPreview) return;
    window.vaxisCitationPreview = true;
    const previews = new WeakMap();
    let active = null;
    let popup = null;
    let timer = null;
    const close = () => {
        clearTimeout(timer);
        active?.removeAttribute('aria-describedby');
        popup?.remove();
        active = popup = null;
    };
    const scheduleClose = () => {
        clearTimeout(timer);
        timer = setTimeout(close, 160);
    };
    const position = () => {
        if (!active?.isConnected || !popup) return close();
        const anchor = active.getBoundingClientRect();
        if (anchor.bottom < 0 || anchor.top > innerHeight) return close();
        const box = popup.getBoundingClientRect();
        const left = Math.max(12, Math.min(anchor.left, innerWidth - box.width - 12));
        const top = anchor.top > box.height + 20 ? anchor.top - box.height - 10 : Math.min(anchor.bottom + 10, innerHeight - box.height - 12);
        popup.style.left = `${left}px`;
        popup.style.top = `${Math.max(12, top)}px`;
    };
    const open = (chip) => {
        clearTimeout(timer);
        if (active === chip && popup) return;
        close();
        const preview = chip.getAttribute('title') || previews.get(chip);
        if (!preview) return;
        previews.set(chip, preview);
        chip.removeAttribute('title');
        active = chip;
        const [title, heading, ...body] = preview.split('\n');
        popup = document.createElement('div');
        popup.className = 'vacxis-source-preview';
        popup.id = 'vacxis-source-preview';
        popup.setAttribute('role', 'tooltip');
        chip.setAttribute('aria-describedby', popup.id);
        for (const [tag, text, className] of [
            ['strong', title, 'source-title'],
            ['div', heading, 'source-heading'],
            ['div', body.join('\n'), 'source-excerpt'],
        ]) {
            const element = document.createElement(tag);
            element.className = className;
            element.textContent = text;
            popup.append(element);
        }
        const href = chip.getAttribute('href');
        const footer = document.createElement(href ? 'a' : 'span');
        footer.textContent = href ? 'Mở tài liệu ↗' : 'Ngữ cảnh đồ thị tri thức';
        if (href) {
            footer.href = href;
            footer.target = '_blank';
            footer.rel = 'noopener noreferrer';
        }
        popup.append(footer);
        popup.addEventListener('pointerenter', () => clearTimeout(timer));
        popup.addEventListener('pointerleave', scheduleClose);
        (chip.closest('.gradio-container') || document.body).append(popup);
        position();
    };
    document.addEventListener('pointerover', event => {
        const chip = event.target.closest?.('#vacxis-chat .vacxis-citation');
        if (chip) open(chip);
    });
    document.addEventListener('pointerout', event => {
        const chip = event.target.closest?.('#vacxis-chat .vacxis-citation');
        if (chip && chip === active && chip !== document.activeElement && !chip.contains(event.relatedTarget)) scheduleClose();
    });
    document.addEventListener('focusin', event => {
        const chip = event.target.closest?.('#vacxis-chat .vacxis-citation');
        if (chip) open(chip);
    });
    document.addEventListener('focusout', event => {
        if (!popup?.contains(event.relatedTarget) && !event.relatedTarget?.closest?.('.vacxis-citation')) scheduleClose();
    });
    document.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
    document.addEventListener('scroll', event => { if (popup && !popup.contains(event.target)) position(); }, true);
    window.addEventListener('resize', close);
})();
