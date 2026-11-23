(() => {
    const themeSelect = document.getElementById('document-theme');
    const themeKey = 'vacxis-document-theme';
    const themes = ['system', 'light', 'dark'];
    let theme = new URLSearchParams(location.search).get('__theme');
    if (!themes.includes(theme)) {
        try {
            theme = localStorage.getItem(themeKey);
        } catch {
            theme = 'system';
        }
    }
    themeSelect.value = themes.includes(theme) ? theme : 'system';
    document.documentElement.dataset.theme = themeSelect.value;
    themeSelect.addEventListener('change', () => {
        document.documentElement.dataset.theme = themeSelect.value;
        try {
            localStorage.setItem(themeKey, themeSelect.value);
        } catch {
            // The selected appearance remains active when browser storage is unavailable.
        }
    });

    const input = document.getElementById('document-search');
    const content = document.getElementById('vaccine-content');
    if (!input || !content) return;
    const previous = document.getElementById('search-previous');
    const next = document.getElementById('search-next');
    const status = document.getElementById('search-status');
    let matches = [];
    let selected = -1;
    let savedExpansion = null;
    let searchedQuery = '';
    let debounce;

    function fold(text) {
        return text.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase('vi').replace(/đ/g, 'd');
    }

    function textMap(text) {
        let normalized = '';
        const starts = [];
        const ends = [];
        let offset = 0;
        for (const character of text) {
            const folded = fold(character);
            normalized += folded;
            for (let index = 0; index < folded.length; index++) {
                starts.push(offset);
                ends.push(offset + character.length);
            }
            if (!folded && ends.length) ends[ends.length - 1] = offset + character.length;
            offset += character.length;
        }
        return { normalized, starts, ends };
    }

    function selectMatch(index) {
        if (!matches.length) return;
        if (selected >= 0) matches[selected].classList.remove('current');
        selected = (index + matches.length) % matches.length;
        const match = matches[selected];
        match.classList.add('current');
        let parent = match.parentElement;
        while (parent && parent !== content) {
            if (parent.tagName === 'DETAILS') parent.open = true;
            parent = parent.parentElement;
        }
        status.textContent = `${selected + 1} / ${matches.length} kết quả`;
        match.scrollIntoView({ block: 'center' });
    }

    function search() {
        for (const mark of matches) mark.replaceWith(document.createTextNode(mark.textContent));
        content.normalize();
        matches = [];
        selected = -1;
        const query = fold(input.value.trim());
        searchedQuery = query;
        if (!query) {
            if (savedExpansion) {
                for (const [details, open] of savedExpansion) details.open = open;
            }
            savedExpansion = null;
            status.textContent = '';
            previous.disabled = next.disabled = true;
            return;
        }
        if (!savedExpansion) {
            savedExpansion = new Map(Array.from(content.querySelectorAll('details'), details => [details, details.open]));
        }
        for (const body of content.querySelectorAll('.category-body')) {
            const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
            const nodes = [];
            while (walker.nextNode()) nodes.push(walker.currentNode);
            for (const node of nodes) {
                const text = node.textContent;
                const mapped = textMap(text);
                let found = mapped.normalized.indexOf(query);
                if (found < 0) continue;
                const fragment = document.createDocumentFragment();
                let cursor = 0;
                while (found >= 0) {
                    const start = mapped.starts[found];
                    const end = mapped.ends[found + query.length - 1];
                    fragment.append(document.createTextNode(text.slice(cursor, start)));
                    const mark = document.createElement('mark');
                    mark.dataset.searchMatch = '';
                    mark.textContent = text.slice(start, end);
                    fragment.append(mark);
                    matches.push(mark);
                    cursor = end;
                    found = mapped.normalized.indexOf(query, found + query.length);
                }
                fragment.append(document.createTextNode(text.slice(cursor)));
                node.replaceWith(fragment);
            }
        }
        for (const category of content.querySelectorAll('.category')) {
            category.open = category.querySelector('mark[data-search-match]') !== null || savedExpansion.get(category);
        }
        previous.disabled = next.disabled = matches.length === 0;
        status.textContent = matches.length ? `${matches.length} kết quả` : 'Không tìm thấy kết quả';
        selectMatch(0);
    }

    input.addEventListener('input', () => {
        clearTimeout(debounce);
        debounce = setTimeout(search, 120);
    });
    input.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            clearTimeout(debounce);
            if (searchedQuery !== fold(input.value.trim())) search();
            else if (matches.length) selectMatch(selected + (event.shiftKey ? -1 : 1));
            else search();
        }
        if (event.key === 'Escape') {
            input.value = '';
            clearTimeout(debounce);
            search();
        }
    });
    previous.addEventListener('click', () => selectMatch(selected - 1));
    next.addEventListener('click', () => selectMatch(selected + 1));
})();
