import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import { buildTextMap, foldVietnamese } from '../utils/vietnameseSearch';
import './DocumentsPage.css';

function parseDocumentsQuery(search) {
  const params = new URLSearchParams(search);
  return {
    docId: params.get('doc'),
    chunkId: params.get('chunk'),
    used: params.getAll('used'),
  };
}

function openAncestorDetails(element, boundary) {
  let parent = element.parentElement;
  while (parent && parent !== boundary) {
    if (parent.tagName === 'DETAILS') {
      parent.open = true;
    }
    parent = parent.parentElement;
  }
}

function restoreDetailsExpansion(expansionMap) {
  if (!expansionMap) return;
  for (const [details, open] of expansionMap) {
    if (details.isConnected) {
      details.open = open;
    }
  }
}

const VaccineContentBody = memo(function VaccineContentBody({ documentData }) {
  return (
    <details className="vaccine" open>
      <summary>{documentData.title}</summary>
      {documentData.categories.map((category) => (
        <details
          key={category.name}
          className="category"
          data-category={category.name}
          open={category.used}
        >
          <summary>
            {category.label}
            {category.used && (
              <span
                className="used-marker"
                aria-label="Được dùng trong câu trả lời"
              >
                ✓
              </span>
            )}
          </summary>
          <div className="category-body">
            {category.items.map((item, idx) =>
              item.html !== undefined ? (
                <article key={`${item.source_file || 'src'}-${idx}`}>
                  {item.source_file && (
                    <h3 className="source-file">{item.source_file}</h3>
                  )}
                  <div dangerouslySetInnerHTML={{ __html: item.html }} />
                </article>
              ) : (
                <article key={`chunk-${idx}`}>
                  {item.header && <h3>{item.header}</h3>}
                  <div className="excerpt">{item.text}</div>
                </article>
              ),
            )}
          </div>
        </details>
      ))}
    </details>
  );
});

function DocumentViewer({ documentData }) {
  const [searchValue, setSearchValue] = useState('');
  const [matchCount, setMatchCount] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [searchStatus, setSearchStatus] = useState('');

  const contentRef = useRef(null);
  const citationRef = useRef(null);
  const matchesRef = useRef([]);
  const selectedRef = useRef(-1);
  const savedExpansionRef = useRef(null);
  const searchedQueryRef = useRef('');
  const debounceRef = useRef(null);

  useEffect(() => {
    if (documentData.cited && window.location.hash === '#citation') {
      citationRef.current?.scrollIntoView({ block: 'start' });
      citationRef.current?.focus({ preventScroll: true });
    }
  }, [documentData]);

  useEffect(() => () => clearTimeout(debounceRef.current), []);

  const clearHighlights = useCallback((restoreExpansion = false) => {
    const content = contentRef.current;
    for (const mark of matchesRef.current) {
      if (mark.parentNode) {
        mark.replaceWith(document.createTextNode(mark.textContent || ''));
      }
    }
    if (content) {
      content.normalize();
    }
    matchesRef.current = [];
    selectedRef.current = -1;
    setMatchCount(0);
    setSelectedIndex(-1);

    if (restoreExpansion && savedExpansionRef.current) {
      restoreDetailsExpansion(savedExpansionRef.current);
      savedExpansionRef.current = null;
    }
  }, []);

  const selectMatch = useCallback((index) => {
    const matches = matchesRef.current;
    const content = contentRef.current;
    if (!matches.length || !content) return;
    if (selectedRef.current >= 0 && matches[selectedRef.current]) {
      matches[selectedRef.current].classList.remove('current');
    }
    const nextIdx = ((index % matches.length) + matches.length) % matches.length;
    selectedRef.current = nextIdx;
    setSelectedIndex(nextIdx);
    const match = matches[nextIdx];
    match.classList.add('current');
    openAncestorDetails(match, content);
    setSearchStatus(`${nextIdx + 1} / ${matches.length} kết quả`);
    match.scrollIntoView({ block: 'center' });
  }, []);

  const runSearch = useCallback(
    (rawQuery) => {
      const content = contentRef.current;
      if (!content) return;

      clearHighlights(false);
      const query = foldVietnamese(rawQuery.trim());
      searchedQueryRef.current = query;

      if (!query) {
        restoreDetailsExpansion(savedExpansionRef.current);
        savedExpansionRef.current = null;
        setSearchStatus('');
        return;
      }

      if (!savedExpansionRef.current) {
        savedExpansionRef.current = new Map(
          Array.from(content.querySelectorAll('details'), (details) => [
            details,
            details.open,
          ]),
        );
      }

      const foundMarks = [];
      for (const body of content.querySelectorAll('.category-body')) {
        const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
        const nodes = [];
        while (walker.nextNode()) {
          nodes.push(walker.currentNode);
        }
        for (const node of nodes) {
          const text = node.textContent || '';
          const mapped = buildTextMap(text);
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
            foundMarks.push(mark);
            cursor = end;
            found = mapped.normalized.indexOf(query, found + query.length);
          }
          fragment.append(document.createTextNode(text.slice(cursor)));
          node.replaceWith(fragment);
        }
      }

      for (const category of content.querySelectorAll('.category')) {
        category.open =
          category.querySelector('mark[data-search-match]') !== null ||
          Boolean(savedExpansionRef.current.get(category));
      }

      matchesRef.current = foundMarks;
      setMatchCount(foundMarks.length);
      if (foundMarks.length === 0) {
        setSearchStatus('Không tìm thấy kết quả');
      } else {
        selectMatch(0);
      }
    },
    [clearHighlights, selectMatch],
  );

  const handleSearchInput = (event) => {
    const nextValue = event.target.value;
    setSearchValue(nextValue);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runSearch(nextValue), 120);
  };

  const handleSearchKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      clearTimeout(debounceRef.current);
      if (searchedQueryRef.current !== foldVietnamese(searchValue.trim())) {
        runSearch(searchValue);
      } else if (matchesRef.current.length > 0) {
        selectMatch(selectedRef.current + (event.shiftKey ? -1 : 1));
      } else {
        runSearch(searchValue);
      }
    }
    if (event.key === 'Escape') {
      setSearchValue('');
      clearTimeout(debounceRef.current);
      runSearch('');
    }
  };

  return (
    <>
      {documentData.cited && (
        <section
          id="citation"
          ref={citationRef}
          className="citation"
          tabIndex={-1}
        >
          <small>ĐOẠN ĐƯỢC TRÍCH DẪN</small>
          <h2>{documentData.cited.header || documentData.cited.doc_title}</h2>
          <div className="excerpt">{documentData.cited.text}</div>
        </section>
      )}

      {!documentData.has_original && (
        <p>
          Chưa có tệp nguồn tương ứng. Nội dung đã lập chỉ mục hiển thị bên dưới.
        </p>
      )}

      <div
        className="document-search"
        role="search"
        aria-label="Tìm trong toàn bộ tài liệu vắc xin"
      >
        <label htmlFor="document-search">Tìm trong tài liệu</label>
        <div className="search-controls">
          <input
            id="document-search"
            type="search"
            placeholder="Nhập từ khóa…"
            autoComplete="off"
            value={searchValue}
            onChange={handleSearchInput}
            onKeyDown={handleSearchKeyDown}
          />
          <button
            id="search-previous"
            type="button"
            disabled={matchCount === 0}
            aria-label="Kết quả trước"
            onClick={() => selectMatch(selectedIndex - 1)}
          >
            ↑
          </button>
          <button
            id="search-next"
            type="button"
            disabled={matchCount === 0}
            aria-label="Kết quả tiếp theo"
            onClick={() => selectMatch(selectedIndex + 1)}
          >
            ↓
          </button>
        </div>
        <span id="search-status" role="status" aria-live="polite">
          {searchStatus}
        </span>
      </div>

      <div id="vaccine-content" ref={contentRef}>
        <VaccineContentBody documentData={documentData} />
      </div>
    </>
  );
}

export default function DocumentsPage({ theme, onThemeChange, onNavigateChat }) {
  const [catalogue, setCatalogue] = useState([]);
  const [catalogueError, setCatalogueError] = useState(null);
  const [routeParams, setRouteParams] = useState(() =>
    parseDocumentsQuery(window.location.search),
  );
  const [docState, setDocState] = useState({
    key: null,
    data: null,
    error: null,
  });

  useEffect(() => {
    document.title = 'Tài liệu vắc xin · VaxiS';
    return () => {
      document.title = 'VaxiS · Trợ lý thông tin vắc xin';
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .listDocuments()
      .then((payload) => {
        if (!cancelled) {
          setCatalogue(payload.documents || []);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setCatalogueError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onPopState = () => {
      setRouteParams(parseDocumentsQuery(window.location.search));
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const usedKey = useMemo(() => routeParams.used.join('|'), [routeParams.used]);
  const requestKey = routeParams.docId
    ? `${routeParams.docId}:${routeParams.chunkId || ''}:${usedKey}`
    : null;

  useEffect(() => {
    if (!routeParams.docId || !requestKey) {
      return undefined;
    }

    let cancelled = false;
    api
      .getDocument(routeParams.docId, {
        chunk: routeParams.chunkId,
        used: routeParams.used,
      })
      .then((payload) => {
        if (!cancelled) {
          setDocState({ key: requestKey, data: payload, error: null });
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setDocState({
            key: requestKey,
            data: null,
            error: err.message || 'Không tìm thấy tài liệu.',
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [routeParams.docId, routeParams.chunkId, routeParams.used, requestKey]);

  const loadingDoc = Boolean(requestKey) && docState.key !== requestKey;
  const documentData =
    requestKey && docState.key === requestKey ? docState.data : null;
  const docError =
    requestKey && docState.key === requestKey ? docState.error : null;

  const handleSelectDocument = (event, docId) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) {
      return;
    }
    event.preventDefault();
    const url = `/documents?${new URLSearchParams({ doc: docId }).toString()}`;
    window.history.pushState({}, '', url);
    setRouteParams({ docId, chunkId: null, used: [] });
  };

  return (
    <div className="documents-page">
      <header className="documents-header">
        <div className="documents-brand">
          <strong>VaxiS</strong>
          <span>Thư viện tài liệu vắc xin</span>
        </div>
        <div className="documents-header-actions">
          <a
            className="back-chat-link"
            href="/"
            onClick={(event) => {
              if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) {
                return;
              }
              event.preventDefault();
              onNavigateChat();
            }}
          >
            ← Trò chuyện
          </a>
          <label className="theme-control" htmlFor="document-theme">
            Giao diện
            <select
              id="document-theme"
              value={theme}
              onChange={(event) => onThemeChange(event.target.value)}
            >
              <option value="system">Theo hệ thống</option>
              <option value="light">Sáng</option>
              <option value="dark">Tối</option>
            </select>
          </label>
        </div>
      </header>

      <div className="documents-layout">
        <nav className="documents-nav" aria-label="Danh sách tài liệu">
          {catalogueError ? (
            <p className="documents-error">{catalogueError}</p>
          ) : (
            catalogue.map((item) => {
              const url = `/documents?${new URLSearchParams({ doc: item.doc_id }).toString()}`;
              const active = item.doc_id === routeParams.docId;
              return (
                <a
                  key={item.doc_id}
                  href={url}
                  aria-current={active ? 'page' : undefined}
                  onClick={(event) => handleSelectDocument(event, item.doc_id)}
                >
                  {item.title}
                </a>
              );
            })
          )}
        </nav>

        <main className="documents-main">
          {!routeParams.docId && (
            <>
              <h1>Tài liệu vắc xin</h1>
              <p>Chọn một tài liệu trong danh sách để đọc.</p>
            </>
          )}

          {loadingDoc && <p className="documents-loading">Đang tải tài liệu…</p>}

          {docError && (
            <div className="documents-error-box" role="alert">
              {docError}
            </div>
          )}

          {documentData && !loadingDoc && (
            <DocumentViewer key={requestKey} documentData={documentData} />
          )}
        </main>
      </div>
    </div>
  );
}
