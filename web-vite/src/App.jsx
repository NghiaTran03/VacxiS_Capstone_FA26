import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import ChatInterface from './components/ChatInterface';
import DocumentsPage from './components/DocumentsPage';
import Sidebar from './components/Sidebar';
import './App.css';

const THEME_STORAGE_KEY = 'vacxis-document-theme';
const VALID_THEMES = new Set(['system', 'light', 'dark']);

function readInitialTheme() {
  const fromQuery = new URLSearchParams(window.location.search).get('__theme');
  if (VALID_THEMES.has(fromQuery)) {
    return fromQuery;
  }
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (VALID_THEMES.has(saved)) {
      return saved;
    }
  } catch {
    // Ignore storage read errors in restricted contexts.
  }
  return 'light';
}

function isDocumentsPath(pathname) {
  return pathname === '/documents' || pathname.startsWith('/documents/');
}

export default function App() {
  const [pathname, setPathname] = useState(() => window.location.pathname);
  const [theme, setTheme] = useState(readInitialTheme);

  const [config, setConfig] = useState({
    preview: false,
    max_question_length: 2000,
  });
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [loadedConversation, setLoadedConversation] = useState(null);
  const [draft, setDraft] = useState('');
  const [focusTick, setFocusTick] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [liveTurn, setLiveTurn] = useState(null);
  const [failedTurn, setFailedTurn] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const currentConversation =
    currentConversationId && loadedConversation?.id === currentConversationId
      ? loadedConversation
      : null;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const handleThemeChange = useCallback((nextTheme) => {
    const valid = VALID_THEMES.has(nextTheme) ? nextTheme : 'light';
    setTheme(valid);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, valid);
    } catch {
      // Selected theme stays active in memory even if storage is unavailable.
    }
  }, []);

  useEffect(() => {
    const onPopState = () => setPathname(window.location.pathname);
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      const list = await api.listConversations();
      setConversations(list);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  }, []);

  const onDocumentsRoute = isDocumentsPath(pathname);

  useEffect(() => {
    if (onDocumentsRoute) return undefined;
    let cancelled = false;
    api
      .getConfig()
      .then((payload) => {
        if (!cancelled) {
          setConfig(payload);
        }
      })
      .catch((error) => console.error('Failed to load config:', error));
    api
      .listConversations()
      .then((list) => {
        if (!cancelled) {
          setConversations(list);
        }
      })
      .catch((error) => console.error('Failed to load conversations:', error));
    return () => {
      cancelled = true;
    };
  }, [onDocumentsRoute]);

  const loadedConversationId = loadedConversation?.id ?? null;

  useEffect(() => {
    if (onDocumentsRoute || !currentConversationId) return undefined;
    if (loadedConversationId === currentConversationId) return undefined;

    let cancelled = false;
    api
      .getConversation(currentConversationId)
      .then((detail) => {
        if (!cancelled) {
          setLoadedConversation(detail);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          console.error('Failed to load conversation:', error);
          setErrorMessage(error.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [onDocumentsRoute, currentConversationId, loadedConversationId]);

  const handleNewConversation = async () => {
    if (isLoading) return;
    setErrorMessage(null);
    setFailedTurn(null);
    setDraft('');

    // Reuse an untouched blank conversation instead of spawning empty drafts.
    if (currentConversation && (currentConversation.messages || []).length === 0) {
      setFocusTick((tick) => tick + 1);
      return;
    }

    try {
      const created = await api.createConversation();
      setConversations((prev) => [
        {
          id: created.id,
          created_at: created.created_at,
          updated_at: created.updated_at,
          title: created.title,
          message_count: 0,
        },
        ...prev.filter((item) => item.id !== created.id),
      ]);
      setLoadedConversation(created);
      setCurrentConversationId(created.id);
      setFocusTick((tick) => tick + 1);
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  const handleSelectConversation = (id) => {
    if (isLoading) return;
    setErrorMessage(null);
    setFailedTurn(null);
    setDraft('');
    setCurrentConversationId(id);
  };

  const handleDeleteConversation = async (id) => {
    if (isLoading) return;
    try {
      await api.deleteConversation(id);
      setConversations((prev) => prev.filter((item) => item.id !== id));
      if (id === currentConversationId) {
        setCurrentConversationId(null);
        setLoadedConversation(null);
        setFailedTurn(null);
      }
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  const handleSendMessage = async (rawText) => {
    const text = rawText.trim();
    if (!text) return;

    const maxLen = config.max_question_length || 2000;
    if (text.length > maxLen) {
      setErrorMessage('Câu hỏi có thể dài tối đa 2.000 ký tự.');
      return;
    }

    setErrorMessage(null);
    setFailedTurn(null);
    setDraft('');
    setIsLoading(true);

    let latestEvents = [];
    let latestElapsed = 0;
    setLiveTurn({ userText: text, events: [], elapsed: 0 });

    try {
      let receivedTerminalEvent = false;
      await api.sendMessageStream(currentConversationId, text, (eventType, payload) => {
        if (eventType === 'progress') {
          latestEvents = payload.events || [];
          latestElapsed = payload.elapsed ?? latestElapsed;
          setLiveTurn({
            userText: text,
            events: latestEvents,
            elapsed: latestElapsed,
          });
          return;
        }

        if (eventType === 'complete') {
          receivedTerminalEvent = true;
          const saved = payload.conversation;
          setLiveTurn(null);
          setLoadedConversation(saved);
          setCurrentConversationId(saved.id);
          refreshConversations();
          return;
        }

        if (eventType === 'error') {
          receivedTerminalEvent = true;
          setLiveTurn(null);
          setFailedTurn({
            events: payload.events || latestEvents,
            elapsed: payload.elapsed ?? latestElapsed,
          });
          setErrorMessage(
            payload.message ||
              'Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại.',
          );
          setDraft(text);
        }
      });

      if (!receivedTerminalEvent) {
        setLiveTurn(null);
        setFailedTurn({ events: latestEvents, elapsed: latestElapsed });
        setErrorMessage(
          'Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại.',
        );
        setDraft(text);
      }
    } catch (error) {
      setLiveTurn(null);
      setFailedTurn({ events: latestEvents, elapsed: latestElapsed });
      setErrorMessage(
        error.message ||
          'Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại.',
      );
      setDraft(text);
    } finally {
      setIsLoading(false);
    }
  };

  if (onDocumentsRoute) {
    return (
      <DocumentsPage
        theme={theme}
        onThemeChange={handleThemeChange}
        onNavigateChat={() => {
          window.history.pushState({}, '', '/');
          setPathname('/');
        }}
      />
    );
  }

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        theme={theme}
        onThemeChange={handleThemeChange}
        mobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />
      <ChatInterface
        conversation={currentConversation}
        isPreview={Boolean(config.preview)}
        maxQuestionLength={config.max_question_length || 2000}
        draft={draft}
        onDraftChange={setDraft}
        focusTick={focusTick}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        liveTurn={liveTurn}
        failedTurn={failedTurn}
        errorMessage={errorMessage}
        onDismissError={() => setErrorMessage(null)}
        onOpenSidebar={() => setMobileSidebarOpen(true)}
      />
    </div>
  );
}
