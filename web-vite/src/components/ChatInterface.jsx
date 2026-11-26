import { useEffect, useRef, useState } from 'react';
import MarkdownAnswer from './MarkdownAnswer';
import ProgressCard from './ProgressCard';
import './ChatInterface.css';

export default function ChatInterface({
  conversation,
  isPreview,
  maxQuestionLength,
  draft,
  onDraftChange,
  focusTick,
  onSendMessage,
  isLoading,
  liveTurn,
  failedTurn,
  errorMessage,
  onDismissError,
  onOpenSidebar,
}) {
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const [copiedId, setCopiedId] = useState(null);

  const messages = conversation?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length, liveTurn?.events?.length, failedTurn]);

  useEffect(() => {
    if (focusTick > 0) {
      textareaRef.current?.focus();
    }
  }, [focusTick]);

  const copyText = async (text, key) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(key);
      setTimeout(() => {
        setCopiedId((prev) => (prev === key ? null : prev));
      }, 1500);
    } catch {
      // Clipboard API may be unavailable in insecure contexts.
    }
  };

  const handleCopyAll = () => {
    const transcript = messages
      .filter((msg) => !msg.is_progress)
      .map((msg) => `${msg.role === 'user' ? 'Bạn' : 'VaxiS'}:\n${msg.content}`)
      .join('\n\n');
    if (transcript) {
      copyText(transcript, 'all');
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (isLoading) return;
    onSendMessage(draft);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  const hasVisibleMessages =
    messages.length > 0 || Boolean(liveTurn) || Boolean(failedTurn);
  const realMessageCount = messages.filter((msg) => !msg.is_progress).length;

  return (
    <main className="chat-interface">
      <header className="chat-topbar">
        <div className="chat-topbar-left">
          <button
            type="button"
            className="mobile-menu-btn"
            onClick={onOpenSidebar}
            aria-label="Mở danh sách cuộc trò chuyện"
          >
            ☰
          </button>
          <span className="chat-topbar-title">
            {conversation?.title || 'VaxiS · Trợ lý thông tin vắc xin'}
          </span>
        </div>
        {realMessageCount > 0 && (
          <button
            type="button"
            className="copy-all-btn"
            onClick={handleCopyAll}
          >
            {copiedId === 'all' ? 'Đã sao chép ✓' : 'Sao chép hội thoại'}
          </button>
        )}
      </header>

      {isPreview && (
        <div className="preview-banner" role="status">
          Bản xem trước giao diện · Phản hồi mẫu từ tài liệu có sẵn. Pipeline AI chưa được kết nối.
        </div>
      )}

      <div className="messages-container" id="vacxis-chat">
        {!hasVisibleMessages ? (
          <div className="empty-state">
            <h2>Bạn muốn tìm hiểu về vắc xin nào?</h2>
            <p>
              Đặt câu hỏi về thành phần, chỉ định, liều dùng, chống chỉ định hoặc phản ứng phụ của vắc xin.
            </p>
          </div>
        ) : (
          <div className="messages-inner">
            {messages.map((msg, index) => {
              const convKey = conversation?.id || 'draft';
              if (msg.is_progress) {
                const progress = msg.progress || {};
                return (
                  <div key={`${convKey}-progress-${index}`} className="message-row assistant-row">
                    <ProgressCard
                      events={progress.events || []}
                      outcome={progress.outcome || 'answered'}
                      elapsed={progress.elapsed || 0}
                      fallbackTitle={msg.metadata?.title}
                      fallbackContent={msg.content}
                    />
                  </div>
                );
              }

              if (msg.role === 'user') {
                return (
                  <div key={`${convKey}-user-${index}`} className="message-row user-row">
                    <div className="user-bubble">
                      <div className="message-label">Bạn</div>
                      <div className="user-text">{msg.content}</div>
                    </div>
                  </div>
                );
              }

              const copyKey = `${convKey}-msg-${index}`;
              return (
                <div key={`${convKey}-assistant-${index}`} className="message-row assistant-row">
                  <div className="assistant-bubble">
                    <div className="assistant-header">
                      <span className="message-label">VaxiS</span>
                      <button
                        type="button"
                        className="copy-msg-btn"
                        onClick={() => copyText(msg.content, copyKey)}
                      >
                        {copiedId === copyKey ? 'Đã sao chép ✓' : 'Sao chép'}
                      </button>
                    </div>
                    <MarkdownAnswer content={msg.content} sources={msg.sources} />
                  </div>
                </div>
              );
            })}

            {liveTurn && (
              <>
                <div className="message-row user-row">
                  <div className="user-bubble">
                    <div className="message-label">Bạn</div>
                    <div className="user-text">{liveTurn.userText}</div>
                  </div>
                </div>
                <div className="message-row assistant-row">
                  <ProgressCard
                    events={liveTurn.events}
                    outcome={null}
                    elapsed={liveTurn.elapsed}
                  />
                </div>
              </>
            )}

            {failedTurn && (
              <div className="message-row assistant-row">
                <ProgressCard
                  events={failedTurn.events}
                  outcome="failed"
                  elapsed={failedTurn.elapsed}
                />
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="composer-shell">
        {errorMessage && (
          <div className="error-banner" role="alert">
            <span>{errorMessage}</span>
            <button
              type="button"
              className="error-dismiss"
              onClick={onDismissError}
              aria-label="Đóng thông báo lỗi"
            >
              ×
            </button>
          </div>
        )}

        <form className="input-form" onSubmit={handleSubmit}>
          <textarea
            ref={textareaRef}
            id="vacxis-question"
            className="message-input"
            placeholder="Hỏi VaxiS về vắc xin…"
            value={draft}
            onChange={(event) => onDraftChange(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={2}
            aria-label="Câu hỏi"
          />
          <button
            type="submit"
            className="send-button"
            disabled={!draft.trim() || isLoading}
          >
            {isLoading ? 'Đang gửi…' : 'Gửi ↑'}
          </button>
        </form>

        <div className="composer-meta">
          <small className="vacxis-disclaimer">
            * Các câu trả lời từ VaxiS chỉ nhằm mục đích tham khảo. Để được tư vấn hoặc chẩn đoán y tế chuẩn xác, hãy tham khảo ý kiến của các bác sĩ chuyên gia. Câu trả lời của VaxiS có thể chứa thông tin không chính xác.
          </small>
          {draft.length > maxQuestionLength * 0.85 && (
            <small
              className={`char-counter ${draft.trim().length > maxQuestionLength ? 'over' : ''}`}
            >
              {draft.trim().length}/{maxQuestionLength}
            </small>
          )}
        </div>
      </div>
    </main>
  );
}
