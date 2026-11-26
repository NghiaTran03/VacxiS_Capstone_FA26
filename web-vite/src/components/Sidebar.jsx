import './Sidebar.css';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  theme,
  onThemeChange,
  mobileOpen,
  onCloseMobile,
}) {
  return (
    <>
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}
      <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <h1>VaxiS</h1>
            <p className="sidebar-subtitle">Trợ lý thông tin vắc xin</p>
          </div>
          <button
            type="button"
            className="new-conversation-btn"
            onClick={() => {
              onNewConversation();
              onCloseMobile?.();
            }}
          >
            + Cuộc trò chuyện mới
          </button>
          <a
            className="library-link"
            href="/documents"
            target="_blank"
            rel="noopener noreferrer"
          >
            Tài liệu vắc xin ↗
          </a>
        </div>

        <div className="conversation-section-label">Các cuộc trò chuyện</div>
        <div className="conversation-list" role="list">
          {conversations.length === 0 ? (
            <div className="no-conversations">Chưa có cuộc trò chuyện nào</div>
          ) : (
            conversations.map((conv) => {
              const active = conv.id === currentConversationId;
              return (
                <div
                  key={conv.id}
                  role="listitem"
                  className={`conversation-item ${active ? 'active' : ''}`}
                >
                  <button
                    type="button"
                    className="conversation-select"
                    aria-current={active ? 'page' : undefined}
                    onClick={() => {
                      onSelectConversation(conv.id);
                      onCloseMobile?.();
                    }}
                  >
                    <span className="conversation-title">
                      {conv.title || 'Cuộc trò chuyện mới'}
                    </span>
                    <span className="conversation-meta">
                      {conv.message_count} tin nhắn
                    </span>
                  </button>
                  <button
                    type="button"
                    className="conversation-delete"
                    title="Xóa cuộc trò chuyện"
                    aria-label={`Xóa cuộc trò chuyện ${conv.title || 'Cuộc trò chuyện mới'}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                  >
                    ×
                  </button>
                </div>
              );
            })
          )}
        </div>

        <div className="sidebar-footer">
          <label className="sidebar-theme-control" htmlFor="chat-theme">
            <span>Giao diện</span>
            <select
              id="chat-theme"
              value={theme}
              onChange={(event) => onThemeChange(event.target.value)}
            >
              <option value="system">Theo hệ thống</option>
              <option value="light">Sáng</option>
              <option value="dark">Tối</option>
            </select>
          </label>
        </div>
      </aside>
    </>
  );
}
