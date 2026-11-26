/**
 * REST and Server-Sent Events client for the VaxiS FastAPI backend.
 */

const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

async function parseErrorDetail(response, fallback) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === 'string' && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
    // Fall back to generic message when body is not JSON.
  }
  return fallback;
}

export const api = {
  async getConfig() {
    const response = await fetch(`${API_BASE}/api/config`);
    if (!response.ok) {
      throw new Error('Không thể tải cấu hình máy chủ.');
    }
    return response.json();
  },

  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Không thể tải danh sách cuộc trò chuyện.');
    }
    return response.json();
  },

  async createConversation() {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      throw new Error('Không thể tạo cuộc trò chuyện mới.');
    }
    return response.json();
  },

  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}`,
    );
    if (!response.ok) {
      throw new Error(
        await parseErrorDetail(response, 'Không thể tải nội dung cuộc trò chuyện.'),
      );
    }
    return response.json();
  },

  async deleteConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}`,
      { method: 'DELETE' },
    );
    if (!response.ok) {
      throw new Error(
        await parseErrorDetail(response, 'Không thể xóa cuộc trò chuyện.'),
      );
    }
    return response.json();
  },

  /**
   * Stream a chat turn over SSE with proper multi-chunk line buffering.
   * @param {string | null} conversationId
   * @param {string} content
   * @param {(eventType: string, payload: object) => void} onEvent
   * @param {AbortSignal} [signal]
   */
  async sendMessageStream(conversationId, content, onEvent, signal) {
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: conversationId, content }),
      signal,
    });

    if (!response.ok) {
      const detail = await parseErrorDetail(
        response,
        'Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại.',
      );
      throw new Error(detail);
    }

    if (!response.body) {
      throw new Error('Máy chủ không hỗ trợ phản hồi streaming.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const processLine = (rawLine) => {
      const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
      if (!line.startsWith('data: ')) {
        return;
      }
      const rawJson = line.slice(6);
      if (!rawJson) {
        return;
      }
      try {
        const event = JSON.parse(rawJson);
        onEvent(event.type, event);
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        processLine(line);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      for (const line of buffer.split('\n')) {
        processLine(line);
      }
    }
  },

  async listDocuments() {
    const response = await fetch(`${API_BASE}/api/documents`);
    if (!response.ok) {
      throw new Error('Không thể tải danh sách tài liệu vắc xin.');
    }
    return response.json();
  },

  async getDocument(docId, { chunk = null, used = [] } = {}) {
    const params = new URLSearchParams();
    if (chunk) {
      params.set('chunk', chunk);
    }
    for (const item of used) {
      if (item) {
        params.append('used', item);
      }
    }
    const query = params.toString();
    const url = `${API_BASE}/api/documents/${encodeURIComponent(docId)}${query ? `?${query}` : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      const err = new Error(
        await parseErrorDetail(response, 'Không tìm thấy tài liệu.'),
      );
      err.status = response.status;
      throw err;
    }
    return response.json();
  },
};
