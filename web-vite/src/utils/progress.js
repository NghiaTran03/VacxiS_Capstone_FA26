/**
 * Vietnamese progress stage labels and elapsed time formatting.
 */

export const STAGES = {
  planning: ['Đang phân tích câu hỏi…', 'Đã phân tích câu hỏi'],
  searching: ['Đang tìm tài liệu liên quan…', 'Đã tìm tài liệu liên quan'],
  reranking: ['Đang xếp hạng lại các đoạn tài liệu…', 'Đã xếp hạng lại các đoạn tài liệu'],
  generating: ['Đang soạn câu trả lời…', 'Đã soạn câu trả lời'],
};

export function formatVietnameseSeconds(elapsed) {
  const safe = Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0;
  return safe.toFixed(1).replace('.', ',');
}

export function buildProgressView(events = [], outcome = null, elapsed = 0) {
  const steps = [];
  events.forEach(([stage, count], index) => {
    const complete =
      index < events.length - 1 || outcome === 'answered' || outcome === 'clarified';
    if (count !== null && count !== undefined) {
      steps.push({
        icon: '✓',
        text: `Đã chọn ${count} đoạn tài liệu để trả lời`,
        complete: true,
      });
    }
    const [pendingText, completedText] = STAGES[stage] || [stage, stage];
    steps.push({
      icon: complete ? '✓' : '◉',
      text: complete ? completedText : pendingText,
      complete,
    });
  });

  if (outcome === 'clarified') {
    steps.push({ icon: '•', text: 'Cần làm rõ câu hỏi', complete: true });
  }
  if (outcome === 'failed') {
    steps.push({ icon: '!', text: 'Quá trình trả lời bị gián đoạn', complete: false });
  }

  const seconds = formatVietnameseSeconds(elapsed);
  let title = 'Quá trình trả lời';
  if (outcome === 'answered') {
    title = `Đã trả lời trong ${seconds} giây`;
  } else if (outcome === null) {
    title = `Đang xử lý · ${seconds} giây`;
  }

  return {
    title,
    status: outcome ? 'done' : 'pending',
    steps,
  };
}
