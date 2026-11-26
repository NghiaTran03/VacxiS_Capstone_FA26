import { describe, expect, it } from 'vitest';
import { buildProgressView, formatVietnameseSeconds } from '../progress';

describe('progress utilities', () => {
  it('formats elapsed seconds using Vietnamese decimal comma', () => {
    expect(formatVietnameseSeconds(2.34)).toBe('2,3');
    expect(formatVietnameseSeconds(0)).toBe('0,0');
  });

  it('builds live, answered, and failed views matching legacy labels', () => {
    const live = buildProgressView([['planning', null], ['generating', 3]], null, 1.5);
    expect(live.title).toBe('Đang xử lý · 1,5 giây');
    expect(live.steps.some((step) => step.text === 'Đã chọn 3 đoạn tài liệu để trả lời')).toBe(true);
    expect(live.steps.at(-1).text).toBe('Đang soạn câu trả lời…');

    const answered = buildProgressView([['planning', null]], 'answered', 2.0);
    expect(answered.title).toBe('Đã trả lời trong 2,0 giây');
    expect(answered.steps[0].text).toBe('Đã phân tích câu hỏi');

    const failed = buildProgressView([['planning', null]], 'failed', 0.8);
    expect(failed.steps.at(-1).text).toBe('Quá trình trả lời bị gián đoạn');
  });
});
