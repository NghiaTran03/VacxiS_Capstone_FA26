import { describe, expect, it } from 'vitest';
import { buildTextMap, foldVietnamese } from '../vietnameseSearch';

describe('vietnameseSearch', () => {
  it('folds Vietnamese diacritics and đ/Đ consistently', () => {
    expect(foldVietnamese('Chỉ định & Điều trị ĐẶC BIỆT')).toBe('chi dinh & dieu tri dac biet');
  });

  it('maps diacritic-insensitive matches back to original character offsets', () => {
    const original = 'Hướng dẫn Chỉ định liều đầu tiên';
    const mapped = buildTextMap(original);
    const query = foldVietnamese('chi dinh');
    const idx = mapped.normalized.indexOf(query);
    expect(idx).toBeGreaterThanOrEqual(0);
    const start = mapped.starts[idx];
    const end = mapped.ends[idx + query.length - 1];
    expect(original.slice(start, end)).toBe('Chỉ định');
  });
});
