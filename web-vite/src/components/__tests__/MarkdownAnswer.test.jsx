import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import MarkdownAnswer from '../MarkdownAnswer';

describe('MarkdownAnswer and CitationChip', () => {
  it('renders safe citation chips with hover/focus tooltip and blocks raw HTML scripts', () => {
    const sources = [
      {
        chunk_id: 'vax_c_0001',
        doc_id: 'vax_a',
        doc_title: 'Vaccine A <img onerror=alert(1)>',
        header: 'Chỉ định',
        text: 'Raw excerpt',
        excerpt: 'Plain excerpt text',
      },
      {
        chunk_id: 'kg_1',
        doc_id: 'lightrag_kg',
        doc_title: 'Đồ thị tri thức',
        header: 'Liên kết',
        text: 'Graph node',
        excerpt: 'Graph node',
      },
    ];

    const { container } = render(
      <MarkdownAnswer
        content="Thông tin [vax_c_0001] và [kg_1] <script>alert(1)</script>"
        sources={sources}
      />,
    );

    expect(container.querySelector('script')).toBeNull();

    const docChip = screen.getByRole('link', { name: /Nguồn 1:/ });
    expect(docChip.getAttribute('href')).toBe('/documents?doc=vax_a&chunk=vax_c_0001#citation');

    fireEvent.focus(docChip);
    expect(screen.getByRole('tooltip')).toBeTruthy();
    expect(screen.getByText('Mở tài liệu ↗')).toBeTruthy();
    expect(screen.getByText('Plain excerpt text')).toBeTruthy();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('tooltip')).toBeNull();

    const kgChip = screen.getByLabelText(/Nguồn 2:/);
    expect(kgChip.tagName).toBe('SPAN');
    fireEvent.focus(kgChip);
    expect(screen.getByText('Ngữ cảnh đồ thị tri thức')).toBeTruthy();
  });
});
