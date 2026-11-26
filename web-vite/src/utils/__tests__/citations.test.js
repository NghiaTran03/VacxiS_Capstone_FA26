import { describe, expect, it } from 'vitest';
import { CITATION_HREF_PREFIX, prepareCitedMarkdown } from '../citations';

function sampleSources() {
  return [
    {
      chunk_id: 'vax_c_0001',
      doc_id: 'vax',
      doc_title: 'Vaccine A',
      header: 'Liều dùng',
      text: 'First excerpt.',
      excerpt: 'First excerpt.',
    },
    {
      chunk_id: 'vax_c_0002',
      doc_id: 'vax',
      doc_title: 'Vaccine A',
      header: 'Tương tác',
      text: 'Second excerpt.',
      excerpt: 'Second excerpt.',
    },
  ];
}

describe('prepareCitedMarkdown', () => {
  it('repeats stable numbers for inline references and preserves markdown', () => {
    const answer = '**Heading**\n\nFirst [vax_c_0001]. Again [vax_c_0001]. Both [vax_c_0001, vax_c_0002].';
    const { markdown, citationsByNumber } = prepareCitedMarkdown(answer, sampleSources());
    expect(markdown.startsWith('**Heading**')).toBe(true);
    expect(markdown.split(`[1](${CITATION_HREF_PREFIX}1)`).length - 1).toBe(3);
    expect(markdown.split(`[2](${CITATION_HREF_PREFIX}2)`).length - 1).toBe(1);
    expect(markdown).not.toContain('Nguồn:');
    expect(citationsByNumber[1].href).toContain('doc=vax');
    expect(citationsByNumber[1].href).toContain('chunk=vax_c_0001');
    expect(citationsByNumber[1].href).toContain('used=vax_c_0001&used=vax_c_0002');
    expect(citationsByNumber[1].href.endsWith('#citation')).toBe(true);
  });

  it('does not rewrite inline code, fenced code, unknown IDs, or prefix IDs', () => {
    const answer = '`[vax_c_0001]`\n\n```text\n[vax_c_0002]\n```\n\n[unknown_c_0001] [vax_c_00010]';
    const { markdown } = prepareCitedMarkdown(answer, sampleSources());
    expect(markdown.startsWith(answer)).toBe(true);
    expect(markdown).toContain(`Nguồn: [1](${CITATION_HREF_PREFIX}1) [2](${CITATION_HREF_PREFIX}2)`);
  });

  it('keeps LightRAG knowledge graph sources without document links', () => {
    const source = { ...sampleSources()[0], chunk_id: 'lightrag_kg_context', doc_id: 'lightrag_kg' };
    const { markdown, citationsByNumber } = prepareCitedMarkdown('Graph [lightrag_kg_context]', [source]);
    expect(markdown).toContain(`[1](${CITATION_HREF_PREFIX}1)`);
    expect(citationsByNumber[1].href).toBeNull();
  });

  it('deduplicates repeated source entries', () => {
    const first = sampleSources()[0];
    const { markdown, citationsByNumber } = prepareCitedMarkdown('No inline citation.', [first, first]);
    expect(Object.keys(citationsByNumber)).toHaveLength(1);
    expect(markdown).toContain(`Nguồn: [1](${CITATION_HREF_PREFIX}1)`);
  });

  it('handles adjacent citations and preserves unknown IDs in mixed groups', () => {
    const answer = 'Adjacent [vax_c_0001][vax_c_0002]. Mixed [vax_c_0001, unknown_c_0003].';
    const { markdown } = prepareCitedMarkdown(answer, sampleSources());
    expect(markdown.split(`[1](${CITATION_HREF_PREFIX}1)`).length - 1).toBe(2);
    expect(markdown.split(`[2](${CITATION_HREF_PREFIX}2)`).length - 1).toBe(1);
    expect(markdown).toContain('[unknown_c_0003]');
    expect(markdown).not.toContain('Nguồn:');
  });

  it('leaves existing markdown links and reference links intact', () => {
    const answer = '[vax_c_0001](https://example.com) [label][vax_c_0002]\n\n[vax_c_0002]: https://example.com';
    const { markdown } = prepareCitedMarkdown(answer, sampleSources());
    expect(markdown.startsWith(answer)).toBe(true);
  });
});
