import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CITATION_HREF_PREFIX, prepareCitedMarkdown } from '../utils/citations';
import CitationChip from './CitationChip';

export default function MarkdownAnswer({ content, sources = [] }) {
  const { markdown, citationsByNumber } = useMemo(
    () => prepareCitedMarkdown(content, sources),
    [content, sources],
  );

  const components = useMemo(
    () => ({
      a(linkProps) {
        const { href, children, node: _node, ...props } = linkProps;
        if (typeof href === 'string' && href.startsWith(CITATION_HREF_PREFIX)) {
          const num = Number(href.slice(CITATION_HREF_PREFIX.length));
          const citation = citationsByNumber[num];
          if (citation) {
            return <CitationChip citation={citation} />;
          }
        }
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {children}
          </a>
        );
      },
    }),
    [citationsByNumber],
  );

  return (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
