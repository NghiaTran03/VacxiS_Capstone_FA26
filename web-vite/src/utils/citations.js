/**
 * Transform known bracketed chunk IDs into safe sentinel citation links
 * without altering raw stored answers or executing untrusted HTML.
 */

export const CITATION_HREF_PREFIX = '#vacxis-citation-';

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripFallbackMarkup(text) {
  return String(text || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findProtectedLines(lines) {
  const protectedLines = new Set();
  let fenceChar = null;
  let fenceLength = 0;
  let inIndentedCode = false;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const lineNoNl = line.replace(/\r?\n$/, '');
    const trimmed = lineNoNl.trim();

    if (fenceChar) {
      protectedLines.add(i);
      const closeMatch = lineNoNl.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/);
      if (
        closeMatch &&
        closeMatch[1][0] === fenceChar &&
        closeMatch[1].length >= fenceLength
      ) {
        fenceChar = null;
        fenceLength = 0;
      }
      continue;
    }

    const openMatch = lineNoNl.match(/^ {0,3}(?:(`{3,})[^`]*$|(~{3,}).*$)/);
    if (openMatch) {
      const fence = openMatch[1] || openMatch[2];
      fenceChar = fence[0];
      fenceLength = fence.length;
      protectedLines.add(i);
      inIndentedCode = false;
      continue;
    }

    const prevBlank = i === 0 || lines[i - 1].trim() === '';
    const isIndented = /^(?: {4}|\t)/.test(line) && trimmed !== '';
    if (isIndented && (prevBlank || inIndentedCode)) {
      inIndentedCode = true;
      protectedLines.add(i);
      continue;
    }

    if (trimmed === '') {
      continue;
    }

    inIndentedCode = false;
    if (/^ {0,3}\[[^\]]+\]:/.test(line)) {
      protectedLines.add(i);
    }
  }

  return protectedLines;
}

/**
 * Build citation metadata and convert inline chunk IDs into `#vacxis-citation-N` links.
 * @param {string} answer
 * @param {Array<object>} sources
 * @returns {{ markdown: string, citationsByNumber: Record<number, object> }}
 */
export function prepareCitedMarkdown(answer, sources) {
  const rawAnswer = typeof answer === 'string' ? answer : '';
  if (!Array.isArray(sources) || sources.length === 0) {
    return { markdown: rawAnswer, citationsByNumber: {} };
  }

  const indexed = new Map();
  const orderedIds = [];
  for (const source of sources) {
    const cid = source?.chunk_id;
    if (!cid) continue;
    if (!indexed.has(cid)) {
      indexed.set(cid, source);
      orderedIds.push(cid);
    }
  }

  if (orderedIds.length === 0) {
    return { markdown: rawAnswer, citationsByNumber: {} };
  }

  const usedIds = orderedIds.filter((cid) => {
    const docId = indexed.get(cid)?.doc_id;
    return Boolean(docId) && docId !== 'lightrag_kg';
  });

  const citationsByNumber = {};
  const tokenById = new Map();

  orderedIds.forEach((cid, idx) => {
    const number = idx + 1;
    const source = indexed.get(cid);
    const title = source.doc_title || 'Tài liệu vắc xin';
    const header = source.header || 'Đoạn trích dẫn';
    const excerpt =
      typeof source.excerpt === 'string'
        ? source.excerpt
        : stripFallbackMarkup(source.text);

    let href = null;
    if (usedIds.includes(cid)) {
      const params = new URLSearchParams();
      params.set('doc', source.doc_id);
      params.set('chunk', cid);
      if (usedIds.length > 1) {
        for (const usedId of usedIds) {
          params.append('used', usedId);
        }
      }
      href = `/documents?${params.toString()}#citation`;
    }

    citationsByNumber[number] = {
      number,
      chunk_id: cid,
      doc_id: source.doc_id || '',
      title,
      header,
      excerpt,
      href,
    };
    tokenById.set(cid, `[${number}](${CITATION_HREF_PREFIX}${number})`);
  });

  const sortedIds = [...orderedIds].sort((a, b) => b.length - a.length);
  const idPattern = new RegExp(
    `(?<![\\w-])(?:${sortedIds.map(escapeRegex).join('|')})(?![\\w-])`,
    'g',
  );
  const referenced = new Set();

  const replaceBracket = (fullMatch, group1, group2) => {
    const groups = group2 !== undefined ? [group1, group2] : [group1];
    for (const group of groups) {
      idPattern.lastIndex = 0;
      if (!idPattern.test(group)) {
        return fullMatch;
      }
    }

    const replacements = [];
    for (const group of groups) {
      idPattern.lastIndex = 0;
      const seenInGroup = new Set();
      let match;
      while ((match = idPattern.exec(group)) !== null) {
        const cid = match[0];
        if (!seenInGroup.has(cid)) {
          seenInGroup.add(cid);
          referenced.add(cid);
          replacements.push(tokenById.get(cid));
        }
      }
      idPattern.lastIndex = 0;
      const unknown = group.replace(idPattern, '').replace(/^[ ,;|\t]+|[ ,;|\t]+$/g, '');
      if (unknown) {
        replacements.push(`[${unknown}]`);
      }
    }
    return replacements.join(' ');
  };

  const lines = rawAnswer.match(/[^\n]*\n|[^\n]+$/g) || (rawAnswer === '' ? [] : [rawAnswer]);
  const protectedLines = findProtectedLines(lines);

  const rendered = lines.map((line, index) => {
    if (protectedLines.has(index)) {
      return line;
    }
    const parts = line.split(/(`+[^`\n]*`+)/);
    return parts
      .map((part, offset) =>
        offset % 2 === 1
          ? part
          : part.replace(/\[([^\]\n]+)\](?:\[([^\]\n]*)\])?(?!\()/g, replaceBracket),
      )
      .join('');
  });

  let markdown = rendered.join('');
  const remaining = orderedIds
    .filter((cid) => !referenced.has(cid))
    .map((cid) => tokenById.get(cid));

  if (remaining.length > 0) {
    markdown += `\n\nNguồn: ${remaining.join(' ')}`;
  }

  return { markdown, citationsByNumber };
}
