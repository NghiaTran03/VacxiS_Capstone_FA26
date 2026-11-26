/**
 * Diacritic-insensitive Vietnamese text folding and character offset mapping.
 */

export function foldVietnamese(text) {
  return String(text || '')
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLocaleLowerCase('vi')
    .replace(/đ/g, 'd');
}

export function buildTextMap(text) {
  let normalized = '';
  const starts = [];
  const ends = [];
  let offset = 0;

  for (const character of text) {
    const folded = foldVietnamese(character);
    normalized += folded;
    for (let index = 0; index < folded.length; index += 1) {
      starts.push(offset);
      ends.push(offset + character.length);
    }
    if (!folded && ends.length > 0) {
      ends[ends.length - 1] = offset + character.length;
    }
    offset += character.length;
  }

  return { normalized, starts, ends };
}
