// The one search rule of the page's model lists: the Models tab's Hugging
// Face and BioImage Zoo lists and the Finetune tab's ranking. Every word
// typed is in the item's text, whatever the case. Spaces, commas, hyphens
// and underscores all separate words, in what is typed and in the text
// alike, so "mito aff", "mito-aff" and "mito_aff" all find
// mito-aff-unet-setup-16 (listed by the ranking as mito_aff_unet_setup_16),
// and "16 nm" reads as "16nm". Blank matches everything. The Hugging Face
// list matched what was typed as one string, so "mito aff" found nothing.

// An item's text to search, from its parts (the empty ones left out).
export function searchText(...parts) {
  return parts.filter((part) => part !== null && part !== undefined && part !== "").join(" ").toLowerCase().replace(/[_-]+/g, " ");
}

// Whether `text` (searchText's) has every word of `query`.
export function matchesWords(text, query) {
  const words = query.toLowerCase().replace(/(\d)\s+nm\b/g, "$1nm").split(/[\s,_-]+/).filter(Boolean);
  return words.every((word) => text.includes(word));
}

// "2d" for a model that reads one slice, "3d" for one that reads more, by
// its input shape (z first); null when the shape is not known. The cellmap
// exports' metadata says spatial_dims 3 for their 2D models too:
// setup-53-2d's input is 1 x 556 x 556.
export function dimsWord(inputShape) {
  return Array.isArray(inputShape) && inputShape.length ? (Number(inputShape[0]) === 1 ? "2d" : "3d") : null;
}
