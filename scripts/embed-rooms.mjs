// Run after changing room prompts in rooms.js:
//   npm install @huggingface/transformers@3.7.6 && node scripts/embed-rooms.mjs
// The text side of CLIP only ever sees these fixed prompts, so we embed them
// once here and ship the vectors. The browser then only downloads the vision half.
import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers';
import { writeFileSync } from 'node:fs';
import { ROOMS, MODEL, promptText, roomVectors } from '../rooms.js';

const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const text = await CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: 'fp32' });
const prompts = ROOMS.flatMap((r) => r.prompts.map(promptText));
const { text_embeds } = await text(tokenizer(prompts, { padding: true, truncation: true }));
const d = text_embeds.dims[1];
const vectors = roomVectors(prompts.map((_, i) => text_embeds.data.slice(i * d, (i + 1) * d)));
const out = { model: MODEL, rooms: Object.fromEntries(ROOMS.map((r, i) => [r.id, Array.from(vectors[i], (x) => +x.toFixed(5))])) };
writeFileSync(new URL('../room-vectors.json', import.meta.url), JSON.stringify(out));
console.log(`Wrote ${ROOMS.length} room vectors from ${prompts.length} prompts.`);
