import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';
import { MODEL, ROOMS, roomById, classify, measure, arrange, normalize, fileNames, slug } from './rooms.js';

env.allowLocalModels = false;

const THUMB = 640;
const SAMPLES = Array.from({ length: 14 }, (_, i) => `IMG_${4012 + i}.jpg`);

const $ = (id) => document.getElementById(id);
const els = {
  start: $('start'), work: $('work'), drop: $('drop'), input: $('file-input'), sample: $('sample-btn'),
  status: $('status'), statusText: $('status-text'), statusBar: $('status-bar'),
  prefix: $('prefix'), counts: $('counts'), add: $('add-btn'), resort: $('resort-btn'), zip: $('zip-btn'),
  listing: $('listing'), aside: $('aside'), asideWrap: $('aside-wrap'), clear: $('clear-btn'), veil: $('drop-veil'), engine: $('engine'),
};

const state = { photos: new Map(), listing: [], aside: [], busy: false };
let modelPromise = null;

// ---------- model ----------

// The graphics card is about ten times faster than WebAssembly here, so use it
// when the browser offers WebGPU with 16-bit floats. Otherwise run the 8-bit
// model on the processor.
async function pickDevice() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter?.features.has('shader-f16')) return { device: 'webgpu', dtype: 'fp16' };
  } catch { /* no WebGPU */ }
  return { device: 'wasm', dtype: 'q8' };
}

function loadModel() {
  if (modelPromise) return modelPromise;
  const files = new Map();
  const progress_callback = (e) => {
    if (e.status !== 'progress' || !e.total) return;
    files.set(e.file, e);
    let loaded = 0, total = 0;
    for (const f of files.values()) { loaded += f.loaded; total += f.total; }
    showStatus(`Downloading the open model, one time only: ${Math.round((loaded / total) * 100)}% of ${Math.round(total / 1e6)} MB`, loaded / total);
  };
  modelPromise = (async () => {
    // Room prompts are embedded ahead of time (scripts/embed-rooms.mjs), so only the vision half of CLIP is needed here.
    const vectors = await (await fetch('room-vectors.json')).json();
    if (vectors.model !== MODEL || ROOMS.some((r) => !vectors.rooms[r.id])) throw new Error('room-vectors.json is out of date');
    const rooms = ROOMS.map((r) => Float32Array.from(vectors.rooms[r.id]));
    const processor = await AutoProcessor.from_pretrained(MODEL, { progress_callback });
    let choice = await pickDevice();
    let vision;
    try {
      vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { ...choice, progress_callback });
    } catch (err) {
      if (choice.device === 'wasm') throw err;
      console.warn('WebGPU failed, using WebAssembly', err);
      choice = { device: 'wasm', dtype: 'q8' };
      vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { ...choice, progress_callback });
    }
    els.engine.textContent = choice.device === 'webgpu' ? 'Running on your graphics card.' : 'Running on your processor.';
    return { processor, vision, rooms };
  })();
  modelPromise.catch(() => { modelPromise = null; });
  return modelPromise;
}

// ---------- reading photos ----------

async function analyse(file, model) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, THUMB / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  const { image_embeds } = await model.vision(await model.processor(RawImage.fromCanvas(canvas)));
  const vec = normalize(image_embeds.data);
  const [best] = classify(vec, model.rooms);

  // Brightness and sharpness on a fixed 256 px width, so the numbers mean the same for every photo.
  const small = document.createElement('canvas');
  small.width = 256;
  small.height = Math.max(2, Math.round((256 * canvas.height) / canvas.width));
  const sctx = small.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(canvas, 0, 0, small.width, small.height);
  const px = sctx.getImageData(0, 0, small.width, small.height).data;
  const gray = new Float32Array(small.width * small.height);
  for (let i = 0; i < gray.length; i++) gray[i] = (0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255;
  const { brightness, sharpness } = measure(gray, small.width, small.height);

  const thumb = URL.createObjectURL(await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.82)));
  return { id: crypto.randomUUID(), file, thumb, vec, room: best.id, confidence: best.p, brightness, sharpness, pinned: null, problem: null, roomByUser: false };
}

async function addFiles(fileList) {
  const files = [...fileList].filter((f) => /^image\/(jpeg|png|webp)$/.test(f.type));
  if (!files.length || state.busy) return;
  state.busy = true;
  setBusy(true);
  try {
    showStatus('Loading the open model…', 0);
    const model = await loadModel();
    for (let i = 0; i < files.length; i++) {
      showStatus(`Looking at photo ${i + 1} of ${files.length}`, i / files.length);
      try {
        const p = await analyse(files[i], model);
        state.photos.set(p.id, p);
      } catch (err) {
        console.warn('Could not read', files[i].name, err);
      }
    }
    sortAll();
    hideStatus();
  } catch (err) {
    console.error(err);
    showStatus('The model did not load. Check the connection and try again. It only needs the internet the first time.', 0);
  } finally {
    state.busy = false;
    setBusy(false);
  }
}

async function loadSamples() {
  showStatus('Fetching sample photos…', 0);
  const files = await Promise.all(SAMPLES.map(async (name) => {
    const blob = await (await fetch(`samples/${name}`)).blob();
    return new File([blob], name, { type: 'image/jpeg' });
  }));
  if (!els.prefix.value) els.prefix.value = '12 Oak Street';
  await addFiles(files);
}

// ---------- order ----------

function sortAll() {
  const { listing, aside } = arrange([...state.photos.values()]);
  state.listing = listing.map((p) => p.id);
  state.aside = aside.map((p) => p.id);
  render();
}

// Put a photo back after the last photo of the same room, or where its room would start.
function insertByRoom(id) {
  const rank = new Map(ROOMS.map((r, i) => [r.id, i]));
  const mine = rank.get(state.photos.get(id).room);
  let at = state.listing.length;
  for (let i = state.listing.length - 1; i >= 0; i--) {
    if (rank.get(state.photos.get(state.listing[i]).room) <= mine) { at = i + 1; break; }
    at = i;
  }
  state.listing.splice(at, 0, id);
}

function move(id, to) {
  const from = state.listing.indexOf(id);
  if (from < 0 || to < 0 || to >= state.listing.length) return;
  state.listing.splice(from, 1);
  state.listing.splice(to, 0, id);
  render();
  els.listing.querySelector(`[data-id="${id}"] [data-act="${to < from ? 'up' : 'down'}"]`)?.focus();
}

function setAside(id) {
  state.listing = state.listing.filter((x) => x !== id);
  state.aside.unshift(id);
  state.photos.get(id).pinned = 'aside';
  render();
}

function putBack(id) {
  state.aside = state.aside.filter((x) => x !== id);
  state.photos.get(id).pinned = 'keep';
  insertByRoom(id);
  render();
}

// ---------- render ----------

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

function reason(p) {
  const pr = p.problem;
  if (p.pinned === 'aside') return 'Set aside by you';
  if (!pr) return '';
  if (pr.kind === 'dark') return 'Too dark';
  if (pr.kind === 'soft') return 'Soft focus';
  const twin = state.listing.indexOf(pr.of);
  return twin >= 0 ? `Same shot as ${String(twin + 1).padStart(2, '0')}` : 'Repeated shot';
}

const roomOptions = (current) => ROOMS.map((r) => `<option value="${r.id}"${r.id === current ? ' selected' : ''}>${r.name}</option>`).join('');

function card(p, i, name, inListing) {
  const num = inListing ? String(i + 1).padStart(2, '0') : '';
  const sure = p.roomByUser || p.confidence >= 0.5 ? '' : '<span class="unsure" title="The model was not sure about this room">Check room</span>';
  const note = inListing ? '' : `<p class="why">${esc(reason(p))}</p>`;
  const controls = inListing
    ? `<button type="button" class="icon" data-act="up" aria-label="Move earlier"${i === 0 ? ' disabled' : ''}>&larr;</button>
       <button type="button" class="icon" data-act="down" aria-label="Move later"${i === state.listing.length - 1 ? ' disabled' : ''}>&rarr;</button>
       ${i === 0 ? '<span class="cover">Cover</span>' : '<button type="button" class="mini" data-act="cover">Make cover</button>'}
       <button type="button" class="mini" data-act="aside">Set aside</button>`
    : '<button type="button" class="mini" data-act="back">Put back</button>';
  return `<li class="card" data-id="${p.id}"${inListing ? ' draggable="true"' : ''}>
    <div class="frame">
      <img src="${p.thumb}" alt="" loading="lazy" draggable="false">
      ${num ? `<span class="num">${num}</span>` : ''}
    </div>
    <div class="body">
      <div class="room-row">
        <select data-act="room" aria-label="Room for ${esc(p.file.name)}">${roomOptions(p.room)}</select>${sure}
      </div>
      <p class="fname" title="${esc(p.file.name)}">${inListing ? esc(name) : esc(p.file.name)}</p>
      ${note}
      <div class="ctrls">${controls}</div>
    </div>
  </li>`;
}

function render() {
  const listing = state.listing.map((id) => state.photos.get(id));
  const aside = state.aside.map((id) => state.photos.get(id));
  const names = fileNames(listing, els.prefix.value);
  els.listing.innerHTML = listing.map((p, i) => card(p, i, names[i], true)).join('');
  els.aside.innerHTML = aside.map((p, i) => card(p, i, '', false)).join('');
  els.asideWrap.hidden = aside.length === 0;
  const has = state.photos.size > 0;
  els.work.hidden = !has;
  els.start.classList.toggle('compact', has);
  els.counts.textContent = has
    ? `${listing.length} in the listing${aside.length ? `, ${aside.length} set aside` : ''}`
    : '';
  els.zip.disabled = listing.length === 0;
}

// ---------- export ----------

function csvCell(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

async function downloadZip() {
  if (!window.JSZip) { alert('The zip tool is still loading. Try again in a moment.'); return; }
  const listing = state.listing.map((id) => state.photos.get(id));
  const aside = state.aside.map((id) => state.photos.get(id));
  const names = fileNames(listing, els.prefix.value);
  const zip = new window.JSZip();
  const rows = [['order', 'file', 'original', 'room', 'room_set_by', 'note']];
  const by = (p) => (p.roomByUser ? 'you' : `model ${Math.round(p.confidence * 100)}%`);
  listing.forEach((p, i) => {
    zip.file(names[i], p.file);
    rows.push([i + 1, names[i], p.file.name, roomById.get(p.room).name, by(p), i === 0 ? 'cover' : '']);
  });
  aside.forEach((p) => rows.push(['', '', p.file.name, roomById.get(p.room).name, by(p), `set aside: ${reason(p).toLowerCase()}`]));
  zip.file('shot-list.csv', rows.map((r) => r.map(csvCell).join(',')).join('\n') + '\n');

  els.zip.disabled = true;
  els.zip.textContent = 'Packing…';
  try {
    // Photos are already compressed, so store them as they are. It is much faster.
    const blob = await zip.generateAsync({ type: 'blob', compression: 'STORE' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${slug(els.prefix.value) || 'listing'}-photos.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  } finally {
    els.zip.disabled = false;
    els.zip.textContent = 'Download zip';
  }
}

// ---------- status ----------

function showStatus(text, fraction) {
  els.status.hidden = false;
  els.statusText.textContent = text;
  els.statusBar.value = Math.round(fraction * 100);
}
function hideStatus() { els.status.hidden = true; }
function setBusy(on) {
  for (const b of [els.add, els.resort, els.zip, els.sample]) b.disabled = on;
  els.drop.classList.toggle('busy', on);
  if (!on) els.zip.disabled = state.listing.length === 0;
}

// ---------- events ----------

els.input.addEventListener('change', () => { addFiles(els.input.files); els.input.value = ''; });
els.add.addEventListener('click', () => els.input.click());
els.sample.addEventListener('click', loadSamples);
// Re-sort with the rooms the user fixed. Photos they kept or set aside by hand stay that way.
els.resort.addEventListener('click', sortAll);
els.zip.addEventListener('click', downloadZip);
els.prefix.addEventListener('input', render);
els.clear.addEventListener('click', () => {
  if (!confirm('Clear these photos and start a new shoot? Your files on disk are not touched.')) return;
  for (const p of state.photos.values()) URL.revokeObjectURL(p.thumb);
  state.photos.clear();
  state.listing = [];
  state.aside = [];
  els.prefix.value = '';
  render();
  window.scrollTo({ top: 0 });
});

for (const list of [els.listing, els.aside]) {
  list.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-act]');
    if (!btn) return;
    const id = btn.closest('.card').dataset.id;
    const i = state.listing.indexOf(id);
    ({
      up: () => move(id, i - 1),
      down: () => move(id, i + 1),
      cover: () => move(id, 0),
      aside: () => setAside(id),
      back: () => putBack(id),
    })[btn.dataset.act]?.();
  });
  list.addEventListener('change', (e) => {
    if (e.target.dataset.act !== 'room') return;
    const p = state.photos.get(e.target.closest('.card').dataset.id);
    p.room = e.target.value;
    p.roomByUser = true;
    render();
  });
}

// Drag to reorder inside the listing.
let dragId = null;
els.listing.addEventListener('dragstart', (e) => {
  const li = e.target.closest('.card');
  if (!li) return;
  dragId = li.dataset.id;
  li.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', dragId);
});
els.listing.addEventListener('dragend', () => {
  dragId = null;
  els.listing.querySelectorAll('.dragging, .over').forEach((n) => n.classList.remove('dragging', 'over'));
});
els.listing.addEventListener('dragover', (e) => {
  if (!dragId) return;
  e.preventDefault();
  const li = e.target.closest('.card');
  els.listing.querySelectorAll('.over').forEach((n) => n !== li && n.classList.remove('over'));
  if (li && li.dataset.id !== dragId) li.classList.add('over');
});
els.listing.addEventListener('drop', (e) => {
  const li = e.target.closest('.card');
  if (!dragId || !li) return;
  e.preventDefault();
  e.stopPropagation();
  move(dragId, state.listing.indexOf(li.dataset.id));
});

// Dropping files anywhere on the page adds them.
let depth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; els.veil.hidden = false; } });
window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; els.veil.hidden = true; } });
window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  depth = 0;
  els.veil.hidden = true;
  addFiles(e.dataTransfer.files);
});

render();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
