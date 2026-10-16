export const MODEL = 'Xenova/clip-vit-base-patch32';
export const promptText = (p) => `a real estate photo of ${p}`;

// Room types in listing order. A listing opens on the front of the house,
// walks through the living spaces, then the private rooms, then outside.
// Each room has a few text prompts; CLIP scores a photo against all of them.
export const ROOMS = [
  { id: 'front', name: 'Front exterior', prompts: ['the front of a house seen from the street', 'the front exterior of a house with a driveway'] },
  { id: 'living', name: 'Living room', prompts: ['a living room with a sofa', 'a lounge room with couches and a rug'] },
  { id: 'kitchen', name: 'Kitchen', prompts: ['a kitchen with cabinets and a countertop', 'a kitchen with an oven and a sink', 'an open plan kitchen with an island bench'] },
  { id: 'dining', name: 'Dining', prompts: ['a dining room with a dining table and chairs'] },
  { id: 'bedroom', name: 'Bedroom', prompts: ['a bedroom with a bed', 'a bedroom with a made bed and pillows'] },
  { id: 'bathroom', name: 'Bathroom', prompts: ['a bathroom with a sink and a toilet', 'a bathroom with a shower', 'a bathroom vanity and mirror'] },
  { id: 'office', name: 'Office', prompts: ['a home office with a desk and chair'] },
  { id: 'laundry', name: 'Laundry', prompts: ['a laundry room with a washing machine'] },
  { id: 'hall', name: 'Hall and stairs', prompts: ['a hallway inside a house', 'a staircase inside a house'] },
  { id: 'garage', name: 'Garage', prompts: ['the inside of a garage', 'a garage door and driveway'] },
  { id: 'balcony', name: 'Balcony', prompts: ['a balcony with a view', 'a view from a window over the city'] },
  { id: 'backyard', name: 'Backyard', prompts: ['the backyard of a house with a lawn', 'an outdoor patio with furniture'] },
  { id: 'pool', name: 'Pool', prompts: ['a swimming pool in a backyard'] },
  { id: 'floorplan', name: 'Floor plan', prompts: ['a floor plan drawing of a house'] },
];

export const roomById = new Map(ROOMS.map((r) => [r.id, r]));

export function slug(text) {
  return text.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export function normalize(vec) {
  let n = 0;
  for (const x of vec) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(vec, (x) => x / n);
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Text embeddings come in prompt order; average them per room.
export function roomVectors(textEmbeds) {
  let i = 0;
  return ROOMS.map((room) => {
    const sum = new Float32Array(textEmbeds[0].length);
    for (const _ of room.prompts) {
      const v = textEmbeds[i++];
      for (let k = 0; k < sum.length; k++) sum[k] += v[k];
    }
    return normalize(sum);
  });
}

// Softmax over CLIP similarities (CLIP's logit scale is 100).
export function classify(imageVec, roomVecs) {
  const logits = roomVecs.map((r) => 100 * dot(imageVec, r));
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return ROOMS.map((room, i) => ({ id: room.id, p: exps[i] / total }))
    .sort((a, b) => b.p - a.p);
}

// Brightness and sharpness from a grayscale image (values 0..1).
// Sharpness is the variance of the Laplacian: blurry photos have weak edges.
export function measure(gray, width, height) {
  let sum = 0;
  for (const g of gray) sum += g;
  const brightness = sum / gray.length;
  let lsum = 0, lsq = 0, n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const l = gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width] - 4 * gray[i];
      lsum += l; lsq += l * l; n++;
    }
  }
  const mean = lsum / n;
  const sharpness = (lsq / n - mean * mean) * 1000;
  return { brightness, sharpness };
}

export const DARK = 0.3;
export const SOFT = 8;
export const SAME_SHOT = 0.95;

// Why a photo should stay out of the listing, or null if it is fine.
// Dark and soft come from the pixels. A duplicate is a photo that is almost
// the same shot as a better one of the same room.
function problem(p, keptSameRoom) {
  if (p.brightness < DARK) return { kind: 'dark' };
  if (p.sharpness < SOFT) return { kind: 'soft' };
  const twin = keptSameRoom.find((k) => dot(k.vec, p.vec) >= SAME_SHOT);
  return twin ? { kind: 'dupe', of: twin.id } : null;
}

// Put photos in listing order: rooms in ROOMS order, and inside a room the
// photo the model is surest about goes first. Returns the photos for the
// listing and the ones set aside. A photo the user kept or set aside by hand
// (p.pinned) stays where they put it.
export function arrange(photos) {
  const order = new Map(ROOMS.map((r, i) => [r.id, i]));
  const sorted = [...photos].sort((a, b) => order.get(a.room) - order.get(b.room) || b.confidence - a.confidence);
  const listing = [], aside = [];
  for (const p of sorted) {
    p.problem = problem(p, listing.filter((k) => k.room === p.room));
    const keep = p.pinned ? p.pinned === 'keep' : !p.problem;
    (keep ? listing : aside).push(p);
  }
  return { listing, aside };
}

export function fileNames(photos, prefix) {
  const counts = new Map();
  const base = slug(prefix || '');
  const width = String(photos.length).length < 2 ? 2 : String(photos.length).length;
  return photos.map((p, i) => {
    const name = slug(roomById.get(p.room).name);
    const n = (counts.get(name) || 0) + 1;
    counts.set(name, n);
    const ext = (p.file.name.match(/\.[a-z0-9]+$/i)?.[0] || '.jpg').toLowerCase();
    const num = String(i + 1).padStart(width, '0');
    return [base, num, name + (n > 1 ? `-${n}` : '')].filter(Boolean).join('-') + ext;
  });
}
