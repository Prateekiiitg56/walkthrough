// App shell, runtime and zip tool are cached here so the app opens with no connection.
// Model weights are cached separately by Transformers.js in the "transformers-cache" store.
const CACHE = 'walkthrough-v2';
const SHELL = ['./', './index.html', './styles.css', './app.js', './rooms.js', './room-vectors.json', './manifest.webmanifest', './icon.svg'];
const CDN = ['cdn.jsdelivr.net', 'cdnjs.cloudflare.com', 'fonts.googleapis.com', 'fonts.gstatic.com'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('walkthrough-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  const isShell = url.origin === location.origin;
  if (!isShell && !CDN.includes(url.hostname)) return;

  if (isShell) {
    // Network first for our own files so updates land; cache when offline.
    event.respondWith(
      fetch(event.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(event.request, copy));
          return res;
        })
        .catch(() => caches.match(event.request, { ignoreSearch: true })),
    );
    return;
  }

  // Versioned CDN files never change: cache first.
  event.respondWith(
    caches.match(event.request).then((hit) => hit || fetch(event.request).then((res) => {
      if (res.ok || res.type === 'opaque') {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy));
      }
      return res;
    })),
  );
});
