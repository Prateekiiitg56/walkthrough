# Walkthrough

Walkthrough sorts a real estate photo shoot into listing order, in your browser.

Drop in the photos from a property shoot. Walkthrough:

- puts them in tour order: front of the house, living spaces, kitchen, bedrooms, bathrooms, then outside;
- sets aside photos that are too dark, out of focus, or almost the same as a better shot;
- renames the original files in order (`12-oak-street-01-front-exterior.jpg`) and downloads them as a zip with a `shot-list.csv`.

You can change any room, drag photos into a new order, pick a different cover, or put a set-aside photo back.

**Live:** https://prateekiiitg56.github.io/walkthrough/

Photos are never uploaded. The model runs in the browser tab.

## How it works

- **Model:** [CLIP ViT-B/32](https://huggingface.co/Xenova/clip-vit-base-patch32), open weights, ONNX.
- **Runtime:** [Transformers.js](https://github.com/huggingface/transformers.js) 3.7.6.
  - With WebGPU and 16-bit float support, the fp16 vision model runs on the graphics card. That is about 80 ms per photo and a 176 MB one-time download.
  - Otherwise, the 8-bit model runs on WebAssembly. That is slower, about 0.8 s per photo, with a 88 MB download.
- **Rooms:** each room type in [`rooms.js`](rooms.js) has one to three text prompts. [`scripts/embed-rooms.mjs`](scripts/embed-rooms.mjs) runs the CLIP text encoder once and saves the averaged vectors to `room-vectors.json`. The browser only needs the vision half of the model.
- **Room for a photo:** the cosine similarity between the photo's embedding and each room vector, then a softmax. Below 50%, the card shows "Check room".
- **Order:** rooms in listing order. Inside a room, the photo the model is surest about comes first.
- **Dark:** mean brightness below 0.30 on a 256 px grayscale copy.
- **Soft focus:** variance of the Laplacian below 8 on the same copy.
- **Same shot:** cosine similarity of 0.95 or more to a photo of the same room that is already in the listing.
- **Zip:** [JSZip](https://stuk.github.io/jszip/), with the original files stored as they are.
- **Offline:** a service worker caches the app and the runtime, and Transformers.js caches the weights. After the first run, it works with no connection.

No framework, no build step, no backend. It is static files on GitHub Pages.

## Run locally

```bash
python -m http.server 8000
```

Then open http://localhost:8000.

To change the room prompts, edit `rooms.js` and run:

```bash
npm install @huggingface/transformers@3.7.6
node scripts/embed-rooms.mjs
```

## License

Code: MIT. Sample photos: CC BY-SA 4.0. See [samples/CREDITS.md](samples/CREDITS.md).
