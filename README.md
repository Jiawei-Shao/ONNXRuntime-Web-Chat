# Build

- `npm install`
- Download Phi3 demo from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web and put them under `model/` folder.
  - Or change `kUseLocalModel` (in `onnx.js`) to `false` if you'd not like to host the models.
- `npm run build`

# Run

- For debug

  `npm run dev`

- For release

  `npm run preview`
