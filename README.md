## SignLanguage

SignLanguage is a privacy-first ASL learning platform that combines a large phrase dictionary with on-device computer vision for interactive practice. The web client is a PWA powered by Eleventy, Tailwind, and Flowbite; ASL recognition runs in-browser using MediaPipe Hands and TensorFlow.js; model training pipelines are provided in Python with TensorFlow/TFLite and Jupyter.


---

### Key Features
- ASL Phrase Dictionary: ~20k curated videos with lightning-fast fuzzy search via MongoDB Atlas Search (Realm HTTPS endpoints).
- AI-Powered Practice: Numbers (0–10) and Alphabet (A–Z) games/quizzes using MediaPipe Hands + TF.js for real-time inference in the browser.
- Progressive Curriculum + Games: Learn and validate with multiple modes (alphabet ranges, numbers, quizzes, balloon popper, memory, space shooter).
- Offline-First PWA: Stale-while-revalidate caching, installable manifest, resilient CDN fallbacks, responsive UI.
- Training Pipeline: Python-based data collection, labeled CSVs, notebooks for retraining MLP classifiers; export to TFLite, mirrored as TF.js for client inference.

---

### System Design (High-Level)
- Client-first architecture: all webcam processing and inference happen in the browser for low latency and privacy.
- Content and search are decoupled from inference. Phrase search uses MongoDB Atlas Search via Realm HTTPS endpoints; games and ASL recognition run entirely locally.
- Build and deploy as static assets; service worker handles offline cache lifecycle.

```
┌──────────────┐   webcam   ┌──────────────────┐   21×2 landmarks   ┌──────────────────┐
│   Browser    │──────────▶│  MediaPipe Hands  │───────────────────▶│  Feature Engine  │
│  (PWA+UI)    │           └──────────────────┘                     │  (normalize+vec) │
│ Eleventy UI  │                                                     └───────┬─────────┘
│ Tailwind     │                                             1×N vector     │
│ Flowbite     │                                                           ┌─▼─────────────┐
└──────┬───────┘                                                           │ TF.js MLP     │
       │                                                                    │ Classifier    │
       │                                                                    └─────┬────────┘
       │                                                                 predicted class
       │
       │                    ┌─────────────────────────────┐
       └───────────────────▶│ MongoDB Realm + AtlasSearch │◀─── query terms
                            │ Phrases, metadata, videos   │
                            └─────────────────────────────┘
```

---

### Architecture
- Frontend (Eleventy/Nunjucks)
  - Static site generation with componentized templates.
  - Tailwind + Flowbite + Alpine.js for responsive, lightweight interactivity.
  - PWA service worker (stale-while-revalidate) and manifest for installability.
- Dictionary Backend (Managed)
  - MongoDB Realm HTTPS endpoints and Atlas Search for fuzzy lookup across phrase metadata.
  - Static JSON content sources for curated lists and assets.
- ML Inference (Client)
  - MediaPipe Hands for landmarks → normalized features → MLP classifier (TF.js).
  - On-device inference (no webcam frames leave the device).
- ML Training (Python)
  - Notebooks and scripts for data collection, labeling, training, evaluation.
  - TensorFlow/TFLite for fast inference targets; reproducible requirements file.

---

### Client-Side Inference Pipeline (Technical)
- Landmark acquisition: MediaPipe Hands (single hand) returns 21 keypoints. For each frame, points are mapped to pixel space.
- Normalization:
  - Translate to wrist-relative coordinates (subtract base point).
  - Flatten to 1D vector; scale by max absolute value for scale invariance.
- Model input:
  - Static pose (numbers/letters): 42-d vector (21 keypoints × 2 coords).
  - Gesture history (when enabled in Python demo): 32-d vector (16 frames × 2 coords).
- Classification: TF.js multilayer perceptron outputs class logits; argmax selects the predicted class. App code maps classes to digits/letters and updates game state and scoring.
- Performance: canvas-based rendering overlays (skeleton, connectors). Stays GPU-friendly; sustained 30+ FPS on modern laptops.

---

### ML Training Details (Python)
- Frameworks: TensorFlow 2.x + scikit-learn utilities, TFLite for optimized inference.
- Data collection: real-time capture with key labels; CSV logs for landmarks and point histories.
- Feature extraction: same normalization strategy as client to ensure train/infer parity.
- Architectures: compact MLPs for low-latency inference; outputs cover digits 0–9 and alphabet ranges based on the task.
- Reproducibility: `ml_model/requirements.txt` pins core dependencies (e.g., mediapipe 0.10.5, tensorflow 2.3.0, OpenCV 4.5.3).

---

### Project Structure (abridged)

```bash
Signify/
├── frontend/                     # Web client (Eleventy)
│   ├── app/                      # Routes/pages (games, quizzes, lessons)
│   ├── assets/                   # Images, icons, ASL assets
│   ├── js/                       # Client JS (mediapipe, tfjs games, PWA)
│   ├── _includes/                # Nunjucks components/layouts/sections
│   ├── _site/                    # Eleventy build output
│   ├── tailwind.config.js        # Tailwind setup (with Flowbite)
│   ├── package.json              # Eleventy/Tailwind scripts
│   └── manifest.json             # PWA manifest
├── ml_model/                     # Python training pipeline
│   ├── app.py                    # Real-time demo / data collection
│   ├── requirements.txt          # Reproducible Python deps
│   ├── model/                    # Classifiers, labels, artifacts
│   └── *.ipynb                   # Jupyter notebooks for training
└── README.md                     # This file
```

---

### Frontend: Run Locally

```bash
cd frontend

# First time
npm i
# Dev server (Eleventy) — terminal 1
npm start
# Tailwind watcher to emit CSS into _site — terminal 2
npm run css

# Production build
npm run build
```

Notes:
- Both the Eleventy dev server and Tailwind watcher should run in parallel during development.
- Build output is in `frontend/_site/`.

---

### Deployment
- Artifact: deploy `frontend/_site/` to any static host (e.g., GitHub Pages, Netlify, Vercel static).
- Caching: service worker uses stale-while-revalidate for same-origin plus host whitelist (`fonts.gstatic.com`, `fonts.googleapis.com`, `cdn.jsdelivr.net`).
- HTTPS: required for `getUserMedia` (camera) outside localhost.
- CDN: MediaPipe and model files can be served from a CDN; ensure CORS and cache headers are set appropriately.

---

### ML Model: Training & Demo (Python)

Prereqs: Python 3.8+, webcam.

```bash
cd ml_model
python -m venv signify_env
source signify_env/bin/activate  # Windows: signify_env\Scripts\activate
pip install -r requirements.txt

# Run real-time demo / data collection
python app.py
```

Training:
- Use `keypoint_classification.ipynb` (hand sign) and `point_history_classification.ipynb` (gesture) for retraining.
- Update label CSVs and class counts, run the notebook, export to TFLite/TF.js.
- Client-side inference loads models via TF.js for instant, on-device predictions.

---

### PWA & Performance
- Service worker implements stale-while-revalidate with hostname whitelisting and cache-busting for origin requests.
- Manifest enables installability; responsive UI via Tailwind + Flowbite; Alpine.js for lightweight interactivity.
- MediaPipe + TF.js pipeline sustains 30+ FPS and low-latency inference on modern hardware.

---

### Security & Privacy
- No webcam frames leave the device during inference; only vectorized keypoints are used locally.
- Phrase search requests hit Realm HTTPS endpoints; avoid including user-identifying data.
- Follow HTTPS for all deployments; restrict external resource domains to the SW whitelist.

---

### Tech Stack
- Frontend: Eleventy (Nunjucks), Tailwind CSS, Flowbite, Alpine.js, PWA (Service Worker + Manifest)
- ML Inference: MediaPipe Hands, TensorFlow.js
- ML Training: Python, TensorFlow/TFLite, OpenCV, scikit-learn, Jupyter
- Search/Content: MongoDB Atlas Search, MongoDB Realm HTTPS endpoints

Version highlights (where applicable):
- Python 3.8+, TensorFlow 2.3.0, mediapipe 0.10.5, OpenCV 4.5.3 (training)
- Tailwind CSS 3.x, Flowbite 1.x, Eleventy (latest), Alpine.js 3.x (frontend)

---

### FYP Demo Checklist (Presenter’s Guide)
- Show installability and offline behavior: add to homescreen, reload without network.
- Demonstrate phrase search latency and fuzziness with misspellings and partials.
- Walk through numbers quiz: show live webcam overlay, correct/incorrect predictions, scoring and confetti thresholds.
- Inspect diagnostics: FPS stability, quick reconnect if camera toggled.
- Explain model parity: same normalization in training and client; highlight on-device privacy.
- Optional: swap model URL/version to demonstrate hot model upgrade without server changes.

---

### License
MIT

---

### Acknowledgments
- MediaPipe and TensorFlow communities for foundational CV/ML tooling.
- Based in part on work by Kazuhito Takahashi for landmark pipelines and classification structure.