# AI Chest X-ray Disease Detection

A Streamlit web app for classifying chest X-ray diseases. No models are
bundled or predefined — every model comes from you, uploaded directly
through the sidebar. Upload as many `.keras` / `.h5` models as you like, run
predictions with a single model, or combine several into one ensemble
prediction.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Edit `class_names.txt` so it lists your models' output classes, one per
   line, in the exact order they were trained on (this order must match the
   models' output layer / softmax units). All models you upload should share
   this same class order.

3. Run the app:
   ```
   streamlit run app.py
   ```

   Note: the file upload limit (for both X-ray images and model uploads) is
   set to 2000 MB via `.streamlit/config.toml` (`server.maxUploadSize`).
   Adjust that number if your models are larger, or if you're on a hosting
   platform that enforces its own lower ceiling.

4. On first launch there are no models yet — the sidebar will prompt you to
   upload one under **"➕ Manage Custom Models"** before you can predict.

## Uploading your own models

In the sidebar, open **"➕ Manage Custom Models"**:
- Give the model a unique name.
- Upload the `.keras` or `.h5` file.
- Pick the preprocessing that matches how you trained it (simple `[0,1]`
  rescale, or ResNet50-style `preprocess_input`).
- Click **Add Model**.

Uploaded models are saved under `models/uploaded/`, and the mapping of
model name → file is written to `models/uploaded/registry.json`. This means
uploads are permanent: they survive page reloads, new browser sessions, and
full app restarts — not just the current session. There's no limit on how
many you can add. Use the 🗑️ button next to a model's name to remove it
(this deletes its saved file and its registry entry).

Note: this persistence is tied to the `models/uploaded/` folder on disk. If
you deploy to a host with ephemeral/wiped storage on every redeploy (some
free cloud tiers), uploads won't survive a redeploy — but they will survive
normal restarts, crashes, and new users opening the app.

## Prediction modes

- **Single Model** — run one uploaded model and see its prediction,
  confidence, and probability breakdown.
- **Ensemble (Multiple Models)** — select two or more uploaded models; the
  app runs all of them, averages their probability outputs (soft voting),
  and shows the combined "Ensemble (Average)" result alongside each
  individual model's result and a comparison chart. Models whose number of
  output classes doesn't match the others are automatically excluded from
  the average, with a warning shown.

Both options only appear once at least one model has been uploaded.

## Project Structure

```
app.py
.streamlit/
    config.toml          # forces a light theme (white + deep emerald), raises upload limit
models/
    uploaded/             # created automatically for user-uploaded models
        registry.json     # persisted mapping of uploaded model name -> file
assets/
class_names.txt
requirements.txt
```

## Notes

- Images are converted to RGB, resized to 224x224, and normalized before
  inference (method depends on each model's chosen preprocessing).
- Models are cached with `@st.cache_resource`, keyed by file path + last
  modified time, so they load once and correctly reload if a file is
  replaced.
- This tool is intended for research/educational purposes only and is not
  a certified diagnostic tool.

Developed by Mahbubul Islam
