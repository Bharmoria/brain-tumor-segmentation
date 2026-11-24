import base64
import io
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

import nibabel as nib
import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from starlette.background import BackgroundTask
from tensorflow.keras import Model, layers

# ---------------------------------------------------------------- config
MODEL_PATH = os.environ.get("MODEL_PATH", "model.weights.h5")
BASE_FILTERS = int(os.environ.get("BASE_FILTERS", "16"))   # must match training
API_KEY = os.environ.get("API_KEY")
ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

NUM_CLASSES = 4
MODALITIES = ["flair", "t1", "t1ce", "t2"]                 # same channel order as training
CROP = (slice(56, 184), slice(56, 184), slice(13, 141))     # same crop as training
FULL_SHAPE = (240, 240, 155)                                # BraTS volume size
COLORS = {1: (0, 120, 255), 2: (0, 200, 0), 4: (255, 0, 0)}  # BraTS label -> RGB
LEGEND = {"necrotic_core": "blue", "edema": "green", "enhancing_tumor": "red"}

state = {}


# ---------------------------------------------------------------- model
def build_unet3d(input_shape=(128, 128, 128, 4), num_classes=NUM_CLASSES, base=16):
    """Identical to the architecture used for training."""
    def block(x, f):
        for _ in range(2):
            x = layers.Conv3D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        return x

    inp = layers.Input(input_shape)
    # ENCODER
    c1 = block(inp, base);      p1 = layers.MaxPool3D()(c1)
    c2 = block(p1, base * 2);   p2 = layers.MaxPool3D()(c2)
    c3 = block(p2, base * 4);   p3 = layers.MaxPool3D()(c3)
    c4 = block(p3, base * 8);   p4 = layers.MaxPool3D()(c4)

    # BOTTLENECK
    b = block(p4, base * 16)
    b = layers.SpatialDropout3D(0.2)(b)

    # DECODER
    u4 = layers.Conv3DTranspose(base * 8, 2, strides=2, padding="same")(b)
    c5 = block(layers.concatenate([u4, c4]), base * 8)
    u3 = layers.Conv3DTranspose(base * 4, 2, strides=2, padding="same")(c5)
    c6 = block(layers.concatenate([u3, c3]), base * 4)
    u2 = layers.Conv3DTranspose(base * 2, 2, strides=2, padding="same")(c6)
    c7 = block(layers.concatenate([u2, c2]), base * 2)
    u1 = layers.Conv3DTranspose(base, 2, strides=2, padding="same")(c7)
    c8 = block(layers.concatenate([u1, c1]), base)

    out = layers.Conv3D(num_classes, 1, activation="softmax", dtype="float32")(c8)
    return Model(inp, out, name="unet3d")


def load_model_any(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found in {os.getcwd()}. "
            "Put it next to main.py or set MODEL_PATH."
        )
    if path.endswith((".weights.h5", ".weights")):
        model = build_unet3d(base=BASE_FILTERS)
        model.load_weights(path)
        return model
    return tf.keras.models.load_model(path, compile=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model_any(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="BraTS segmentation API", version="2.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- auth + inputs
def check_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


def scan_files(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
):
    return {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}


# ---------------------------------------------------------------- core logic
def normalize(vol):
    """Identical to training: z-score over non-zero (brain) voxels only."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol.astype(np.float32)
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vol[mask] - vol[mask].mean()) / (vol[mask].std() + 1e-8)
    return out


def save_upload(upload: UploadFile, folder: str) -> str:
    suffix = ".nii.gz" if (upload.filename or "").endswith(".gz") else ".nii"
    path = os.path.join(folder, f"{os.urandom(4).hex()}{suffix}")
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def run_inference(files: dict, folder: str):
    """Returns (full-size label volume, reference nib image, raw flair volume)."""
    chans, ref, flair_raw = [], None, None
    for m in MODALITIES:
        try:
            img = nib.load(save_upload(files[m], folder))
            vol = img.get_fdata(dtype=np.float32)
        except Exception:
            raise HTTPException(422, f"{m}: not a valid NIfTI file")
        if vol.shape != FULL_SHAPE:
            raise HTTPException(422, f"{m}: expected shape {FULL_SHAPE}, got {vol.shape}")
        if m == "flair":
            flair_raw = vol
        chans.append(normalize(vol)[CROP])
        ref = ref or img

    x = np.stack(chans, axis=-1)[None]                      # (1,128,128,128,4)
    probs = state["model"].predict(x, verbose=0)[0]
    pred = np.argmax(probs, axis=-1).astype(np.uint8)

    full = np.zeros(FULL_SHAPE, dtype=np.uint8)
    full[CROP] = pred
    full[full == 3] = 4                                     # back to BraTS convention
    return full, ref, flair_raw


def compute_stats(full, ref):
    vox = float(np.prod(ref.header.get_zooms()[:3]))
    counts = {
        "necrotic_core": int((full == 1).sum()),
        "edema": int((full == 2).sum()),
        "enhancing_tumor": int((full == 4).sum()),
    }
    wt = sum(counts.values())
    tc = counts["necrotic_core"] + counts["enhancing_tumor"]
    return {
        "tumor_detected": wt > 0,
        "volume_ml": {k: round(v * vox / 1000, 2) for k, v in counts.items()},
        "whole_tumor_ml": round(wt * vox / 1000, 2),
        "tumor_core_ml": round(tc * vox / 1000, 2),
        "voxel_volume_mm3": vox,
        "voxels": counts,
    }


def pick_index(seg, axis):
    """Slice through the tumor centre (or the middle of the brain if no tumor)."""
    idx = np.argwhere(seg > 0)
    if len(idx) == 0:
        return seg.shape[axis] // 2
    return int(idx[:, axis].mean())


def make_overlay_png(flair, seg, view="axial", index=None, scale=2):
    """Returns (png_bytes, slice_index)."""
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[view]
    i = pick_index(seg, axis) if index is None or index < 0 else min(index, seg.shape[axis] - 1)
    f = np.take(flair, i, axis=axis)
    s = np.take(seg, i, axis=axis)

    hi = np.percentile(f[f > 0], 99) if (f > 0).any() else 1.0
    gray = (np.clip(f / (hi + 1e-8), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    for label, color in COLORS.items():
        m = s == label
        rgb[m] = 0.5 * rgb[m] + 0.5 * np.array(color, dtype=np.float32)

    rgb = np.rot90(rgb.astype(np.uint8))                    # upright orientation
    img = Image.fromarray(np.ascontiguousarray(rgb))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), i


def cleanup(folder):
    shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict/app", dependencies=[Depends(check_key)])
def predict_app(files: dict = Depends(scan_files)):
    """One call for the app: tumor stats + 3 overlay images (base64 PNG data URIs)."""
    folder = tempfile.mkdtemp()
    try:
        full, ref, flair = run_inference(files, folder)
        previews = {}
        for view in ("axial", "coronal", "sagittal"):
            png, i = make_overlay_png(flair, full, view)
            previews[view] = {
                "slice_index": i,
                "image": "data:image/png;base64," + base64.b64encode(png).decode(),
            }
        return {**compute_stats(full, ref), "legend": LEGEND, "previews": previews}
    finally:
        cleanup(folder)


@app.post("/predict/preview", dependencies=[Depends(check_key)])
def predict_preview(
    files: dict = Depends(scan_files),
    view: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    slice: int = Query(-1, description="slice index, -1 = through the tumor centre"),
):
    """Returns a single PNG overlay."""
    folder = tempfile.mkdtemp()
    try:
        full, _, flair = run_inference(files, folder)
        png, i = make_overlay_png(flair, full, view, slice)
        return Response(png, media_type="image/png", headers={"X-Slice-Index": str(i)})
    finally:
        cleanup(folder)


@app.post("/predict/stats", dependencies=[Depends(check_key)])
def predict_stats(files: dict = Depends(scan_files)):
    """JSON only: tumor volumes."""
    folder = tempfile.mkdtemp()
    try:
        full, ref, _ = run_inference(files, folder)
        return compute_stats(full, ref)
    finally:
        cleanup(folder)


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(files: dict = Depends(scan_files)):
    """Segmentation mask as .nii.gz (for 3D Slicer / ITK-SNAP)."""
    folder = tempfile.mkdtemp()
    try:
        full, ref, _ = run_inference(files, folder)
        out = os.path.join(folder, "segmentation.nii.gz")
        nib.save(nib.Nifti1Image(full, ref.affine, ref.header), out)
    except Exception:
        cleanup(folder)
        raise
    return FileResponse(out, media_type="application/gzip", filename="segmentation.nii.gz",
                        background=BackgroundTask(cleanup, folder))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
