#!/usr/bin/env python3
"""Qwen3-TTS 本地工作台。

三种模式（各自一个模型，按需加载，内存里只留 QWEN_TTS_KEEP 个）：
  custom  预置音色  Qwen3-TTS-12Hz-1.7B-CustomVoice  9 个成品音色 + instruct 调风格
  design  设计音色  Qwen3-TTS-12Hz-1.7B-VoiceDesign  文字描述造一个全新音色
  clone   克隆音色  Qwen3-TTS-12Hz-1.7B-Base         3 秒参考音复刻音色

启动: ./run.sh          → http://127.0.0.1:8808
"""
from __future__ import annotations

import base64
import gc
import io
import json
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"
PROFILES = ROOT / "profiles"
WEB_DIR = ROOT / "web"
try:
    PROFILES.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass

DEVICE = os.environ.get("QWEN_TTS_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE_NAME = os.environ.get("QWEN_TTS_DTYPE") or ("bfloat16" if DEVICE == "mps" else "float32")
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE_NAME]
KEEP = max(1, int(os.environ.get("QWEN_TTS_KEEP", "1")))

MODES = OrderedDict({
    "custom": {"label": "预置音色", "repo": "Qwen3-TTS-12Hz-1.7B-CustomVoice", "kind": "custom_voice"},
    "design": {"label": "设计音色", "repo": "Qwen3-TTS-12Hz-1.7B-VoiceDesign", "kind": "voice_design"},
    "clone": {"label": "克隆音色", "repo": "Qwen3-TTS-12Hz-1.7B-Base", "kind": "base"},
})

SPEAKERS = [
    {"id": "Vivian", "name": "Vivian · 薇", "desc": "明亮略带锋利的年轻女声", "lang": "中文", "tag": "女 · 青年"},
    {"id": "Serena", "name": "Serena · 瑟", "desc": "温暖柔和的年轻女声", "lang": "中文", "tag": "女 · 青年"},
    {"id": "Uncle_Fu", "name": "Uncle Fu · 福", "desc": "低沉醇厚的成熟男声", "lang": "中文", "tag": "男 · 中年"},
    {"id": "Dylan", "name": "Dylan · 迪", "desc": "京腔年轻男声，清亮自然", "lang": "中文（北京话）", "tag": "男 · 青年"},
    {"id": "Eric", "name": "Eric · 艾", "desc": "成都话男声，略带沙哑的亮度", "lang": "中文（四川话）", "tag": "男 · 青年"},
    {"id": "Ryan", "name": "Ryan · 瑞", "desc": "节奏驱动力强的男声", "lang": "English", "tag": "男 · 英文"},
    {"id": "Aiden", "name": "Aiden · 艾登", "desc": "阳光美式男声，中频清晰", "lang": "English", "tag": "男 · 英文"},
    {"id": "Ono_Anna", "name": "Ono Anna · 小野", "desc": "轻快跳跃的日语女声", "lang": "日本語", "tag": "女 · 日语"},
    {"id": "Sohee", "name": "Sohee · 秀熙", "desc": "情感丰富的韩语女声", "lang": "한국어", "tag": "女 · 韩语"},
]

LANGUAGES = [
    ("Auto", "自动识别"), ("Chinese", "中文"), ("English", "English"),
    ("Japanese", "日本語"), ("Korean", "한국어"), ("German", "Deutsch"),
    ("French", "Français"), ("Russian", "Русский"), ("Portuguese", "Português"),
    ("Spanish", "Español"), ("Italian", "Italiano"),
]

# ── 模型管理 ────────────────────────────────────────────────
_loaded: "OrderedDict[str, dict]" = OrderedDict()
_gen_lock = threading.Lock()      # 生成串行
_load_lock = threading.Lock()     # 加载串行
_state: dict = {"preload": None}


def model_dir(mode: str) -> Path:
    return MODELS_ROOT / MODES[mode]["repo"]


def available(mode: str) -> bool:
    d = model_dir(mode)
    return d.exists() and (d / "model.safetensors").exists()


def _evict_until_room():
    while len(_loaded) >= KEEP:
        old_key, old = _loaded.popitem(last=False)
        del old["model"]
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        print(f"[model] 卸载 {old_key}", flush=True)


def get_model(mode: str):
    if mode not in MODES:
        raise KeyError(f"未知模式: {mode}")
    if not available(mode):
        raise FileNotFoundError(f"模型未下载: {model_dir(mode)}")
    with _load_lock:
        if mode in _loaded:
            _loaded.move_to_end(mode)
            return _loaded[mode]["model"]
        _evict_until_room()
        t0 = time.time()
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(model_dir(mode)), device_map=DEVICE, dtype=DTYPE)
        _loaded[mode] = {"model": model, "load_s": round(time.time() - t0, 2)}
        print(f"[model] 加载 {mode} {_loaded[mode]['load_s']}s", flush=True)
        return model


# ── 音频工具 ────────────────────────────────────────────────
def wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    w = np.asarray(wav, dtype=np.float32)
    w = w - w.mean()
    peak = float(np.abs(w).max()) or 1.0
    w = (w / peak * 0.95).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, w, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def b64(wav: np.ndarray, sr: int) -> str:
    return base64.b64encode(wav_bytes(wav, sr)).decode()


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fa5-]+", "-", name.strip()).strip("-")
    return (s or "voice")[:40]


# ── 请求模型 ────────────────────────────────────────────────
class CustomReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    speaker: str = "Vivian"
    language: str = "Chinese"
    instruct: str = ""
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50


class DesignReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    instruct: str = Field(..., min_length=2, max_length=800)
    language: str = "Chinese"
    n: int = Field(4, ge=1, le=6)
    temperature: float = 0.9
    top_p: float = 1.0


class CloneReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = "Chinese"
    profile: str | None = None
    ref_text: str = ""
    x_vector_only: bool = False


class SaveProfileReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    audio_b64: str
    ref_text: str = ""
    instruct: str = ""
    note: str = ""


# ── 应用 ────────────────────────────────────────────────────
def _preload():
    try:
        get_model("custom")
        _state["preload"] = "ok"
    except Exception as e:  # noqa: BLE001
        _state["preload"] = f"{type(e).__name__}: {e}"
        print(f"[preload] {e}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="Qwen3-TTS 本地工作台", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/info")
def info():
    return {
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "torch": torch.__version__,
        "keep": KEEP,
        "loaded": {k: v["load_s"] for k, v in _loaded.items()},
        "modes": [
            {
                "id": k,
                "label": v["label"],
                "repo": v["repo"],
                "kind": v["kind"],
                "available": available(k),
                "loaded": k in _loaded,
            }
            for k, v in MODES.items()
        ],
        "speakers": SPEAKERS,
        "languages": [{"id": a, "label": b} for a, b in LANGUAGES],
        "sample_rate": 24000,
    }


@app.get("/api/profiles")
def list_profiles():
    out = []
    for d in PROFILES.iterdir():
        meta_f, wav_f = d / "meta.json", d / "ref.wav"
        if not (d.is_dir() and meta_f.exists() and wav_f.exists()):
            continue
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        x, sr = sf.read(wav_f)
        out.append({**meta, "id": d.name, "dur": round(len(x) / sr, 2),
                    "url": f"/api/profiles/{d.name}/ref.wav"})
    out.sort(key=lambda m: m.get("created", ""), reverse=True)
    return out


@app.get("/api/profiles/{pid}/ref.wav")
def profile_wav(pid: str):
    f = PROFILES / pid / "ref.wav"
    if not f.exists():
        raise HTTPException(404, "not found")
    return FileResponse(f, media_type="audio/wav")


@app.post("/api/profiles")
def save_profile(req: SaveProfileReq):
    raw = req.audio_b64
    if "," in raw[:120]:
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    pid = slugify(req.name)
    d = PROFILES / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "ref.wav").write_bytes(data)
    meta = {"name": req.name, "id": pid, "ref_text": req.ref_text, "instruct": req.instruct,
            "note": req.note, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "id": pid, "meta": meta}


@app.get("/api/health")
def health():
    return {"ok": True, "loaded": list(_loaded), "preload": _state["preload"]}


def _audio_response(wav: np.ndarray, sr: int, gen: float):
    return Response(
        content=wav_bytes(wav, sr), media_type="audio/wav",
        headers={
            "X-Gen-Seconds": f"{gen:.2f}",
            "X-Audio-Seconds": f"{len(wav)/sr:.2f}",
            "X-Rtf": f"{gen/max(len(wav)/sr, 1e-6):.2f}",
            "X-Sample-Rate": str(sr),
            "Access-Control-Expose-Headers": "X-Gen-Seconds,X-Audio-Seconds,X-Rtf,X-Sample-Rate",
        },
    )


@app.post("/api/tts")
def tts_custom(req: CustomReq):
    """预置音色。"""
    model = get_model("custom")
    t0 = time.time()
    with _gen_lock, torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=req.text.strip(), speaker=req.speaker, language=req.language,
            instruct=req.instruct.strip() or None, do_sample=True,
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
        )
    return _audio_response(wavs[0], sr, time.time() - t0)


@app.post("/api/design")
def design(req: DesignReq):
    """文字描述造音色，一次出 n 个候选（同一 instruct，不同采样）。"""
    model = get_model("design")
    t0 = time.time()
    n = req.n
    with _gen_lock, torch.inference_mode():
        wavs, sr = model.generate_voice_design(
            text=[req.text.strip()] * n,
            language=[req.language] * n,
            instruct=[req.instruct.strip()] * n,
            do_sample=True, temperature=req.temperature, top_p=req.top_p,
        )
    gen = time.time() - t0
    return {
        "candidates": [
            {"id": i, "audio": b64(w, sr), "dur": round(len(w) / sr, 2)}
            for i, w in enumerate(wavs)
        ],
        "gen": round(gen, 2), "sr": sr,
        "instruct": req.instruct.strip(), "text": req.text.strip(), "language": req.language,
    }


@app.post("/api/clone")
def clone(req: CloneReq):
    """用参考音克隆音色念新台词。"""
    if not req.profile:
        raise HTTPException(400, "请先在左侧选择一个音色卡")
    model = get_model("clone")
    d = PROFILES / req.profile
    if not (d / "ref.wav").exists():
        raise HTTPException(404, f"音色卡不存在: {req.profile}")

    ref_audio = str(d / "ref.wav")
    ref_text = req.ref_text.strip()
    x_only = req.x_vector_only
    if (d / "meta.json").exists():
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if not ref_text:
            ref_text = (meta.get("ref_text") or "").strip()
        if meta.get("x_vector_only"):
            x_only = True

    t0 = time.time()
    with _gen_lock, torch.inference_mode():
        wavs, sr = model.generate_voice_clone(
            text=req.text.strip(), language=req.language,
            ref_audio=ref_audio, ref_text=ref_text or None,
            x_vector_only_mode=x_only or not ref_text,
        )
    return _audio_response(wavs[0], sr, time.time() - t0)


@app.post("/api/upload_ref")
async def upload_ref(file: UploadFile = File(...), name: str = Form("上传音色"), ref_text: str = Form("")):
    """上传一段自己的参考音（用于克隆）。"""
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(413, "音频过大（上限 30 MB）")
    pid = slugify(name) + "-" + str(int(time.time()))[-5:]
    d = PROFILES / pid
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "_raw"
    tmp.write_bytes(data)
    try:
        import librosa
        y, _ = librosa.load(str(tmp), sr=24000, mono=True)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"音频解码失败: {e}")
    tmp.unlink(missing_ok=True)
    sf.write(d / "ref.wav", np.asarray(y, dtype=np.float32), 24000)
    meta = {"name": name, "id": pid, "ref_text": ref_text.strip(), "instruct": "",
            "note": "上传", "x_vector_only": not ref_text.strip(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "id": pid, "meta": meta, "dur": round(len(y) / 24000, 2)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("QWEN_TTS_PORT", "8808"))
    print(json.dumps({"device": DEVICE, "dtype": DTYPE_NAME, "keep": KEEP,
                      "models": {k: str(model_dir(k)) for k in MODES}}, ensure_ascii=False, indent=2))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
