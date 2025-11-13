#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MSR-VTT 1k-A evaluation with ImageBind
Outputs JSON:
  /scratch365/jzheng7/ImageBind/msrvtt_results/msrvtt_1kA_metrics.json

Prereqs (in conda env 'imagebind'):
  pip install torch torchvision torchaudio pandas decord tqdm pillow opencv-python einops
  conda install -c conda-forge ffmpeg
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# --- ensure local repo is importable even if not pip-installed ---
sys.path.insert(0, str(Path("/scratch365/jzheng7/ImageBind")))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import pandas as pd

# decord import (clear error if missing)
try:
    from decord import VideoReader, cpu
except Exception:
    print("[ERROR] 'decord' not found. Install with:  pip install decord   (or: conda install -c conda-forge decord)")
    raise

# imagebind imports + compatibility for ModalityType location difference
from imagebind.models import imagebind_model
try:
    from imagebind.models.modality_type import ModalityType
except Exception:
    # some forks keep ModalityType in imagebind_model.py
    from imagebind.models.imagebind_model import ModalityType
from imagebind import data as ib_data


# ------------------------- Paths -------------------------
ROOT    = Path("/scratch365/jzheng7/ImageBind")
VID_DIR = ROOT / "msrvtt_videos"
AUD_DIR = ROOT / "msrvtt_audio"
ANN_DIR = ROOT / "msrvtt_annotation"
RES_DIR = ROOT / "msrvtt_results"
RES_DIR.mkdir(parents=True, exist_ok=True)

LIST_1KA = ANN_DIR / "msrvtt1kA.txt"                 # video_id \t caption_index
JSF_CSV  = ANN_DIR / "MSRVTT_JSFUSION_test.csv"      # has columns: video_id, sentence, ...


# ------------------------- Utils -------------------------
def require_file(p: Path, hint: str):
    if not p.exists():
        print(f"[ERROR] Missing file: {p}\n       {hint}")
        sys.exit(1)

def vision_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

def uniform_sample_frames(vpath: Path, num_frames: int):
    vr = VideoReader(str(vpath), ctx=cpu(0))
    n = len(vr)
    if n == 0:
        return []
    idxs = torch.linspace(0, n - 1, steps=num_frames).long().tolist()
    return [Image.fromarray(vr[i].asnumpy()) for i in idxs]

@torch.no_grad()
def encode_video(model, device, video_paths, num_frames=16, image_size=224, use_fp16=True):
    tfm = vision_transform(image_size)
    embs = []
    for vp in tqdm(video_paths, desc="Encode video", ncols=100):
        try:
            frames = uniform_sample_frames(vp, num_frames)
        except Exception:
            frames = []
        if not frames:
            embs.append(torch.zeros(1024))
            continue
        imgs = torch.stack([tfm(im) for im in frames]).to(device)  # [T,3,H,W]
        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model({ModalityType.VISION: imgs})[ModalityType.VISION]  # [T,D]
        else:
            out = model({ModalityType.VISION: imgs})[ModalityType.VISION]
        embs.append(out.mean(0).detach().cpu())
    return torch.stack(embs, 0)  # [N,D]

@torch.no_grad()
def encode_text(model, device, texts, use_fp16=True):
    td = ib_data.load_and_transform_text(texts, device=device)
    if use_fp16 and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            e = model({ModalityType.TEXT: td})[ModalityType.TEXT]
    else:
        e = model({ModalityType.TEXT: td})[ModalityType.TEXT]
    return e.detach().cpu()

@torch.no_grad()
def encode_audio(model, device, audio_paths, use_fp16=True):
    ad = ib_data.load_and_transform_audio_data(audio_paths, device=device)
    if use_fp16 and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            e = model({ModalityType.AUDIO: ad})[ModalityType.AUDIO]
    else:
        e = model({ModalityType.AUDIO: ad})[ModalityType.AUDIO]
    return e.detach().cpu()

def compute_metrics(sim, ks=(1, 5, 10)):
    N = sim.size(0)
    topk = {k: 0 for k in ks}
    ranks = []
    for i in range(N):
        order = torch.argsort(sim[i], descending=True)
        r = (order == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(r)
        for k in ks:
            if r <= k:
                topk[k] += 1
    out = {f"R@{k}": 100.0 * topk[k] / N for k in ks}
    rnp = np.array(ranks)
    out["MedR"] = float(np.median(rnp))
    out["MeanR"] = float(np.mean(rnp))
    return out

def load_1kA_pairs():
    """
    Return vids, cap_idx, texts (texts pulled from JSFusion CSV per cap_idx)
    """
    require_file(JSF_CSV, "Put MSRVTT_JSFUSION_test.csv in msrvtt_annotation/")
    df = pd.read_csv(JSF_CSV)

    if LIST_1KA.exists():
        lines = [ln.strip() for ln in LIST_1KA.read_text().splitlines() if ln.strip()]
        vids, cap_idx = [], []
        for ln in lines:
            parts = ln.split()
            vids.append(parts[0])
            cap_idx.append(int(parts[1]) if len(parts) > 1 else 0)
        # build captions mapping
        caps_by_vid = {}
        for _, r in df.iterrows():
            caps_by_vid.setdefault(r["video_id"], []).append(r["sentence"])
        texts = []
        for v, ci in zip(vids, cap_idx):
            cands = caps_by_vid.get(v, [])
            if not cands:
                texts.append("")
            else:
                ci = max(0, min(ci, len(cands) - 1))
                texts.append(cands[ci])
        return vids, cap_idx, texts
    else:
        # fallback: first caption per video_id
        seen, vids, cap_idx, texts = set(), [], [], []
        for _, r in df.iterrows():
            v = r["video_id"]
            if v not in seen:
                seen.add(v)
                vids.append(v); cap_idx.append(0); texts.append(r["sentence"])
        return vids, cap_idx, texts

def load_has_audio_idx(vids):
    has_file = ANN_DIR / "has_audio.txt"
    if not has_file.exists():
        return None
    keep = {ln.strip() for ln in has_file.read_text().splitlines() if ln.strip()}
    idx = [i for i, v in enumerate(vids) if v in keep]
    if len(idx) == 0:
        return None
    return torch.tensor(idx, dtype=torch.long)


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", choices=["base", "large", "huge"], default="huge")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--paper_faithful", action="store_true",
                    help="Only standard outputs; no has-audio subset metrics")
    args = ap.parse_args()

    # Sanity
    require_file(JSF_CSV, "Missing JSFusion CSV.")
    if not VID_DIR.exists() or not any(VID_DIR.glob("*.mp4")):
        print(f"[ERROR] No videos under {VID_DIR}")
        sys.exit(1)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model builder auto-detect & fallback ----
    builder = None
    if args.model_size == "huge" and hasattr(imagebind_model, "imagebind_huge"):
        builder = imagebind_model.imagebind_huge
    elif args.model_size == "large" and hasattr(imagebind_model, "imagebind_large"):
        builder = imagebind_model.imagebind_large
    elif args.model_size == "base" and hasattr(imagebind_model, "imagebind_base"):
        builder = imagebind_model.imagebind_base
    else:
        if hasattr(imagebind_model, "imagebind_large"):
            print(f"[WARN] requested {args.model_size}, falling back to large")
            builder = imagebind_model.imagebind_large
            args.model_size = "large"
        elif hasattr(imagebind_model, "imagebind_base"):
            print(f"[WARN] requested {args.model_size}, falling back to base")
            builder = imagebind_model.imagebind_base
            args.model_size = "base"
        elif hasattr(imagebind_model, "imagebind_huge"):
            print(f"[WARN] requested {args.model_size}, only huge available; using it")
            builder = imagebind_model.imagebind_huge
            args.model_size = "huge"
        else:
            raise RuntimeError("No imagebind_{base,large,huge} in your imagebind_model.")

    model = builder(pretrained=True).to(device).eval()
    use_fp16 = (not args.no_fp16)

    # ---- Load 1k-A pairs ----
    vids, cap_idx, texts = load_1kA_pairs()
    vpaths = [VID_DIR / f"{v}.mp4" for v in vids]
    apaths = [AUD_DIR / f"{v}.wav" for v in vids]

    # ---- Embedding cache paths ----
    e_txt = RES_DIR / "emb_text.pt"
    e_vid = RES_DIR / "emb_video.pt"
    e_aud = RES_DIR / "emb_audio.pt"

    # ---- Encode text/video/audio with caching ----
    if e_txt.exists():
        text = torch.load(e_txt)
    else:
        text = encode_text(model, device, texts, use_fp16=use_fp16)
        text = F.normalize(text, dim=-1)  # FIX: Normalize before saving!
        torch.save(text, e_txt)

    if e_vid.exists():
        video = torch.load(e_vid)
    else:
        video = encode_video(model, device, vpaths, num_frames=args.frames,
                             image_size=args.image_size, use_fp16=use_fp16)
        video = F.normalize(video, dim=-1)  # FIX: Normalize before saving!
        torch.save(video, e_vid)

    if e_aud.exists():
        audio = torch.load(e_aud)
    else:
        # handle missing wavs by substituting an existing one, then zero-out missing
        exist_flags = [p.exists() for p in apaths]
        placeholder = next((p for p in apaths if p.exists()), None)
        if placeholder is None:
            D = 1024
            audio = torch.zeros(len(apaths), D)
        else:
            paths_feed = [str(p if p.exists() else placeholder) for p in apaths]
            audio = encode_audio(model, device, paths_feed, use_fp16=use_fp16)
            for i, ok in enumerate(exist_flags):
                if not ok:
                    audio[i].zero_()
        audio = F.normalize(audio, dim=-1)  # FIX: Normalize before saving!
        torch.save(audio, e_aud)

    # ---- Normalize ---- (already normalized when saved, but ensure consistency)
    text  = F.normalize(text,  dim=-1)
    video = F.normalize(video, dim=-1)
    audio = F.normalize(audio, dim=-1)

    results = {}

    # 1) Text -> Video
    sim_tv = text @ video.t()
    results["Text2Video"] = compute_metrics(sim_tv)

    # 2) Video -> Text
    sim_vt = video @ text.t()
    results["Video2Text"] = compute_metrics(sim_vt)

    # 3) Text -> Audio (ALL)
    sim_ta_all = text @ audio.t()
    results["Text2Audio_all"] = compute_metrics(sim_ta_all)

    # 3b) Text -> Audio (subset with audio)
    if not args.paper_faithful:
        idx_has = load_has_audio_idx(vids)
        if idx_has is not None and idx_has.numel() > 0:
            sim_ta_sub = text[idx_has] @ audio[idx_has].t()
            results["Text2Audio_subset_hasAudio"] = compute_metrics(sim_ta_sub)
            results["AudioCoverage"] = {
                "has_audio": int(idx_has.numel()),
                "total": int(len(vids)),
                "ratio": float(idx_has.numel()/len(vids))
            }

    # 4) Text -> (Video + Audio) fusion (sum + normalize)
    fused = F.normalize(video + audio, dim=-1)
    sim_tva = text @ fused.t()
    results["Text2VideoAudioFusion"] = compute_metrics(sim_tva)

    # ---- Save ----
    out = {
        "config": {
            "model": f"imagebind_{args.model_size}",
            "frames": args.frames,
            "image_size": args.image_size,
            "fp16": use_fp16,
            "paper_faithful": bool(args.paper_faithful),
        },
        "counts": {"N": len(vids)},
        "results": results
    }
    with open(RES_DIR / "msrvtt_1kA_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()