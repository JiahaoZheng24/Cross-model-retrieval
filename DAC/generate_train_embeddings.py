#!/usr/bin/env python3
"""
Generate ImageBind embeddings for MSR-VTT TRAINING SET
This is what you should use to train PCME projectors!

MSR-VTT split:
- Train: video0 - video6512 (6513 videos)
- Val: video6513 - video7009 (497 videos)
- Test: video7010 - video9999 (2990 videos, 1kA uses 1000 of these)
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path("/scratch365/jzheng7/ImageBind")))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from imagebind.models import imagebind_model

try:
    from imagebind.models.modality_type import ModalityType
except:
    from imagebind.models.imagebind_model import ModalityType
from imagebind import data as ib_data

# Import encoding functions from eval script
import eval_msrvtt_1kA as eval_script


def get_train_split():
    """
    Get training set video IDs and captions
    Returns: video_ids, captions
    """
    import json

    ann_dir = Path("/scratch365/jzheng7/ImageBind/msrvtt_annotation")

    print("\n" + "=" * 60)
    print("Determining train split")
    print("=" * 60)

    # Standard MSR-VTT split from original paper
    standard_train_ids = set([f"video{i}" for i in range(6513)])

    # Option 1: Check MSRVTT_train.9k.csv
    train_9k_csv = ann_dir / "MSRVTT_train.9k.csv"
    train_video_ids = None

    if train_9k_csv.exists():
        df = pd.read_csv(train_9k_csv)
        train_9k_ids = set(df['video_id'].tolist())

        # Check if 9k is superset of standard train
        overlap = len(train_9k_ids & standard_train_ids)

        print(f"Found MSRVTT_train.9k.csv with {len(train_9k_ids)} videos")
        print(f"  Overlap with standard train (6513): {overlap} videos")

        if overlap >= 6500:
            # 9k likely includes train + val
            print(f"  → Using standard 6513 train split (for proper benchmarking)")
            train_video_ids = standard_train_ids
        else:
            # 9k might be completely different split
            print(f"  → Using all 9k videos as training set")
            train_video_ids = train_9k_ids

    # Option 2: Use standard split if no CSV or unclear
    if train_video_ids is None:
        print("Using standard MSR-VTT split: video0-video6512 (6513 videos)")
        train_video_ids = standard_train_ids

    print(f"Final decision: {len(train_video_ids)} training videos")
    print("=" * 60 + "\n")

    # Load captions from MSRVTT_data.json
    json_path = ann_dir / "MSRVTT_data.json"
    if not json_path.exists():
        print(f"❌ ERROR: {json_path} not found!")
        return [], []

    print(f"Loading captions from {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Build video -> captions mapping
    video_to_captions = {}
    for sent_info in data['sentences']:
        vid = sent_info['video_id']
        caption = sent_info['caption']
        if vid not in video_to_captions:
            video_to_captions[vid] = []
        video_to_captions[vid].append(caption)

    print(f"Loaded captions for {len(video_to_captions)} videos")

    # Filter to training videos and get first caption
    video_ids = []
    captions = []

    for vid in sorted(train_video_ids):
        if vid in video_to_captions and video_to_captions[vid]:
            video_ids.append(vid)
            captions.append(video_to_captions[vid][0])  # Take first caption

    print(f"Final training set: {len(video_ids)} videos with captions")

    if len(video_ids) == 0:
        print("❌ ERROR: No training videos found with captions!")

    return video_ids, captions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("Loading ImageBind model...")
    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

    # Get training data
    print("\nLoading training split...")
    video_ids, captions = get_train_split()
    print(f"Training set size: {len(video_ids)} videos")

    if len(video_ids) == 0:
        print("❌ ERROR: No training videos found!")
        print("   Check that annotation files exist in:")
        print("   /scratch365/jzheng7/ImageBind/msrvtt_annotation/")
        sys.exit(1)

    if len(captions) == 0:
        print("❌ ERROR: No captions found!")
        sys.exit(1)

    print(f"Sample video IDs: {video_ids[:5]}")
    print(f"Sample caption: {captions[0]}")

    if len(video_ids) != 6513:
        print(f"⚠️  WARNING: Expected 6513 training videos, got {len(video_ids)}")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(0)

    # Prepare paths
    vid_dir = Path("/scratch365/jzheng7/ImageBind/msrvtt_videos")
    video_paths = [vid_dir / f"{vid}.mp4" for vid in video_ids]

    # Check how many videos exist
    existing = sum(1 for p in video_paths if p.exists())
    print(f"Found {existing}/{len(video_paths)} video files")

    # Encode
    print("\nEncoding training set...")
    print("This will take a while (~30-60 minutes)...")

    text_emb = eval_script.encode_text(model, device, captions, use_fp16=True)
    video_emb = eval_script.encode_video(
        model, device, video_paths,
        num_frames=16, image_size=224, use_fp16=True
    )

    # Normalize
    text_emb = F.normalize(text_emb, dim=-1)
    video_emb = F.normalize(video_emb, dim=-1)

    # Save
    output_dir = Path("/scratch365/jzheng7/ImageBind/msrvtt_train_embeddings")
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.save(text_emb, output_dir / "emb_text.pt")
    torch.save(video_emb, output_dir / "emb_video.pt")

    # Save metadata
    import json
    metadata = {
        'n_samples': len(video_ids),
        'split': 'train',
        'video_ids': video_ids[:10] + ['...'] + video_ids[-10:],  # Save first/last 10
        'embedding_dim': text_emb.shape[1]
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved training embeddings to: {output_dir}")
    print(f"  Text embeddings: {text_emb.shape}")
    print(f"  Video embeddings: {video_emb.shape}")
    print("\nNow you can train PCME with:")
    print(f"  python train_pcme_projector.py --emb_dir {output_dir} --save_dir ./pcme_checkpoints_correct")


if __name__ == '__main__':
    main()