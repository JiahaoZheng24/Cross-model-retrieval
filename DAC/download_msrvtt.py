#!/usr/bin/env python3
"""
MSR-VTT Dataset Downloader
Downloads videos and annotations from official sources
"""

import os
import sys
import subprocess
from pathlib import Path
import zipfile
import shutil


def run_command(cmd, description=""):
    """Run shell command with progress"""
    if description:
        print(f"\n{'=' * 60}")
        print(f"{description}")
        print(f"{'=' * 60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error running: {cmd}")
        sys.exit(1)
    print(f"✓ Done\n")


def download_file(url, output_path, description=""):
    """Download file with wget"""
    if Path(output_path).exists():
        print(f"✓ Already exists: {output_path}")
        return

    run_command(
        f"wget -c '{url}' -O '{output_path}'",
        description or f"Downloading {Path(output_path).name}"
    )


def extract_zip(zip_path, extract_to, description=""):
    """Extract zip file"""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return False

    print(f"\n{'=' * 60}")
    print(description or f"Extracting {zip_path.name}")
    print(f"{'=' * 60}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress
            total_size = sum(info.file_size for info in zip_ref.infolist())
            extracted_size = 0

            for info in zip_ref.infolist():
                zip_ref.extract(info, extract_to)
                extracted_size += info.file_size
                progress = (extracted_size / total_size) * 100
                print(f"\rProgress: {progress:.1f}%", end='', flush=True)

            print(f"\n✓ Extracted to: {extract_to}\n")
            return True
    except Exception as e:
        print(f"\n❌ Error extracting: {e}")
        return False


def main():
    # Configuration
    BASE_DIR = Path("/scratch365/jzheng7/ImageBind")
    DOWNLOAD_DIR = BASE_DIR / "msrvtt_downloads"
    VIDEO_DIR = BASE_DIR / "msrvtt_videos"
    ANNOTATION_DIR = BASE_DIR / "msrvtt_annotation"

    print("""
╔════════════════════════════════════════════════════════════╗
║         MSR-VTT Dataset Downloader & Setup Tool           ║
╚════════════════════════════════════════════════════════════╝

This script will download:
1. MSR-VTT Videos (10K videos, ~40GB) from Frozen-in-Time
2. MSR-VTT Annotations from CLIP4Clip
3. Extract and organize into the correct directory structure

Target directories:
  Videos:      {VIDEO_DIR}
  Annotations: {ANNOTATION_DIR}
  Downloads:   {DOWNLOAD_DIR}

""")

    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        sys.exit(0)

    # Create directories
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: Downloading Annotations (~3MB)")
    print("=" * 60)

    # Download annotations
    annotation_zip = DOWNLOAD_DIR / "msrvtt_data.zip"
    download_file(
        "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip",
        annotation_zip,
        "Downloading MSR-VTT annotations from CLIP4Clip"
    )

    # Extract annotations
    if annotation_zip.exists():
        extract_zip(annotation_zip, DOWNLOAD_DIR, "Extracting annotations")

        # Move annotation files to correct location
        annotation_source = DOWNLOAD_DIR / "msrvtt_data"
        if annotation_source.exists():
            print("Moving annotation files...")
            for item in annotation_source.glob("*"):
                dest = ANNOTATION_DIR / item.name
                if dest.exists():
                    print(f"  Skipping (exists): {item.name}")
                else:
                    shutil.move(str(item), str(dest))
                    print(f"  ✓ Moved: {item.name}")

    print("\n" + "=" * 60)
    print("STEP 2: Downloading Videos (~40GB)")
    print("=" * 60)
    print("⚠️  WARNING: This is a large download (~40GB)")
    print("    It may take 1-3 hours depending on your connection")
    print()

    response = input("Download videos now? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        video_zip = DOWNLOAD_DIR / "MSRVTT.zip"
        download_file(
            "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip",
            video_zip,
            "Downloading MSR-VTT videos from Frozen-in-Time"
        )

        # Extract videos
        if video_zip.exists():
            extract_zip(video_zip, DOWNLOAD_DIR, "Extracting videos")

            # Move videos to correct location
            video_source_options = [
                DOWNLOAD_DIR / "MSRVTT" / "videos" / "all",
                DOWNLOAD_DIR / "MSRVTT" / "videos",
                DOWNLOAD_DIR / "videos",
            ]

            video_source = None
            for option in video_source_options:
                if option.exists() and any(option.glob("*.mp4")):
                    video_source = option
                    break

            if video_source:
                print(f"\nMoving videos from {video_source}...")
                mp4_files = list(video_source.glob("*.mp4"))
                print(f"Found {len(mp4_files)} video files")

                for i, video_file in enumerate(mp4_files, 1):
                    dest = VIDEO_DIR / video_file.name
                    if dest.exists():
                        continue
                    shutil.move(str(video_file), str(dest))
                    if i % 100 == 0:
                        print(f"  Moved {i}/{len(mp4_files)} videos...")

                print(f"✓ Moved all {len(mp4_files)} videos")
            else:
                print("⚠️  Could not find video files in expected locations")
                print("   Please check the extracted directory structure")
    else:
        print("\n⚠️  Skipping video download")
        print("   You'll need to download videos manually to continue")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Count videos
    video_files = list(VIDEO_DIR.glob("*.mp4"))
    print(f"\n✓ Videos found: {len(video_files)}")
    if len(video_files) == 10000:
        print("  ✓ Complete! (10,000 videos)")
    elif len(video_files) > 0:
        print(f"  ⚠️  Expected 10,000, found {len(video_files)}")
    else:
        print("  ❌ No videos found")

    # Check annotations
    required_files = [
        "MSRVTT_data.json",
        "MSRVTT_JSFUSION_test.csv",
        "MSRVTT_train.9k.csv"
    ]

    print(f"\n✓ Annotations:")
    for filename in required_files:
        filepath = ANNOTATION_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename:<30} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {filename:<30} (missing)")

    # Generate summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    print(f"""
Directory structure:
  {VIDEO_DIR}/
    ├── video0.mp4
    ├── video1.mp4
    └── ... ({len(video_files)} total)

  {ANNOTATION_DIR}/
    ├── MSRVTT_data.json
    ├── MSRVTT_JSFUSION_test.csv
    ├── MSRVTT_train.9k.csv
    └── ...

Next steps:
  1. Run: python generate_train_embeddings.py
  2. Run: bash run_pcme_benchmark_fixed.sh
  3. Or follow the corrected workflow in PCME_Data_Leakage_Analysis.md
""")

    # Cleanup option
    print("\n" + "=" * 60)
    total_download_size = sum(f.stat().st_size for f in DOWNLOAD_DIR.glob("*.zip")) / (1024 ** 3)

    if total_download_size > 0:
        print(f"Download folder size: {total_download_size:.2f} GB")
        response = input("\nDelete downloaded zip files to save space? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            for zip_file in DOWNLOAD_DIR.glob("*.zip"):
                zip_file.unlink()
                print(f"  ✓ Deleted: {zip_file.name}")
            print(f"✓ Freed {total_download_size:.2f} GB")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)