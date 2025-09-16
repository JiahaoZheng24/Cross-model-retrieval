"""
COCO Dataset Solution for Cross-Modal Retrieval Experiments
Uses COCO validation set with CLIP embeddings for realistic experiments
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COCODatasetManager:
    """Manage COCO dataset for cross-modal retrieval experiments"""

    def __init__(self, data_root: str = "./coco_data", num_samples: int = 1000, seed: int = 42):
        """
        Initialize COCO dataset manager

        Args:
            data_root: Directory to store COCO data
            num_samples: Number of image-caption pairs to use
            seed: global seed for reproducible shuffling
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.num_samples = num_samples
        self.seed = seed

        # COCO API URLs
        self.annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        self.images_base_url = "http://images.cocodataset.org/val2017/"

        # RNG
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass

    def download_coco_annotations(self) -> bool:
        """Download COCO annotations if not exists"""

        annotations_file = self.data_root / "annotations" / "instances_val2017.json"
        captions_file = self.data_root / "annotations" / "captions_val2017.json"

        if annotations_file.exists() and captions_file.exists():
            logger.info("COCO annotations already exist")
            return True

        logger.info("COCO annotations not found. You can download manually:")
        logger.info("  http://images.cocodataset.org/annotations/annotations_trainval2017.zip (~240MB)")
        logger.info("  unzip to: ./coco_data/annotations/")
        return False

    def load_coco_captions(self) -> List[Dict]:
        """Load COCO captions from annotation file"""

        captions_file = self.data_root / "annotations" / "captions_val2017.json"

        if not captions_file.exists():
            logger.error("COCO captions file not found. Using sample data instead.")
            return self.create_sample_coco_data()

        logger.info(f"Loading COCO captions from: {captions_file}")

        with open(captions_file, 'r') as f:
            coco_data = json.load(f)

        images = {img['id']: img['file_name'] for img in coco_data['images']}

        samples = []
        for ann in coco_data['annotations'][:self.num_samples]:
            image_id = ann['image_id']
            if image_id in images:
                samples.append({
                    'image_id': image_id,
                    'filename': images[image_id],
                    'caption': ann['caption'],
                    'image_url': f"{self.images_base_url}{images[image_id]}"
                })

        logger.info(f"Loaded {len(samples)} image-caption pairs")
        return samples

    def create_sample_coco_data(self) -> List[Dict]:
        """Create sample COCO-like data when real data is not available"""

        logger.info(f"Creating {self.num_samples} sample image-caption pairs...")

        sample_captions = [
            "A person riding a bicycle on a city street",
            "A cat sitting on a wooden table",
            "A group of people playing soccer in a park",
            "A red car parked in front of a building",
            "A dog running through a grassy field",
            "A woman holding an umbrella in the rain",
            "A plate of food with vegetables and meat",
            "A bird flying over the ocean",
            "Children playing on a playground",
            "A train arriving at a busy station"
        ]

        samples = []
        for i in range(self.num_samples):
            caption = sample_captions[i % len(sample_captions)]
            if i >= len(sample_captions):
                caption += f" (variation {i // len(sample_captions)})"

            samples.append({
                'image_id': i,
                'filename': f"sample_{i:06d}.jpg",
                'caption': caption,
                'image_url': f"https://via.placeholder.com/640x480/color/text=Image{i}"
            })

        return samples

    def extract_clip_features(self, samples: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract CLIP features from images and captions"""

        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise

        logger.info("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        logger.info(f"Using device: {device}")

        image_embeddings = []
        text_embeddings = []

        logger.info("Extracting CLIP features...")

        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                image = self.load_image(sample['image_url'])
                if image is None:
                    image = Image.new('RGB', (224, 224), color='gray')
                caption = sample['caption']

                inputs = processor(
                    text=[caption],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    img_emb = outputs.image_embeds.cpu().numpy()[0]
                    txt_emb = outputs.text_embeds.cpu().numpy()[0]
                    image_embeddings.append(img_emb)
                    text_embeddings.append(txt_emb)

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                image_embeddings.append(np.random.randn(512).astype(np.float32))
                text_embeddings.append(np.random.randn(512).astype(np.float32))

        image_embeddings = np.array(image_embeddings)
        text_embeddings = np.array(text_embeddings)

        logger.info(f"Extracted embeddings: {image_embeddings.shape}")

        return {
            'image_512': image_embeddings,
            'text_512': text_embeddings
        }

    def load_image(self, image_url: str) -> Optional[Image.Image]:
        """Load image from URL or local path"""
        try:
            if image_url.startswith('http'):
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    return image
            else:
                image_path = self.data_root / "images" / image_url
                if image_path.exists():
                    image = Image.open(image_path).convert('RGB')
                    return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_url}: {e}")
        return None

    def create_multi_scale_embeddings(self, base_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create embeddings of different dimensions"""

        logger.info("Creating multi-scale embeddings...")

        embeddings = base_embeddings.copy()
        text_512 = base_embeddings['text_512']
        image_512 = base_embeddings['image_512']

        # 创建 256/1024 维（随机投影），并统一 L2 normalize
        rng = np.random.default_rng(self.seed)

        logger.info("Creating 256-dim embeddings...")
        reduction_matrix_256 = rng.standard_normal((512, 256)) / np.sqrt(512)
        embeddings['text_256'] = text_512 @ reduction_matrix_256
        embeddings['video_256'] = image_512 @ reduction_matrix_256

        logger.info("Creating 1024-dim embeddings...")
        expansion_matrix_1024 = rng.standard_normal((512, 1024)) / np.sqrt(512)
        embeddings['text_1024'] = text_512 @ expansion_matrix_1024
        embeddings['video_1024'] = image_512 @ expansion_matrix_1024

        embeddings['video_512'] = embeddings.pop('image_512')

        for key in list(embeddings.keys()):
            if key.startswith(('text_', 'video_')):
                norms = np.linalg.norm(embeddings[key], axis=1, keepdims=True) + 1e-12
                embeddings[key] = embeddings[key] / norms

        logger.info("Multi-scale embeddings created")
        return embeddings

    def save_dataset(self, embeddings: Dict[str, np.ndarray], ground_truth: List[int]) -> str:
        """Save the processed dataset"""
        embeddings['ground_truth'] = np.array(ground_truth)
        save_path = self.data_root / "coco_embeddings.npz"
        np.savez_compressed(save_path, **embeddings)

        metadata = {
            "dataset_type": "coco_validation_or_sample",
            "num_samples": len(ground_truth),
            "embedding_dims": [256, 512, 1024],
            "file_size_mb": save_path.stat().st_size / (1024 * 1024),
            "description": "COCO validation set with CLIP embeddings (or sample) for cross-modal retrieval",
            "seed": self.seed
        }

        metadata_path = self.data_root / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset saved to: {save_path} ({metadata['file_size_mb']:.2f} MB)")
        return str(save_path)

    def create_coco_dataset(self) -> str:
        """Main function to create COCO dataset"""

        logger.info("=== Creating COCO Dataset for Cross-Modal Retrieval ===")

        samples = self.load_coco_captions()
        base_embeddings = self.extract_clip_features(samples)
        all_embeddings = self.create_multi_scale_embeddings(base_embeddings)

        # Ground truth: 1:1 alignment (i -> i)
        ground_truth = list(range(len(samples)))

        # ---- 全局 shuffle（仅打乱 queries = text_*，保持 gallery = video_* 不变）----
        # 这样 ground_truth 仍指向“原始 gallery 全局索引”，只需同步打乱 ground_truth 的顺序即可
        perm = np.random.permutation(len(ground_truth))
        for key in list(all_embeddings.keys()):
            if key.startswith('text_'):
                all_embeddings[key] = all_embeddings[key][perm]
        ground_truth = np.array(ground_truth)[perm].tolist()
        logger.info("Applied global shuffle to queries with fixed seed for reproducibility")

        dataset_path = self.save_dataset(all_embeddings, ground_truth)
        logger.info("✓ COCO dataset creation completed")
        return dataset_path


def main():
    """Main function"""
    print("=== COCO Dataset Solution for Cross-Modal Retrieval ===\n")
    num_samples = 1000
    seed = 42
    print(f"Creating dataset with {num_samples} samples (seed={seed})...")

    manager = COCODatasetManager(data_root="./coco_data", num_samples=num_samples, seed=seed)

    try:
        dataset_path = manager.create_coco_dataset()
        print(f"\n✓ COCO dataset ready!")
        print(f"Dataset path: {dataset_path}")
        print(f"Next: python run_experiments.py --data_path {dataset_path} --output_dir ./result")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        print("Error creating dataset. Check logs.")

if __name__ == "__main__":
    main()
