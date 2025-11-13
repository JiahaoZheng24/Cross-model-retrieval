#!/usr/bin/env python3
"""
Fixed PCME Projector Training
- Uses proper probabilistic cross-modal matching loss
- Samples from distributions during training
- Better variance regularization
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


class ProbabilisticProjector(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.mu_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )
        self.logvar_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        f_mu = self.mu_proj(x)
        mu = x + f_mu  # Residual
        mu = F.normalize(mu, dim=-1)  # ← 加这一行！强制mu的norm=1
        logvar = torch.clamp(self.logvar_proj(x), -5, 2)
        return mu, logvar


class EmbeddingDataset(Dataset):
    def __init__(self, text, video):
        self.text = text
        self.video = video

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i], self.video[i]


def pcme_loss_monte_carlo(t_mu, t_logvar, v_mu, v_logvar,
                          temperature=0.07, n_samples=5):
    """
    Proper PCME loss with Monte Carlo sampling

    Key difference from your original:
    - Samples from N(μ, σ²) during training
    - Loss is computed on samples, not just μ
    - Variance is trained to be useful, not penalized
    """
    t_std = torch.exp(0.5 * t_logvar)
    v_std = torch.exp(0.5 * v_logvar)

    total_loss = 0.0

    for _ in range(n_samples):
        # Sample from distributions
        eps_t = torch.randn_like(t_mu)
        eps_v = torch.randn_like(v_mu)
        z_t = t_mu + eps_t * t_std
        z_v = v_mu + eps_v * v_std

        # Normalize
        z_t_norm = F.normalize(z_t, dim=-1)
        z_v_norm = F.normalize(z_v, dim=-1)

        # Similarity matrix
        sim = z_t_norm @ z_v_norm.t() / temperature

        # Bidirectional InfoNCE
        labels = torch.arange(len(z_t), device=z_t.device)
        loss_t2v = F.cross_entropy(sim, labels)
        loss_v2t = F.cross_entropy(sim.t(), labels)

        total_loss += (loss_t2v + loss_v2t)

    return total_loss / n_samples


def kl_divergence_loss(mu, logvar):
    """
    KL divergence to N(0, I) prior
    Encourages variance around 1, prevents collapse
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def variance_lower_bound_loss(logvar, min_var=0.1):
    """
    Soft constraint: variance should be at least min_var
    Only penalizes when variance drops below threshold
    """
    var = torch.exp(logvar)
    return F.relu(min_var - var).mean()


def variance_upper_bound_loss(logvar, max_var=10.0):
    """
    Soft constraint: variance should be at most max_var
    Only penalizes when variance exceeds threshold
    """
    var = torch.exp(logvar)
    return F.relu(var - max_var).mean()


def variance_target_loss(logvar, target_sigma=0.3):
    """
    Pull variance toward target value: sigma = target_sigma
    """
    sigma = torch.exp(0.5 * logvar)
    return ((sigma - target_sigma) ** 2).mean()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load embeddings
    print(f"Loading embeddings from {args.emb_dir}")
    text = torch.load(Path(args.emb_dir) / 'emb_text.pt')
    video = torch.load(Path(args.emb_dir) / 'emb_video.pt')
    N = text.shape[0]
    assert N >= 6000, f"Looks like TEST split (N={N}). Point --emb_dir to msrvtt_train_embeddings."
    try:
        audio = torch.load(Path(args.emb_dir) / 'emb_audio.pt')
        print(f"  Text: {text.shape}, Video: {video.shape}, Audio: {audio.shape}")
    except:
        audio = None
        print(f"  Text: {text.shape}, Video: {video.shape}")

    # Models
    dim = text.size(-1)
    text_proj = ProbabilisticProjector(dim).to(device)
    video_proj = ProbabilisticProjector(dim).to(device)
    audio_proj = ProbabilisticProjector(dim).to(device) if audio is not None else None

    # Optimizer
    params = list(text_proj.parameters()) + list(video_proj.parameters())
    if audio_proj is not None:
        params += list(audio_proj.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Dataloader
    dataset = EmbeddingDataset(text, video)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

    # Training
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    best_loss = float('inf')

    print(f"\nTraining configuration:")
    print(f"  Loss: {args.loss_type}")
    print(f"  MC samples: {args.n_samples}")
    print(f"  Variance reg: {args.var_reg_type} (weight={args.var_reg_weight})")
    print(f"  Temperature: {args.temperature}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}\n")

    for epoch in range(args.epochs):
        text_proj.train()
        video_proj.train()
        if audio_proj is not None:
            audio_proj.train()

        total_loss = 0
        total_cont_loss = 0
        total_reg_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for text_batch, video_batch in pbar:
            text_batch = text_batch.to(device)
            video_batch = video_batch.to(device)

            optimizer.zero_grad()

            # Forward
            text_batch = F.normalize(text_batch, dim=-1)
            video_batch = F.normalize(video_batch, dim=-1)
            t_mu, t_logvar = text_proj(text_batch)
            v_mu, v_logvar = video_proj(video_batch)

            # Main contrastive loss
            if args.loss_type == 'pcme_mc':
                # ✅ NEW: Proper PCME with Monte Carlo
                cont_loss = pcme_loss_monte_carlo(
                    t_mu, t_logvar, v_mu, v_logvar,
                    temperature=args.temperature,
                    n_samples=args.n_samples
                )
            elif args.loss_type == 'deterministic':
                # Your original: just train on μ (for comparison)
                t_mu_norm = F.normalize(t_mu, dim=-1)
                v_mu_norm = F.normalize(v_mu, dim=-1)
                sim = t_mu_norm @ v_mu_norm.t() / args.temperature
                labels = torch.arange(len(text_batch), device=device)
                cont_loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)

            # Variance regularization
            if args.var_reg_type == 'kl':
                # KL divergence to N(0, I)
                reg_loss = kl_divergence_loss(t_mu, t_logvar) + \
                           kl_divergence_loss(v_mu, v_logvar)
            elif args.var_reg_type == 'lower_bound':
                # Prevent collapse below threshold
                reg_loss = variance_lower_bound_loss(t_logvar, args.min_var) + \
                           variance_lower_bound_loss(v_logvar, args.min_var)
            elif args.var_reg_type == 'upper_bound':
                # Prevent variance from getting too large
                reg_loss = variance_upper_bound_loss(t_logvar, args.max_var) + \
                           variance_upper_bound_loss(v_logvar, args.max_var)
            elif args.var_reg_type == 'target':
                # Pull variance toward target sigma
                reg_loss = variance_target_loss(t_logvar, args.target_sigma) + \
                           variance_target_loss(v_logvar, args.target_sigma)
            elif args.var_reg_type == 'penalty':
                # Your original (for comparison) - not recommended
                reg_loss = t_logvar.exp().mean() + v_logvar.exp().mean()
            else:
                # No regularization
                reg_loss = 0.0

            # Total loss
            if isinstance(reg_loss, float):
                loss = cont_loss
            else:
                loss = cont_loss + args.var_reg_weight * reg_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cont_loss += cont_loss.item()
            if not isinstance(reg_loss, float):
                total_reg_loss += reg_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cont': f'{cont_loss.item():.4f}',
                'reg': f'{reg_loss.item() if not isinstance(reg_loss, float) else 0:.4f}'
            })

        scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_cont = total_cont_loss / len(loader)
        avg_reg = total_reg_loss / len(loader)

        # Check variance health
        with torch.no_grad():
            t_mu_all, t_lv_all = text_proj(text.to(device))
            v_mu_all, v_lv_all = video_proj(video.to(device))
            t_var = torch.exp(t_lv_all).mean().item()
            v_var = torch.exp(v_lv_all).mean().item()

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Cont={avg_cont:.4f}, "
              f"Reg={avg_reg:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        print(f"  Variance: text={t_var:.6f}, video={v_var:.6f}")

        if t_var < 0.01 or v_var < 0.01:
            print(f"  ⚠️  WARNING: Variance collapsing!")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            cfg = dict(vars(args))
            cfg.update({
                "train_size": N,
                "emb_dir": str(Path(args.emb_dir)),
                "in_dim": text.size(-1),
                "hidden": 2048,
                "out_dim": text.size(-1),
            })
            save_dict = {
                "epoch": epoch,
                "text": text_proj.state_dict(),
                "video": video_proj.state_dict(),
                "loss": best_loss,
                "config": cfg
            }
            if audio_proj is not None:
                save_dict["audio"] = audio_proj.state_dict()
            torch.save(save_dict, Path(args.save_dir) / "best_projectors.pth")
            print(f"  ✓ Saved best (loss: {best_loss:.4f})")

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"\nFinal variance check:")
    print(f"  Text variance:  {t_var:.6f}")
    print(f"  Video variance: {v_var:.6f}")

    if t_var > 0.05 and v_var > 0.05:
        print(f"  ✅ Healthy variance - PCME should work well!")
    elif t_var > 0.01 and v_var > 0.01:
        print(f"  ⚠️  Marginal variance - may need tuning")
    else:
        print(f"  ❌ Variance collapsed - retrain with different settings")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)

    # Loss configuration
    parser.add_argument('--loss_type', type=str,
                        choices=['pcme_mc', 'deterministic'],
                        default='pcme_mc',
                        help='pcme_mc: proper PCME, deterministic: your original')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Monte Carlo samples for PCME loss')

    # Variance regularization
    parser.add_argument('--var_reg_type', type=str,
                        choices=['none', 'kl', 'lower_bound', 'penalty', 'upper_bound', 'target'],
                        default='kl',
                        help='kl: KL to N(0,I), lower_bound: prevent collapse, '
                             'upper_bound: prevent too large variance, '
                             'target: pull to target sigma, '
                             'penalty: your original (not recommended), none: no reg')
    parser.add_argument('--var_reg_weight', type=float, default=0.001,
                        help='Weight for variance regularization')
    parser.add_argument('--min_var', type=float, default=0.1,
                        help='Minimum variance threshold (for lower_bound)')
    parser.add_argument('--max_var', type=float, default=0.09,
                        help='Maximum variance threshold (for upper_bound)')
    parser.add_argument('--target_sigma', type=float, default=0.3,
                        help='Target sigma value (for target regularization)')

    args = parser.parse_args()

    train(args)
