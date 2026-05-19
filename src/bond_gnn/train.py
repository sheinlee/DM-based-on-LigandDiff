"""
Train Bond GNN on CIF-derived bond labels.
Usage:
  python src/bond_gnn/train.py \
      --xyz_dir /home/scli/diffusion_model_new/xyz_ln_all \
      --cif_dir /home/scli/diffusion_model/cif_output \
      --out     src/bond_gnn/bond_gnn.pt \
      --epochs  50
"""

import argparse, os, sys, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.bond_gnn.model   import BondGNN
from src.bond_gnn.dataset import BondDataset, collate_fn


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Building dataset...')
    ds = BondDataset(args.xyz_dir, args.cif_dir, max_samples=args.max_samples)
    if len(ds) == 0:
        print('ERROR: empty dataset'); return

    n_val  = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    model = BondGNN(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} parameters')

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)

    best_val = float('inf')
    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────
        model.train()
        train_loss = train_acc = train_n = 0
        for pos, oh, labels, edges in train_loader:
            pos, oh, labels, edges = (pos.to(device), oh.to(device),
                                       labels.to(device), edges.to(device))
            src, dst = edges[:, 0], edges[:, 1]
            edge_index = torch.stack([src, dst])
            diff = pos[src] - pos[dst]
            dist = diff.norm(dim=-1, keepdim=True)

            # Re-run forward with explicit edges (avoid re-computing in full model)
            h = model.node_init(oh)
            for layer in model.mp_layers:
                h = layer(h, edge_index, dist)
            pair_feat = torch.cat([h[src], h[dst], dist], dim=-1)
            logits = model.bond_head(pair_feat).squeeze(-1)

            # Weighted BCE: pos bonds are rare (~10%), up-weight them
            pos_weight = torch.tensor([(labels == 0).sum() /
                                       (labels == 1).sum().clamp(min=1)],
                                       device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            preds = (logits.sigmoid() > 0.5).float()
            train_acc += (preds == labels).float().mean().item()
            train_loss += loss.item()
            train_n += 1
        scheduler.step()

        # ── Validate ───────────────────────────────────────
        model.eval()
        val_loss = val_prec = val_rec = val_n = 0
        with torch.no_grad():
            for pos, oh, labels, edges in val_loader:
                pos, oh, labels, edges = (pos.to(device), oh.to(device),
                                           labels.to(device), edges.to(device))
                src, dst = edges[:, 0], edges[:, 1]
                edge_index = torch.stack([src, dst])
                diff = pos[src] - pos[dst]
                dist = diff.norm(dim=-1, keepdim=True)

                h = model.node_init(oh)
                for layer in model.mp_layers:
                    h = layer(h, edge_index, dist)
                pair_feat = torch.cat([h[src], h[dst], dist], dim=-1)
                logits = model.bond_head(pair_feat).squeeze(-1)

                pos_weight = torch.tensor([(labels == 0).sum() /
                                           (labels == 1).sum().clamp(min=1)],
                                           device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight)
                val_loss += loss.item()

                preds = (logits.sigmoid() > 0.5).float()
                tp = ((preds == 1) & (labels == 1)).sum().float()
                fp = ((preds == 1) & (labels == 0)).sum().float()
                fn = ((preds == 0) & (labels == 1)).sum().float()
                val_prec += (tp / (tp + fp + 1e-8)).item()
                val_rec  += (tp / (tp + fn + 1e-8)).item()
                val_n += 1

        vl = val_loss / max(val_n, 1)
        tl = train_loss / max(train_n, 1)
        ta = train_acc / max(train_n, 1)
        vp = val_prec / max(val_n, 1)
        vr = val_rec / max(val_n, 1)
        vf1 = 2*vp*vr / (vp+vr+1e-8)
        print(f'Epoch {epoch+1:3d}/{args.epochs}  '
              f'train_loss={tl:.4f} acc={ta:.3f}  '
              f'val_loss={vl:.4f} prec={vp:.3f} rec={vr:.3f} F1={vf1:.3f}')

        if vl < best_val:
            best_val = vl
            torch.save({'model': model.state_dict(),
                        'hidden_dim': args.hidden_dim,
                        'n_layers': args.n_layers}, args.out)
            print(f'  → saved (best val_loss={best_val:.4f})')

    print(f'\nDone. Best model saved to {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--xyz_dir',    default='/home/scli/diffusion_model_new/xyz_ln_all')
    p.add_argument('--cif_dir',    default='/home/scli/diffusion_model/cif_output')
    p.add_argument('--out',        default='src/bond_gnn/bond_gnn.pt')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_layers',   type=int, default=3)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--max_samples',type=int, default=None)
    train(p.parse_args())
