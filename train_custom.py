import sys
import os
import random
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'dataloader'))

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score

from dataset import train_data_loader, test_data_loader
from model import OACSFM
from loss import OACSFMJointLoss


def set_seed(seed=42):
    """Fix random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train OA-CSFM for dynamic facial expression recognition')

    parser.add_argument('--dataset', type=str, default='DFEW', help='Dataset name, such as DFEW or FERV39K')
    parser.add_argument('--fold', type=int, default=1, help='Cross-validation fold index')
    parser.add_argument('--data_mode', type=str, default='norm', choices=['norm', 'rv', 'flow'],
                        help='Feature mode. Keep this consistent with the existing feature folders.')
    parser.add_argument('--is_face', action='store_true', default=True, help='Use face-cropped feature folders')

    parser.add_argument('--epochs', type=int, default=30, help='Number of total training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='Warmup epochs for regularization terms')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--in_dim', type=int, default=1408, help='Input feature dimension')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension in OA-CSFM')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of expression categories')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention head number in VCCF')
    parser.add_argument('--num_groups', type=int, default=4, help='Feature group number in VAMS')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')

    parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for prediction consistency loss')
    parser.add_argument('--visibility_weight', type=float, default=0.05, help='Weight for visibility regularization')
    parser.add_argument('--reliability_weight', type=float, default=0.05, help='Weight for reliability sparsity regularization')
    parser.add_argument('--rho', type=float, default=0.05, help='Target sparsity ratio for reliability selection')

    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory for model checkpoints')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, max_warmup_epochs):
    model.train()
    total_loss_val = 0.0
    total_cls_val = 0.0
    total_con_val = 0.0
    total_vis_val = 0.0
    total_rel_val = 0.0

    lambda_regular = min(1.0, epoch / max(1, max_warmup_epochs))

    for batch_data in dataloader:
        (x_high, mask_high), (x_middle, mask_middle), (x_low, mask_low), labels = batch_data

        x_high = x_high.to(device, non_blocking=True)
        mask_high = mask_high.to(device, non_blocking=True)
        x_middle = x_middle.to(device, non_blocking=True)
        mask_middle = mask_middle.to(device, non_blocking=True)
        x_low = x_low.to(device, non_blocking=True)
        mask_low = mask_low.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits_tuple, aux_tuple = model(
            (x_high, mask_high),
            (x_middle, mask_middle),
            (x_low, mask_low)
        )
        masks_tuple = (mask_high, mask_middle, mask_low)

        total_loss, loss_cls, loss_con, loss_vis, loss_rel = criterion(
            logits_tuple=logits_tuple,
            targets=labels,
            aux_tuple=aux_tuple,
            masks_tuple=masks_tuple,
            lambda_regular=lambda_regular
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_val += total_loss.item()
        total_cls_val += loss_cls.item()
        total_con_val += loss_con.item()
        total_vis_val += loss_vis.item()
        total_rel_val += loss_rel.item()

    num_batches = max(1, len(dataloader))
    return {
        'loss': total_loss_val / num_batches,
        'cls': total_cls_val / num_batches,
        'con': total_con_val / num_batches,
        'vis': total_vis_val / num_batches,
        'rel': total_rel_val / num_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch_data in dataloader:
        (x_high, mask_high), (x_middle, mask_middle), (x_low, mask_low), labels = batch_data

        logits_tuple, _ = model(
            (x_high.to(device, non_blocking=True), mask_high.to(device, non_blocking=True)),
            (x_middle.to(device, non_blocking=True), mask_middle.to(device, non_blocking=True)),
            (x_low.to(device, non_blocking=True), mask_low.to(device, non_blocking=True))
        )

        logits_high, logits_middle, logits_low = logits_tuple
        logits = (logits_high + logits_middle + logits_low) / 3.0
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    war = accuracy_score(all_labels, all_preds) * 100.0
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0
    return war, uar


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('========== OA-CSFM Training Config ==========')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print(f'Using device: {device}')
    print('=============================================')

    train_loader = torch.utils.data.DataLoader(
        train_data_loader(dataset=args.dataset, data_set=args.fold, data_mode=args.data_mode, is_face=args.is_face),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_data_loader(dataset=args.dataset, data_set=args.fold, data_mode=args.data_mode, is_face=args.is_face),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = OACSFM(
        in_dim=args.in_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_groups=args.num_groups,
        dropout=args.dropout
    ).to(device)

    criterion = OACSFMJointLoss(
        consistency_weight=args.consistency_weight,
        visibility_weight=args.visibility_weight,
        reliability_weight=args.reliability_weight,
        rho=args.rho
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_uar = 0.0
    best_war = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_warmup_epochs=args.warmup_epochs
        )

        war, uar = evaluate(model, test_loader, device)
        scheduler.step()

        print(
            f"Epoch: {epoch:03d}/{args.epochs} | LR: {current_lr:.6f} | "
            f"Loss: {train_metrics['loss']:.4f} | Cls: {train_metrics['cls']:.4f} | "
            f"Con: {train_metrics['con']:.4f} | Vis: {train_metrics['vis']:.4f} | "
            f"Rel: {train_metrics['rel']:.4f} | WAR: {war:.2f}% | UAR: {uar:.2f}%"
        )

        if uar > best_uar:
            best_uar = uar
            best_war = war
            save_path = os.path.join(args.save_dir, f'best_oa_csfm_{args.dataset}_fold{args.fold}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_uar': best_uar,
                'best_war': best_war,
                'args': vars(args),
            }, save_path)
            print(f'--> Saved best OA-CSFM checkpoint to {save_path} (UAR: {best_uar:.2f}%)')

    print(f'Training finished. Fold: {args.fold} | Best UAR: {best_uar:.2f}% | Best WAR: {best_war:.2f}%')


if __name__ == '__main__':
    main()
