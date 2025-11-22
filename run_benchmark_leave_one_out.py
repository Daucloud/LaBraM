# --------------------------------------------------------
# Leave-one-subject-out 40-class SSVEP classification on Benchmark
# ---------------------------------------------------------

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from timm.models import create_model
import modeling_pretrain  # register custom models

from engine_for_ssvep_cls import train_one_epoch, evaluate
from ssvep_dataset import build_Benchmark_dataset_leave_one_out
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def get_args():
    parser = argparse.ArgumentParser('Benchmark leave-one-out classification', add_help=False)
    parser.add_argument('--data_root', type=str, default='./data', help='root dir that contains Benchmark folder')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--win_len', type=float, default=1.0, help='signal window length in seconds')
    parser.add_argument('--filterbank', type=int, default=1, help='number of filterbank subbands')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test_sub', default=None, type=int, help='run a single held-out subject (1-35); if None, run all')
    parser.add_argument('--model', default='labram_base_patch200_1600_cls40', type=str)
    parser.add_argument('--init_ckpt', default='', type=str, help='path to pretrain checkpoint')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--warmup_steps', default=-1, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    subjects = [args.test_sub] if args.test_sub is not None else list(range(1, 36))
    all_acc = []

    for sub in subjects:
        print(f'=== Held-out subject: S{sub} ===')
        train_dataset, test_dataset, ch_names = build_Benchmark_dataset_leave_one_out(
            root_dir=args.data_root, winLEN=args.win_len, test_sub=sub, filterbank_num=args.filterbank)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

        model = create_model(args.model, num_classes=40)
        model.to(device)
        if args.init_ckpt:
            print(f'Loading pretrained weights from {args.init_ckpt} (skip cls_head)...')
            checkpoint = torch.load(args.init_ckpt, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('cls_head'):
                    continue
                new_state_dict[k] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f'Load result: {msg}')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_scaler = NativeScaler()

        total_train_steps = len(train_loader)
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, total_train_steps,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
        wd_schedule_values = None

        best_acc = 0.0
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, data_loader=train_loader, optimizer=optimizer,
                                          device=device, epoch=epoch, loss_scaler=loss_scaler,
                                          lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                                          start_steps=epoch * total_train_steps, ch_names=ch_names, args=args)
            test_stats = evaluate(model=model, data_loader=test_loader, device=device,
                                  header=f'Test S{sub}', ch_names=ch_names)
            best_acc = max(best_acc, test_stats['acc'])
            print(f'Epoch {epoch}: train acc {train_stats.get("acc", 0):.4f}, '
                  f'test acc {test_stats["acc"]:.4f}, best {best_acc:.4f}')

        all_acc.append(best_acc)
        print(f'Fold S{sub} best acc: {best_acc:.4f}')

    mean_acc = float(np.mean(all_acc)) if len(all_acc) > 0 else 0.0
    print(f'Overall mean accuracy across {len(all_acc)} folds: {mean_acc:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
