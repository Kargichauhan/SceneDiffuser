import os, csv, torch

@torch.inference_mode()
def pairwise_collision_rate(pred_xy: torch.Tensor, validity: torch.Tensor, radius: float = 4.0) -> float:
    A, T, _ = pred_xy.shape
    collisions, pairs = 0, 0
    for t in range(T):
        idx = validity[:, t].nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n <= 1:
            continue
        d = torch.cdist(pred_xy[idx, t], pred_xy[idx, t])
        triu = torch.triu(d, diagonal=1)
        collisions += (triu < radius).sum().item()
        pairs += n * (n - 1) // 2
    return 100.0 * collisions / max(pairs, 1)

@torch.inference_mode()
def ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor, validity: torch.Tensor):
    diff = (pred_xy - gt_xy).norm(dim=-1)
    mask = validity.bool()
    ade = diff[mask].mean().item() if mask.any() else float("nan")
    fdes, (A, T) = [], validity.shape
    for a in range(A):
        last = validity[a].nonzero(as_tuple=True)[0]
        if len(last):
            fdes.append(diff[a, last[-1]].item())
    fde = sum(fdes) / len(fdes) if fdes else float("nan")
    return ade, fde

def write_metrics_csv(rows, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "method", "CR_%", "ADE_m", "FDE_m"])
        w.writerows(rows)
