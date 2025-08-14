# eval_kernel.py
import os, csv, torch

@torch.inference_mode()
def pairwise_collision_rate(pred_xy: torch.Tensor, validity: torch.Tensor, radius: float = 4.0) -> float:
    """
    pred_xy: [A,T,2] predicted positions in meters
    validity: [A,T] bool mask for agent presence
    returns: collision rate in percent (0..100)
    """
    A, T, _ = pred_xy.shape
    collisions, pairs = 0, 0
    for t in range(T):
        idx = validity[:, t].nonzero(as_tuple=True)[0]   # indices of present agents at t
        n = idx.numel()
        if n <= 1:
            continue
        d = torch.cdist(pred_xy[idx, t], pred_xy[idx, t])  # [n,n]
        triu = torch.triu(d, diagonal=1)                   # upper triangle: no double count, no self
        collisions += (triu < radius).sum().item()
        pairs += n * (n - 1) // 2
    return 100.0 * collisions / max(pairs, 1)

@torch.inference_mode()
def ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor, validity: torch.Tensor):
    """
    pred_xy, gt_xy: [A,T,2]
    validity: [A,T] bool
    returns: (ADE meters, FDE meters)
    """
    diff = (pred_xy - gt_xy).norm(dim=-1)   # [A,T]
    mask = validity.bool()
    ade = diff[mask].mean().item() if mask.any() else float("nan")

    fdes = []
    A, T = validity.shape
    for a in range(A):
        last = validity[a].nonzero(as_tuple=True)[0]      # <-- this was the line with the broken '('
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
