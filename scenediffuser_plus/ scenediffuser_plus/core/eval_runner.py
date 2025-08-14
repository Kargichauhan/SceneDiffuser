# eval_runner.py
import os, sys, argparse, inspect, torch
from typing import Dict, Any, Tuple, Iterable
import eval_kernel
from eval_kernel import pairwise_collision_rate, ade_fde, write_metrics_csv

print("USING eval_kernel FROM:", eval_kernel.__file__)

# ------------------------------
# Utilities to load model & data
# ------------------------------

def try_import_real_components():
    """
    Try to import your real model + dataloader. Edit these imports to match your repo.
    Return (ModelClass, make_eval_loader_fn) or (None, None) if not available.
    """
    try:
        from components import SceneDiffuserPlusPlus as ModelClass  # e.g., components.py
        from evaluation import make_eval_loader                     # e.g., evaluation.py
        return ModelClass, make_eval_loader
    except Exception as e:
        print(f"i Could not import real components ({e}). Falling back to dummy loader.")
        return None, None


class DummyEval(torch.utils.data.Dataset):
    """Fallback dataset to test the plumbing if you haven't wired the real loader yet."""
    def __init__(self, num_scenes: int = 10, A: int = 12, T: int = 50):
        self.num_scenes, self.A, self.T = num_scenes, A, T
    def __len__(self): return self.num_scenes
    def __getitem__(self, idx: int):
        A, T = self.A, self.T
        future_xy = torch.randn(A, T, 2) * 0.5
        valid = torch.ones(A, T, dtype=torch.bool)
        context = {"history": None, "map": None}
        return {"future_xy": future_xy, "valid": valid, "context": context}


def make_fallback_loader(batch_size: int = 1) -> torch.utils.data.DataLoader:
    ds = DummyEval()
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


class SampleAdapter(torch.nn.Module):
    """
    Wrap any torch.nn.Module to guarantee a .sample(ctx) method that returns [A,T,2]
    or [B,A,T,2]. If your model already has .sample, we call it; otherwise we call forward.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def sample(self, context: Dict[str, Any]) -> torch.Tensor:
        if hasattr(self.model, "sample"):
            out = self.model.sample(context)
        else:
            try:
                out = self.model(context)
            except Exception:
                A = 12; T = 50
                out = torch.zeros(A, T, 2)
        return out


def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    ModelClass, _ = try_import_real_components()
    if ModelClass is None:
        class DummyModel(torch.nn.Module):
            @torch.inference_mode()
            def sample(self, ctx):  # [A,T,2]
                A = 12; T = 50
                return torch.zeros(A, T, 2)
        model = DummyModel()
        return SampleAdapter(model).to(device).eval()

    model = ModelClass()
    obj = torch.load(ckpt_path, map_location=device)
    state = obj.get("state_dict", obj.get("model", obj))
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"WARNING load_state_dict(strict=False) failed ({e}); trying strict=True")
        model.load_state_dict(state, strict=True)
    return SampleAdapter(model).to(device).eval()


def make_dataloader(batch_size: int = 1) -> torch.utils.data.DataLoader:
    _, make_eval_loader = try_import_real_components()
    if make_eval_loader is not None:
        try:
            return make_eval_loader(split="val", batch_size=batch_size)
        except TypeError:
            return make_eval_loader()
    return make_fallback_loader(batch_size=batch_size)


# ------------------------------
# Evaluation loop
# ------------------------------

@torch.inference_mode()
def _per_item_metrics(pred_xy: torch.Tensor, gt_xy: torch.Tensor, valid: torch.Tensor, radius: float) -> Tuple[float,float,float]:
    """
    pred_xy: [A,T,2], gt_xy: [A,T,2], valid: [A,T]
    returns: (CR%, ADE, FDE)
    """
    dev = valid.device
    if pred_xy.device != dev:
        pred_xy = pred_xy.to(dev)
    if gt_xy.device != dev:
        gt_xy = gt_xy.to(dev)

    cr = pairwise_collision_rate(pred_xy, valid, radius=radius)
    ade, fde = ade_fde(pred_xy, gt_xy, valid)
    return cr, ade, fde



@torch.inference_mode()
def run_eval(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: str,
             method_name: str,
             out_csv: str,
             radius: float):
    model.to(device).eval()
    rows = []
    scene_id = 0

    for batch in loader:
        gt = batch["future_xy"]
        valid = batch["valid"]
        if gt.ndim == 4:
            B, A, T, _ = gt.shape
        elif gt.ndim == 3:
            B = 1
            A, T, _ = gt.shape
            gt = gt.unsqueeze(0)
            valid = valid.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected gt shape {gt.shape}")

        gt = gt.to(device)
        valid = valid.to(device).bool()
        ctx = batch.get("context", {})
        if isinstance(ctx, dict):
            for k, v in list(ctx.items()):
                if torch.is_tensor(v): ctx[k] = v.to(device)

        pred = model.sample(ctx)  # [A,T,2] or [B,A,T,2]
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        elif pred.ndim != 4:
            raise ValueError(f"Unexpected pred shape {pred.shape}")

        if pred.shape[0] == 1 and gt.shape[0] > 1:
            pred = pred.expand(gt.shape[0], -1, -1, -1)

        pred = pred.to(device)  # <<< IMPORTANT: fix device mismatch

        for b in range(gt.shape[0]):
            cr, ade, fde = _per_item_metrics(pred[b], gt[b], valid[b], radius)
            rows.append([scene_id, method_name, cr, ade, fde])
            scene_id += 1

    os.makedirs("results", exist_ok=True)
    write_metrics_csv(rows, out_csv)
    print(f"✅ wrote {out_csv} ({len(rows)} scenes)")
    return rows


# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", required=True, choices=["Baseline","Ours"])
    p.add_argument("--out", default="results/metrics_tmp.csv")
    p.add_argument("--device", default="mps")  # or "cuda" / "cpu"
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--radius", type=float, default=4.0)
    args = p.parse_args()

    model = load_model(args.ckpt, args.device)
    loader = make_dataloader(batch_size=args.batch_size)
    run_eval(model, loader, args.device, args.method, args.out, args.radius)
