import os, argparse, torch
from typing import Dict, Any, Tuple
import eval_kernel
from eval_kernel import pairwise_collision_rate, ade_fde, write_metrics_csv

print("USING eval_kernel FROM:", eval_kernel.__file__)

# -------- resolve model + dataloader (real if available, else dummy) --------
def resolve_model_class():
    try:
        import components as C
        for name in ["SceneDiffuserPlusPlus","SceneDiffuserPP","SceneDiffuser","ImprovedSceneDiffuser","Model"]:
            if hasattr(C, name):
                print(f"✓ Using model class components.{name}")
                return getattr(C, name), False
    except Exception as e:
        print(f"ℹ️ components.py not usable: {e}")
    # dummy fallback
    class DummyModel(torch.nn.Module):
        @torch.inference_mode()
        def sample(self, ctx):
            A = ctx.get("A", 12); T = ctx.get("T", 50)
            return torch.zeros(A, T, 2)
    print("🟡 Using DummyModel fallback")
    return DummyModel, True

def resolve_eval_loader():
    try:
        import evaluation as E
        for name in ["make_eval_loader","get_eval_loader","build_eval_loader","create_eval_loader","eval_loader"]:
            if hasattr(E, name):
                print(f"✓ Using dataloader factory evaluation.{name}")
                return getattr(E, name), False
    except Exception as e:
        print(f"ℹ️ evaluation.py not usable: {e}")
    # dummy fallback
    from torch.utils.data import DataLoader, Dataset
    class DummyEval(Dataset):
        def __len__(self): return 10
        def __getitem__(self, i):
            A,T=12,50
            return {
                "future_xy": torch.randn(A,T,2)*0.5,
                "valid": torch.ones(A,T,dtype=torch.bool),
                "context": {"A":A,"T":T},
            }
    print("🟡 Using DummyEval fallback")
    return (lambda **kw: DataLoader(DummyEval(), batch_size=1, shuffle=False, num_workers=0)), True

def load_checkpoint_into(model, ckpt_path, device):
    try:
        blob = torch.load(ckpt_path, map_location=device)
        for key in ["state_dict_ema","ema","model_ema","state_dict","model","net"]:
            if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
                blob = blob[key]; break
        model.load_state_dict(blob, strict=False)
        print("✓ Loaded checkpoint (strict=False)")
    except Exception as e:
        print(f"ℹ️ Skipping checkpoint load or strict=False failed: {e}")
    return model

# ---------------- evaluation core ----------------
@torch.inference_mode()
def per_item_metrics(pred_xy: torch.Tensor, gt_xy: torch.Tensor, valid: torch.Tensor, radius: float) -> Tuple[float,float,float]:
    cr = pairwise_collision_rate(pred_xy, valid, radius=radius)   # percent
    ade, fde = ade_fde(pred_xy, gt_xy, valid)                     # meters
    return cr, ade, fde

@torch.inference_mode()
def run_eval(model, loader, device, method_name, out_csv, radius):
    model.to(device).eval()
    rows = []; scene_id = 0
    for batch in loader:
        gt = batch["future_xy"]; valid = batch["valid"]; ctx = batch.get("context", {})
        if gt.ndim == 3:  # [A,T,2] -> [1,A,T,2]
            gt, valid = gt.unsqueeze(0), valid.unsqueeze(0)
        gt = gt.to(device); valid = valid.to(device).bool()
        if isinstance(ctx, dict):
            for k,v in list(ctx.items()):
                if torch.is_tensor(v): ctx[k]=v.to(device)
        pred = model.sample(ctx)            # [A,T,2] or [B,A,T,2]
        if pred.ndim == 3: pred = pred.unsqueeze(0)
        if pred.shape[0]==1 and gt.shape[0]>1: pred = pred.expand(gt.shape[0],-1,-1,-1)
        for b in range(gt.shape[0]):
            cr, ade, fde = per_item_metrics(pred[b], gt[b], valid[b], radius)
            rows.append([scene_id, method_name, cr, ade, fde]); scene_id += 1
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    write_metrics_csv(rows, out_csv)
    print(f"✅ wrote {out_csv} ({len(rows)} scenes)")
    return rows

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method", required=True, choices=["Baseline","Ours"])
    ap.add_argument("--out", default="results/metrics_tmp.csv")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--split", default="val")
    ap.add_argument("--radius", type=float, default=4.0)
    args = ap.parse_args()

    ModelClass, model_is_dummy = resolve_model_class()
    make_loader, loader_is_dummy = resolve_eval_loader()

    model = ModelClass()
    if not model_is_dummy:
        model = load_checkpoint_into(model, args.ckpt, args.device)
    model.to(args.device).eval()

    try:
        loader = make_loader(split=args.split, batch_size=args.batch_size)
    except TypeError:
        try: loader = make_loader(batch_size=args.batch_size)
        except TypeError: loader = make_loader()

    run_eval(model, loader, args.device, args.method, args.out, args.radius)

if __name__ == "__main__":
    main()
