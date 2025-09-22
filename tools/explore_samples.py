# tools/explore_samples.py
import os, glob, torch, numpy as np
from collections import Counter

SAMPLES_DIR = os.environ.get("DUMP_DIR", "/content/local_models/samples")

def main():
    files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.pt")))
    print(f"Found {len(files)} samples in {SAMPLES_DIR}")
    if not files:
        return

    nz_counts, K_list = [], []
    for p in files:
        s = torch.load(p, map_location="cpu")
        x = s["infoset"]  # [1,16,4,13]
        nz_counts.append(int((x.numpy() != 0).sum()))
        a = s.get("actions")
        if a is not None:
            K_list.append(a.shape[0])

    print("Infoset nonzero: min/median/max =", np.min(nz_counts), np.median(nz_counts), np.max(nz_counts))
    if K_list:
        from collections import Counter
        print("K actions distribution:", Counter(K_list))
    else:
        print("No actions in saved samples (value-only requests?)")

    # Печатаем подробности последнего файла
    last = files[-1]
    print("\nLast sample:", last)
    s = torch.load(last, map_location="cpu")
    for k, v in s.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Пример: сколько единиц в каждом action-канд. (если есть)
    if s.get("actions") is not None:
        a = s["actions"]
        ones_per_action = (a > 0).sum(dim=1).tolist()
        print("Ones per action (first 20):", ones_per_action[:20])

    # Экспорт одного семпла в NPZ (опционально)
    if os.environ.get("EXPORT_NPZ", "0") == "1":
        npz_path = os.path.join(SAMPLES_DIR, "sample_export.npz")
        np.savez(npz_path,
                 infoset=s["infoset"].numpy(),
                 actions=(s["actions"].numpy() if s["actions"] is not None else None),
                 street=s["street"].numpy())
        print("Exported to:", npz_path)

if __name__ == "__main__":
    main()
