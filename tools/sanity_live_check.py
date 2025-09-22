# tools/sanity_live_check.py
import os, glob, time, random, numpy as np, torch, torch.nn as nn

device = torch.device("cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS, ACTION_SIZE = 16,4,13,208
STREET_START_IDX, STREET_END_IDX = 9, 14

class ResBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(groups, channels)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        id = x
        x = self.act(self.gn1(x)); x = self.conv1(x)
        x = self.act(self.gn2(x)); x = self.conv2(x)
        return x + id

class OFC_CNN_Network(nn.Module):
    def __init__(self, hidden_size=512, channels=64, num_res_blocks=6, groups=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_FEATURE_CHANNELS, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )
        self.res_trunk = nn.Sequential(*[ResBlock(channels, groups) for _ in range(num_res_blocks)])
        conv_output_size = channels*NUM_SUITS*NUM_RANKS
        self.body_fc = nn.Sequential(nn.Linear(conv_output_size, hidden_size), nn.SiLU(inplace=True))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(inplace=True), nn.Linear(hidden_size//2, 1))
        self.action_proj = nn.Sequential(nn.Linear(ACTION_SIZE, 128), nn.SiLU(inplace=True))
        self.street_proj = nn.Sequential(nn.Linear(5, 8), nn.SiLU(inplace=True))
        self.body_ln = nn.LayerNorm(512); self.action_ln = nn.LayerNorm(128); self.street_ln = nn.LayerNorm(8)
        self.policy_head = nn.Sequential(nn.Linear(512+128+8, 512), nn.SiLU(inplace=True), nn.Linear(512,256), nn.SiLU(inplace=True), nn.Linear(256,1))
    def forward(self, state_image, action_vec=None, street_vector=None):
        x = self.stem(state_image); x = self.res_trunk(x)
        body_out = self.body_fc(x.view(x.size(0), -1))
        if action_vec is None: return self.value_head(body_out)
        if street_vector is None: raise ValueError("street_vector required")
        action_embedding = self.action_proj(action_vec); street_embedding = self.street_proj(street_vector)
        z = torch.cat([self.body_ln(body_out), self.action_ln(action_embedding), self.street_ln(street_embedding)], dim=1)
        return self.policy_head(z), None

def find_model():
    candidates = [
        "/content/local_models/paqn_model_latest.pth",
        "/content/PAQN/paqn_model_latest.pth",
        "/content/paqn_model_latest.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def street_from_infoset(x): return x[:, STREET_START_IDX:STREET_END_IDX, 0, 0]

def gen_infoset_live_like(batch, k=None):
    x = torch.zeros(batch, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS)
    for i in range(batch):
        kk = k if k is not None else random.choice([3,4])
        ch = random.sample(list(range(STREET_START_IDX, STREET_END_IDX)), kk)
        x[i, ch, :, :] = 1.0
    return x

def gen_actions_live_like(K):
    A = torch.zeros(K, ACTION_SIZE)
    for i in range(K):
        t = random.choice([3,5])
        idx = torch.randperm(ACTION_SIZE)[:t]
        A[i, idx] = 1.0
    return A

def main():
    model = OFC_CNN_Network().to(device)
    mp = find_model()
    if mp is None:
        print("No model found, using random init.")
    else:
        model.load_state_dict(torch.load(mp, map_location=device), strict=True)
        print("Loaded weights:", mp)
    model.eval()

    samples_dir = os.environ.get("DUMP_DIR", "/content/local_models/samples")
    files = sorted(glob.glob(os.path.join(samples_dir, "*.pt")))
    if files:
        sp = files[-1]
        s = torch.load(sp, map_location="cpu")
        S_base = s["infoset"].float()
        A_base = s["actions"].float() if s.get("actions") is not None else gen_actions_live_like(45)
        print("Using sample:", sp, "| actions shape:", tuple(A_base.shape))
    else:
        print("No samples found, using live-like fallback.")
        S_base = gen_infoset_live_like(1, k=4)
        A_base = gen_actions_live_like(45)

    with torch.no_grad():
        K = A_base.size(0)
        S_fix = S_base.repeat(K,1,1,1)
        ST_fix = street_from_infoset(S_fix)
        logits_a, _ = model(S_fix, A_base, ST_fix)
        std_a = logits_a.std().item()

        S_var = torch.cat([gen_infoset_live_like(64, k=3), gen_infoset_live_like(64, k=4)], dim=0)
        ST_var = street_from_infoset(S_var)
        A_fix = A_base[:1].repeat(S_var.size(0),1)
        logits_s, _ = model(S_var, A_fix, ST_var)
        std_s = logits_s.std().item()

        V3 = model(gen_infoset_live_like(128, k=3)).cpu().numpy().flatten()
        V4 = model(gen_infoset_live_like(128, k=4)).cpu().numpy().flatten()

    print(f"Policy STD by actions: {std_a:.6f} | by states: {std_s:.6f}")
    print(f"Value μ/σ -> k=3: {V3.mean():.6f}/{V3.std():.6f} | k=4: {V4.mean():.6f}/{V4.std():.6f}")

if __name__ == "__main__":
    main()
