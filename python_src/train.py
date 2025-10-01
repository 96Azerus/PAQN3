# python_src/train.py

# ✅ КРИТИЧЕСКИ ВАЖНО: Никаких тяжелых импортов (torch, aim) на глобальном уровне!
import os
import sys
import time
import numpy as np
import traceback
from collections import deque, defaultdict
import multiprocessing as mp
import queue
import random
import glob
import subprocess
import shutil
import psutil
import threading
from multiprocessing import shared_memory

# ✅ Устанавливаем spawn-метод ПЕРЕД любыми другими действиями с mp
if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

# --- КОНСТАНТЫ И ПУТИ ---
NUM_INFERENCE_WORKERS = 56
NUM_CPP_WORKERS = 56
print(f"⚡ Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

ACTION_LIMIT = 100
LEARNING_RATE = 0.0001
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 512
MIN_BUFFER_FILL_SAMPLES = 50000
POLICY_WEIGHT_START = 0.2
POLICY_WEIGHT_END = 1.0
POLICY_WEIGHT_SCHEDULE_STEPS = 100000
VALUE_CLIP_VALUE = 50.0
INFERENCE_MAX_BATCH_SIZE = 256
INFERENCE_BATCH_TIMEOUT_MS = 0.5
FIRST_STREET_CANDIDATES = 2000
STATS_INTERVAL_SECONDS = 15
SAVE_INTERVAL_STEPS = 100
GIT_PUSH_INTERVAL_STEPS = 500
BASE_DIR = "/kaggle/working"
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "local_models")
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "paqn_model_latest.pth")
VERSION_FILE = os.path.join(LOCAL_MODEL_DIR, "latest_version.txt")
LOCAL_OPPONENT_POOL_DIR = os.path.join(LOCAL_MODEL_DIR, "opponent_pool")
MAX_OPPONENTS_IN_POOL = 20
GIT_REPO_OWNER = "Azerus96"
GIT_REPO_NAME = "PAQN3"
GIT_BRANCH = "main"
PUSH_REPO_DIR = os.path.join(BASE_DIR, "PAQN3_for_push")

# --- Вспомогательные функции ---
def run_git_command(command, repo_path):
    try:
        subprocess.run(command, cwd=repo_path, check=True, capture_output=True, text=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        output = e.stderr if hasattr(e, 'stderr') else str(e)
        print(f"Git command failed: {' '.join(command)}\nError: {output}")
        return False

def git_push(commit_message, auth_repo_url):
    print(f"\n--- Attempting to push to GitHub: '{commit_message}' ---")
    if os.path.exists(PUSH_REPO_DIR): shutil.rmtree(PUSH_REPO_DIR)
    if not run_git_command(["git", "clone", auth_repo_url, PUSH_REPO_DIR], BASE_DIR): return
    if os.path.exists(MODEL_PATH): shutil.copy2(MODEL_PATH, os.path.join(PUSH_REPO_DIR, "paqn_model_latest.pth"))
    opponent_pool_git_path = os.path.join(PUSH_REPO_DIR, "opponent_pool")
    os.makedirs(opponent_pool_git_path, exist_ok=True)
    if os.path.exists(LOCAL_OPPONENT_POOL_DIR):
        for f in glob.glob(os.path.join(LOCAL_OPPONENT_POOL_DIR, "*.pth")):
            shutil.copy2(f, opponent_pool_git_path)
    if not run_git_command(["git", "add", "."], PUSH_REPO_DIR): return
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=PUSH_REPO_DIR, capture_output=True, text=True)
    if not status_result.stdout.strip():
        print("No changes to commit."); shutil.rmtree(PUSH_REPO_DIR); return
    if not run_git_command(["git", "commit", "-m", commit_message], PUSH_REPO_DIR): return
    if not run_git_command(["git", "push", "origin", f"HEAD:{GIT_BRANCH}"], PUSH_REPO_DIR): return
    print("--- Push successful ---"); shutil.rmtree(PUSH_REPO_DIR)

def git_pull(repo_path, auth_repo_url):
    print("\n--- Pulling latest model from GitHub ---")
    if not run_git_command(["git", "pull", auth_repo_url, GIT_BRANCH], repo_path):
        print("Git pull failed. Continuing with local version if available.")

def update_opponent_pool(model_version):
    if not os.path.exists(MODEL_PATH): return
    os.makedirs(LOCAL_OPPONENT_POOL_DIR, exist_ok=True)
    new_opponent_path = os.path.join(LOCAL_OPPONENT_POOL_DIR, f"paqn_model_v{model_version}.pth")
    try:
        shutil.copy2(MODEL_PATH, new_opponent_path)
        print(f"Added model version {model_version} to local opponent pool.")
    except Exception as e: print(f"Error updating opponent pool: {e}"); return
    pool_files = sorted(glob.glob(os.path.join(LOCAL_OPPONENT_POOL_DIR, "*.pth")), key=os.path.getmtime)
    while len(pool_files) > MAX_OPPONENTS_IN_POOL:
        try: os.remove(pool_files.pop(0)); print(f"Removed oldest opponent from pool.")
        except OSError as e: print(f"Warning: Could not remove old opponent file: {e}")

def get_params_for_optimizer(model, base_lr, weight_decay, head_lr_mult=2.0, head_wd=0.0):
    import torch.nn as nn
    head_names = ["value_head", "action_proj", "street_proj", "policy_head_fc", "body_ln", "action_ln", "street_ln"]
    params_body_decay, params_body_no_decay, params_head_decay, params_head_no_decay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        is_head = any(h_name in name for h_name in head_names)
        is_no_decay = param.dim() <= 1 or name.endswith(".bias") or "norm" in name
        if is_head: (params_head_no_decay if is_no_decay else params_head_decay).append(param)
        else: (params_body_no_decay if is_no_decay else params_body_decay).append(param)
    return [
        {'params': params_body_decay, 'weight_decay': weight_decay, 'lr': base_lr},
        {'params': params_body_no_decay, 'weight_decay': 0.0, 'lr': base_lr},
        {'params': params_head_decay, 'weight_decay': head_wd, 'lr': base_lr * head_lr_mult},
        {'params': params_head_no_decay, 'weight_decay': 0.0, 'lr': base_lr * head_lr_mult},
    ]

def initialize_model_and_state(model, optimizer, device, auth_repo_url):
    import torch
    model_version, global_step = 0, 0
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            global_step = state_dict.get('global_step', 0)
            model_version = state_dict.get('model_version', 0)
            print(f"✅ Loaded model, optimizer, and state. Resuming from step {global_step}, version {model_version}")
            return model_version, global_step
        except Exception as e:
            print(f"FATAL: Local model file is corrupted. Error: {e}. Exiting."); sys.exit(1)
    print("Local model not found. Attempting to pull from GitHub...")
    git_pull(os.path.join(BASE_DIR, GIT_REPO_NAME), auth_repo_url)
    if os.path.exists(MODEL_PATH):
        return initialize_model_and_state(model, optimizer, device, auth_repo_url)
    print("No model found. Starting from scratch and saving initial model.")
    try:
        torch.save({'global_step': 0, 'model_version': 0, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, MODEL_PATH)
        with open(VERSION_FILE, 'w') as f: f.write('0')
        print("✅ Initial model saved successfully.")
    except Exception as e:
        print(f"FATAL: Could not save initial model: {e}"); sys.exit(1)
    return model_version, global_step

class InferenceWorker(mp.Process):
    def __init__(self, name, task_queue, result_shm_info, log_queue, stop_event):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_shm_info = result_shm_info
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.model_version = -1
        self.last_version_check_time = 0
        self.inference_count = 0
        self.total_items_in_batch = 0
        self.batch_count = 0
        self.last_throughput_log = time.time()

    def _log(self, message):
        self.log_queue.put(f"[{self.name}] {message}")

    def _initialize(self):
        self._log("Started.")
        import torch
        from python_src.model import OFC_CNN_Network
        
        self.torch = torch
        self.device = torch.device("cpu")
        
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        self.latest_model = OFC_CNN_Network().to(self.device)
        self.opponent_model = OFC_CNN_Network().to(self.device)
        self._load_models()
        
        self.latest_model.eval()
        self.opponent_model.eval()
        
        if hasattr(torch, 'compile'):
            try:
                self._log("Compiling models with torch.compile...")
                self.latest_model = torch.compile(self.latest_model, mode='reduce-overhead')
                self.opponent_model = torch.compile(self.opponent_model, mode='reduce-overhead')
                self._log("✅ Models compiled successfully!")
            except Exception as e:
                self._log(f"⚠️ torch.compile failed: {e}, continuing without compilation")
        
        shm_name, shape, dtype = self.result_shm_info
        self.result_shm = shared_memory.SharedMemory(name=shm_name)
        self.result_array = np.ndarray(shape, dtype=dtype, buffer=self.result_shm.buf)

    def _load_models(self):
        try:
            if os.path.exists(MODEL_PATH):
                state_dict = self.torch.load(MODEL_PATH, map_location=self.device)
                self.latest_model.load_state_dict(state_dict.get('model_state_dict', state_dict))
                self.model_version = state_dict.get('model_version', -1)
                self._log(f"Loaded latest model (version {self.model_version}).")
            else:
                self._log(f"FATAL: No latest model found at {MODEL_PATH}."); os._exit(1)
        except Exception as e:
            self._log(f"!!! FATAL EXCEPTION during model loading: {e}. Worker stopping."); os._exit(1)
        
        try:
            opponent_pool_files = glob.glob(os.path.join(LOCAL_OPPONENT_POOL_DIR, "*.pth"))
            if opponent_pool_files:
                opponent_path = random.choice(opponent_pool_files)
                state_dict = self.torch.load(opponent_path, map_location=self.device)
                self.opponent_model.load_state_dict(state_dict.get('model_state_dict', state_dict))
                self._log(f"Loaded opponent model: {os.path.basename(opponent_path)}")
            else:
                self.opponent_model.load_state_dict(self.latest_model.state_dict())
                self._log("Opponent pool is empty, using latest model as opponent.")
        except Exception as e: self._log(f"!!! EXCEPTION during opponent model loading: {e}")

    def _check_for_updates(self):
        if time.time() - self.last_version_check_time < 5: return
        self.last_version_check_time = time.time()
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, 'r') as f: latest_version = int(f.read())
                if latest_version > self.model_version:
                    time.sleep(int(self.name.split('-')[-1]) * 0.1)
                    self._log(f"New model version detected ({latest_version}). Reloading...")
                    self._load_models()
        except (IOError, ValueError) as e: self._log(f"Could not check for model update: {e}")

    def collect_batch(self):
        batch = []
        timeout = INFERENCE_BATCH_TIMEOUT_MS / 1000.0
        try:
            first_req = self.task_queue.get(timeout=timeout)
            batch.append(first_req)
        except queue.Empty:
            return batch
        while len(batch) < INFERENCE_MAX_BATCH_SIZE and not self.task_queue.empty():
            try: batch.append(self.task_queue.get_nowait())
            except queue.Empty: break
        return batch

    def process_batch(self, batch):
        if not batch: return
        groups = defaultdict(list)
        for req in batch: groups['latest' if req[3] else 'opponent'].append(req)
        with self.torch.inference_mode():
            for model_key, reqs in groups.items():
                model = self.latest_model if model_key == 'latest' else self.opponent_model
                infosets = self.torch.tensor([r[1] for r in reqs], dtype=self.torch.float32, device=self.device).view(-1, 16, 4, 13)
                body_outputs = model.forward_body(infosets)
                values = model.forward_value_head(body_outputs)
                policy_req_indices = [i for i, r in enumerate(reqs) if r[2] is not None]
                for i, req in enumerate(reqs):
                    req_id = req[0]
                    self.result_array[req_id, 0] = values[i].item()
                    if i not in policy_req_indices: self.result_array[req_id, 1] = 1
                if policy_req_indices:
                    action_vectors, splits = [], []
                    for i in policy_req_indices:
                        action_vecs_for_req = reqs[i][2]
                        action_vectors.extend(action_vecs_for_req)
                        splits.append(len(action_vecs_for_req))
                    action_tensor = self.torch.tensor(action_vectors, dtype=self.torch.float32, device=self.device)
                    repeat_counts = self.torch.tensor(splits, device=self.device)
                    repeated_body_outputs = self.torch.repeat_interleave(body_outputs[policy_req_indices], repeat_counts, dim=0)
                    repeated_infosets = self.torch.repeat_interleave(infosets[policy_req_indices], repeat_counts, dim=0)
                    street_tensor = repeated_infosets[:, 9:14, 0, 0]
                    logits = model.forward_policy_head(repeated_body_outputs, action_tensor, street_tensor)
                    results_flat = logits.cpu().numpy().flatten()
                    current_pos = 0
                    for i, num_actions in enumerate(splits):
                        req_id = reqs[policy_req_indices[i]][0]
                        self.result_array[req_id, 2:2+num_actions] = results_flat[current_pos : current_pos + num_actions]
                        self.result_array[req_id, 1] = 1
                        current_pos += num_actions
        self.inference_count += len(batch)
        self.total_items_in_batch += sum(len(r[2]) if r[2] is not None else 1 for r in batch)
        self.batch_count += 1
        now = time.time()
        if now - self.last_throughput_log > 60:
            elapsed = now - self.last_throughput_log
            throughput = self.inference_count / elapsed
            avg_batch_size = self.total_items_in_batch / self.batch_count if self.batch_count > 0 else 0
            self._log(f"📊 Throughput: {throughput:.1f} req/s, Avg batch items: {avg_batch_size:.1f}")
            self.inference_count = self.total_items_in_batch = self.batch_count = 0
            self.last_throughput_log = now

    def run(self):
        try:
            self._initialize()
            while not self.stop_event.is_set():
                batch = self.collect_batch()
                if batch: self.process_batch(batch)
                else: self._check_for_updates()
        except (KeyboardInterrupt, SystemExit): pass
        except Exception: self._log(f"---!!! FATAL EXCEPTION IN {self.name} !!!---\n{traceback.format_exc()}")
        finally:
            self._log("Stopped.")
            if hasattr(self, 'result_shm'): self.result_shm.close()

def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    import torch
    import torch.optim as optim
    from torch.nn.utils import clip_grad_norm_
    import aim
    from python_src.model import OFC_CNN_Network
    from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    build_dir = os.path.join(project_root, 'build')
    if project_root not in sys.path: sys.path.insert(0, project_root)
    if build_dir not in sys.path: sys.path.insert(0, build_dir)

    with mp.Manager() as manager:
        aim_run = aim.Run(experiment="paqn_ofc_poker_v4_stable")
        aim_run["hparams"] = { "num_cpp_workers": NUM_CPP_WORKERS, "num_inference_workers": NUM_INFERENCE_WORKERS, "learning_rate": LEARNING_RATE, "buffer_capacity": BUFFER_CAPACITY, "batch_size": BATCH_SIZE, "policy_weight_start": POLICY_WEIGHT_START, "policy_weight_end": POLICY_WEIGHT_END, "policy_weight_schedule": POLICY_WEIGHT_SCHEDULE_STEPS, "value_clip_value": VALUE_CLIP_VALUE, "head_lr_mult": 2.0, "head_wd": 0.0, "head_warmup_steps": 2000, "inference_max_batch_size": INFERENCE_MAX_BATCH_SIZE, "inference_batch_timeout_ms": INFERENCE_BATCH_TIMEOUT_MS }

        def monitor_resources():
            p = psutil.Process(os.getpid())
            while True:
                try:
                    rss_gb = p.memory_info().rss / 1024**3
                    threads = p.num_threads()
                    print(f"[MONITOR] RSS={rss_gb:.2f} GB, Threads={threads}", flush=True)
                    if aim_run.active:
                        aim_run.track(rss_gb, name="system/memory_rss_gb")
                        aim_run.track(threads, name="system/num_threads")
                    time.sleep(15)
                except (psutil.NoSuchProcess, KeyboardInterrupt): break
        
        threading.Thread(target=monitor_resources, daemon=True).start()

        git_username = os.environ.get('GIT_USERNAME')
        git_token = os.environ.get('GIT_TOKEN')
        if not git_username or not git_token: print("ERROR: GIT_USERNAME and GIT_TOKEN must be set."); sys.exit(1)
        
        auth_repo_url = f"https://{git_username}:{git_token}@github.com/{GIT_REPO_OWNER}/{GIT_REPO_NAME}.git"
        run_git_command(["git", "config", "--global", "user.email", f"{git_username}@users.noreply.github.com"], project_root)
        run_git_command(["git", "config", "--global", "user.name", git_username], project_root)
        
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        os.makedirs(LOCAL_OPPONENT_POOL_DIR, exist_ok=True)
        
        print("Initializing C++ hand evaluator...", flush=True)
        initialize_evaluator()
        print("C++ evaluator initialized.", flush=True)

        device = torch.device("cpu")
        model = OFC_CNN_Network().to(device)
        optimizer = optim.AdamW(get_params_for_optimizer(model, LEARNING_RATE, 0.01))
        
        model_version, global_step = initialize_model_and_state(model, optimizer, device, auth_repo_url)
        
        GIT_OPPONENT_POOL_DIR = os.path.join(project_root, "opponent_pool")
        if os.path.exists(GIT_OPPONENT_POOL_DIR):
            print("Syncing opponent pool from Git...")
            for f in glob.glob(os.path.join(GIT_OPPONENT_POOL_DIR, "*.pth")): shutil.copy2(f, LOCAL_OPPONENT_POOL_DIR)
            print(f"Synced {len(os.listdir(LOCAL_OPPONENT_POOL_DIR))} opponents.")

        head_warmup_steps = 2000
        if global_step < head_warmup_steps:
            print(f"!!! HEAD-ONLY WARMUP ENABLED for the next {head_warmup_steps - global_step} steps !!!")
        
        policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
        value_buffer = ReplayBuffer(BUFFER_CAPACITY)
        
        MAX_PENDING_REQUESTS = NUM_CPP_WORKERS * 4 
        RESULT_ROW_SIZE = FIRST_STREET_CANDIDATES + 2 
        
        request_queue = mp.Queue()
        log_queue = manager.Queue()
        stop_event = mp.Event()

        with shared_memory.SharedMemory(create=True, size=MAX_PENDING_REQUESTS * RESULT_ROW_SIZE * np.dtype(np.float32).itemsize) as shm:
            result_array = np.ndarray((MAX_PENDING_REQUESTS, RESULT_ROW_SIZE), dtype=np.float32, buffer=shm.buf)
            result_array.fill(0)
            
            print(f"🚀 Starting {NUM_INFERENCE_WORKERS} InferenceWorkers...", flush=True)
            inference_workers = [InferenceWorker(f"InferenceWorker-{i}", request_queue, (shm.name, result_array.shape, result_array.dtype), log_queue, stop_event) for i in range(NUM_INFERENCE_WORKERS)]
            for w in inference_workers: w.start()
            
            time.sleep(5)

            print(f"Creating C++ SolverManager with {NUM_CPP_WORKERS} workers...", flush=True)
            solver_manager = SolverManager(
                num_workers=NUM_CPP_WORKERS, action_limit=ACTION_LIMIT,
                policy_buffer=policy_buffer, value_buffer=value_buffer,
                request_queue=request_queue, result_array=result_array, log_queue=log_queue,
                first_street_candidates=FIRST_STREET_CANDIDATES,
                max_pending_requests=MAX_PENDING_REQUESTS
            )
            solver_manager.start()
            print("C++ workers are running.", flush=True)
            
            policy_losses, value_losses = deque(maxlen=100), deque(maxlen=100)
            last_stats_time = time.time()
            training_started = False
            min_fill = BATCH_SIZE * 4 if global_step > 0 else MIN_BUFFER_FILL_SAMPLES
            print(f"Training will start when buffer size reaches {min_fill:,} samples.")
            last_save_step, last_push_step = global_step, global_step

            try:
                while True:
                    if time.time() - last_stats_time > STATS_INTERVAL_SECONDS:
                        while not log_queue.empty():
                            try: print(log_queue.get(timeout=0.01), flush=True)
                            except queue.Empty: break
                        
                        total_generated = policy_buffer.total_generated()
                        avg_p_loss = np.mean(policy_losses) if policy_losses else float('nan')
                        avg_v_loss = np.mean(value_losses) if value_losses else float('nan')
                        
                        print("\n" + "="*20 + " STATS UPDATE " + "="*20, flush=True)
                        print(f"Time: {time.strftime('%H:%M:%S')}", flush=True)
                        print(f"Model Version: {model_version}", flush=True)
                        print(f"Global Step: {global_step}", flush=True)
                        print(f"Total Generated: {total_generated:,}", flush=True)
                        print(f"Buffer Fill -> Policy: {policy_buffer.size():,}/{BUFFER_CAPACITY:,} ({policy_buffer.size()/BUFFER_CAPACITY:.1%}) | Value: {value_buffer.size():,}/{BUFFER_CAPACITY:,} ({value_buffer.size()/BUFFER_CAPACITY:.1%})", flush=True)
                        print(f"Avg Losses (last 100) -> Policy: {avg_p_loss:.6f} | Value: {avg_v_loss:.6f}", flush=True)
                        print(f"Request Queue: {request_queue.qsize()}", flush=True)
                        print("="*54, flush=True)
                        
                        if aim_run.active:
                            aim_run.track(total_generated, name="system/total_samples_generated", step=global_step)
                            aim_run.track(policy_buffer.size(), name="buffer/policy_buffer_size", step=global_step)
                            aim_run.track(value_buffer.size(), name="buffer/value_buffer_size", step=global_step)
                            aim_run.track(request_queue.qsize(), name="system/request_queue_size", step=global_step)
                            if policy_losses: aim_run.track(avg_p_loss, name="loss/policy_loss_avg", step=global_step)
                            if value_losses: aim_run.track(avg_v_loss, name="loss/value_loss_avg", step=global_step)
                        
                        last_stats_time = time.time()

                    if value_buffer.size() < min_fill or policy_buffer.size() < min_fill:
                        if int(time.time()) % 5 == 0:
                            print(f"Waiting for buffer... P: {policy_buffer.size():,}/{min_fill:,} | V: {value_buffer.size():,}/{min_fill:,}", flush=True)
                        time.sleep(1)
                        continue

                    if not training_started: 
                        print("\n🚀 Buffer ready. Starting training...")
                        training_started = True

                    model.train()
                    if head_warmup_steps > 0 and global_step < head_warmup_steps:
                        head_names = ["value_head", "action_proj", "street_proj", "policy_head_fc", "body_ln", "action_ln", "street_ln"]
                        for name, param in model.named_parameters(): 
                            param.requires_grad = any(h in name for h in head_names)
                    else:
                        for param in model.parameters(): 
                            param.requires_grad = True

                    v_batch = value_buffer.sample(BATCH_SIZE)
                    if not v_batch: continue
                    v_infosets_np, _, v_targets_np = v_batch
                    
                    p_batch = policy_buffer.sample(BATCH_SIZE)
                    if not p_batch: continue
                    p_infosets_np, p_actions_np, p_advantages_np = p_batch
                    
                    v_infosets = torch.from_numpy(v_infosets_np).view(-1, 16, 4, 13).to(device)
                    v_targets = torch.from_numpy(v_targets_np).unsqueeze(1).to(device)
                    v_targets_clipped = torch.clamp(v_targets, -VALUE_CLIP_VALUE, VALUE_CLIP_VALUE)
                    
                    p_infosets = torch.from_numpy(p_infosets_np).view(-1, 16, 4, 13).to(device)
                    p_actions = torch.from_numpy(p_actions_np).to(device)
                    p_advantages = torch.from_numpy(p_advantages_np).to(device)
                    
                    adv_ranks = torch.argsort(torch.argsort(p_advantages.squeeze())).float()
                    p_advantages_normalized = (adv_ranks / (adv_ranks.size(0) - 1) - 0.5) * 2.0 if adv_ranks.size(0) > 1 else torch.zeros_like(adv_ranks)
                    p_advantages_normalized = p_advantages_normalized.unsqueeze(1)

                    p_street_vector = p_infosets[:, 9:14, 0, 0]
                    
                    pred_logits, _ = model(p_infosets, p_actions, p_street_vector)
                    pred_values_for_value = model(v_infosets)
                    
                    loss_v = torch.nn.functional.huber_loss(pred_values_for_value, v_targets_clipped, delta=1.0)
                    loss_p = torch.nn.functional.huber_loss(pred_logits, p_advantages_normalized, delta=1.0)

                    optimizer.zero_grad()
                    current_policy_weight = min(POLICY_WEIGHT_END, POLICY_WEIGHT_START + (POLICY_WEIGHT_END - POLICY_WEIGHT_START) * (global_step / POLICY_WEIGHT_SCHEDULE_STEPS))
                    total_loss = loss_v + current_policy_weight * loss_p
                    total_loss.backward()
                    grad_norm = clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    
                    value_losses.append(loss_v.item())
                    policy_losses.append(loss_p.item())
                    global_step += 1
                    
                    if aim_run.active:
                        aim_run.track(loss_v.item(), name="loss/value_loss", step=global_step)
                        aim_run.track(loss_p.item(), name="loss/policy_loss", step=global_step)
                        aim_run.track(grad_norm.item(), name="diagnostics/grad_norm", step=global_step)
                        aim_run.track(current_policy_weight, name="hparams/current_policy_weight", step=global_step)
                        with torch.no_grad():
                            aim_run.track(float(v_targets.std()), name="targets/value_raw/std", step=global_step)
                            aim_run.track(float(p_advantages.std()), name="targets/advantage_raw/std", step=global_step)
                            var_t = float(torch.var(v_targets_clipped))
                            ev = 1.0 - float(torch.var(v_targets_clipped - pred_values_for_value)) / max(var_t, 1e-6)
                            aim_run.track(ev, name="diagnostics/value_explained_var", step=global_step)

                    is_first_save = (global_step >= 100) and (last_save_step < 100)
                    is_regular_save = (global_step - last_save_step) >= SAVE_INTERVAL_STEPS
                    
                    if training_started and (is_first_save or is_regular_save):
                        print(f"\n--- Saving model at step {global_step} ---", flush=True)
                        model_version += 1
                        torch.save({'global_step': global_step, 'model_version': model_version, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, MODEL_PATH + ".tmp")
                        os.rename(MODEL_PATH + ".tmp", MODEL_PATH)
                        with open(VERSION_FILE, 'w') as f: f.write(str(model_version))
                        update_opponent_pool(model_version)
                        last_save_step = global_step
                    
                    if training_started and (global_step - last_push_step) >= GIT_PUSH_INTERVAL_STEPS:
                        git_push(f"Periodic save: v{model_version}, step {global_step}", auth_repo_url)
                        last_push_step = global_step

            except KeyboardInterrupt: 
                print("\n⚠️ Training interrupted by user.", flush=True)
            finally:
                print("\n" + "="*15 + " SHUTDOWN PROCEDURE STARTED " + "="*15, flush=True)
                if 'aim_run' in locals() and aim_run.active: 
                    print("1. Closing Aim session...", flush=True)
                    aim_run.close()
                    print("   ✅ Aim session closed.", flush=True)
                print("2. Sending stop signal to all workers...", flush=True)
                stop_event.set()
                print("3. Stopping C++ workers...", flush=True)
                if 'solver_manager' in locals(): solver_manager.stop()
                print("   ✅ C++ workers stopped.", flush=True)
                print("4. Stopping Python workers...", flush=True)
                if 'inference_workers' in locals():
                    for w in inference_workers:
                        w.join(timeout=5)
                        if w.is_alive(): print(f"   - Force terminating {w.name}...", flush=True); w.terminate()
                print("   ✅ Python workers stopped.", flush=True)
                if training_started:
                    print("5. Final model save and push...", flush=True)
                    try:
                        torch.save({'global_step': global_step, 'model_version': model_version, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, MODEL_PATH)
                        print("   ✅ Model saved locally.", flush=True)
                        git_push(f"Final save on exit: v{model_version}, step {global_step}", auth_repo_url)
                    except Exception as e: print(f"   ---! ❌ ERROR on final save/push: {e}", flush=True)
                print("="*58)
                print("✅ Training process finished correctly.")

if __name__ == "__main__":
    main()
