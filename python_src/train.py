import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("HEAD_WARMUP_STEPS", "2000") # Requirement 2.2

import sys
import time
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
from collections import deque
import multiprocessing as mp
import queue
import random
import glob
import subprocess
import shutil
import gc
import psutil
import threading
import aim
import json

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
build_dir = os.path.join(project_root, 'build')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from python_src.model import OFC_CNN_Network
from ofc_engine import ReplayBuffer, initialize_evaluator, SolverManager

# --- КОНСТАНТЫ ---
NUM_FEATURE_CHANNELS = 16
NUM_SUITS = 4
NUM_RANKS = 13
INFOSET_SIZE = NUM_FEATURE_CHANNELS * NUM_SUITS * NUM_RANKS

# --- НАСТРОЙКИ ---
NUM_INFERENCE_WORKERS = 24
NUM_CPP_WORKERS = 48
print(f"Configuration: {NUM_CPP_WORKERS} C++ workers, {NUM_INFERENCE_WORKERS} Python inference workers.")

# --- ГИПЕРПАРАМЕТРЫ (Requirement 2.3) ---
ACTION_LIMIT = 100
LEARNING_RATE = 0.0001
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 512
MIN_BUFFER_FILL_SAMPLES = 50000
POLICY_LOSS_WEIGHT = 0.5
ADV_CLIP_VALUE = 5.0
VALUE_CLIP_VALUE = 50.0
VALIDATION_BUFFER_SIZE = 50000
VALIDATION_SPLIT_PROB = 0.01
VALIDATION_INTERVAL_STEPS = 200
RESULT_TTL_SECONDS = 120

# --- ПУТИ И ИНТЕРВАЛЫ ---
STATS_INTERVAL_SECONDS = 15
FIRST_SAVE_STEP = 100
SAVE_INTERVAL_STEPS = 100
GIT_PUSH_INTERVAL_STEPS = 100

LOCAL_MODEL_DIR = "/content/local_models"
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "paqn_model_latest.pth")
VERSION_FILE = os.path.join(LOCAL_MODEL_DIR, "latest_version.txt")
GIT_OPPONENT_POOL_DIR = os.path.join(project_root, "opponent_pool")
LOCAL_OPPONENT_POOL_DIR = os.path.join(LOCAL_MODEL_DIR, "opponent_pool")
MAX_OPPONENTS_IN_POOL = 20
LIVE_METRICS_FILE = os.path.join(LOCAL_MODEL_DIR, "live_metrics.json")
GIT_MODEL_PATH = os.path.join(project_root, "paqn_model_latest.pth")

# --- НАСТРОЙКИ GIT ---
GIT_REPO_OWNER = "Azerus96"
GIT_REPO_NAME = "PAQN"
GIT_BRANCH = "main"

def run_git_command(command, repo_path):
    try:
        result = subprocess.run(command, cwd=repo_path, check=True, capture_output=True, text=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(command)}\nError: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Git command timed out: {' '.join(command)}")
        return False

def git_push(commit_message, repo_path, auth_repo_url):
    print(f"\n--- Attempting to push to GitHub: '{commit_message}' ---")
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, GIT_MODEL_PATH)
        print(f"Copied {MODEL_PATH} to {GIT_MODEL_PATH}.")
    
    os.makedirs(GIT_OPPONENT_POOL_DIR, exist_ok=True)
    if os.path.exists(LOCAL_OPPONENT_POOL_DIR):
        for f in glob.glob(os.path.join(LOCAL_OPPONENT_POOL_DIR, "*.pth")):
            shutil.copy2(f, GIT_OPPONENT_POOL_DIR)
        print(f"Synced local opponent pool to git directory.")

    if not run_git_command(["git", "add", GIT_MODEL_PATH, GIT_OPPONENT_POOL_DIR], repo_path): return
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True)
    if not status_result.stdout.strip():
        print("No changes to commit.")
        return
    if not run_git_command(["git", "commit", "-m", commit_message], repo_path): return
    if not run_git_command(["git", "push", auth_repo_url, f"HEAD:{GIT_BRANCH}"], repo_path): return
    print("--- Push successful ---")

def git_pull(repo_path, auth_repo_url):
    print("\n--- Pulling latest model from GitHub ---")
    if not run_git_command(["git", "pull", auth_repo_url, GIT_BRANCH], repo_path):
        print("Git pull failed. Continuing with local version if available.")

class InferenceWorker(mp.Process):
    def __init__(self, name, task_queue, result_dict, log_queue, stop_event):
        super().__init__(name=name)
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.latest_model = None
        self.opponent_model = None
        self.device = None
        self.opponent_pool_files = []
        self.model_version = -1
        self.last_version_check_time = 0
        self.request_counter = 0

    def _log(self, message):
        self.log_queue.put(f"[{self.name}] {message}")

    def _initialize(self):
        self._log("Started.")
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        self.latest_model = OFC_CNN_Network().to(self.device)
        self.opponent_model = OFC_CNN_Network().to(self.device)
        self._load_models()
        self.latest_model.eval()
        self.opponent_model.eval()

    def _load_models(self):
        try:
            if os.path.exists(MODEL_PATH):
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                self.latest_model.load_state_dict(state_dict.get('model_state_dict', state_dict))
                self.model_version = state_dict.get('model_version', -1)
                self._log(f"Loaded latest model (version {self.model_version}).")
            else:
                self._log("No latest model found, using initialized weights.")
        except Exception as e:
            self._log(f"!!! EXCEPTION during latest model loading: {e}")

        try:
            self.opponent_pool_files = glob.glob(os.path.join(LOCAL_OPPONENT_POOL_DIR, "*.pth"))
            if self.opponent_pool_files:
                opponent_path = random.choice(self.opponent_pool_files)
                state_dict = torch.load(opponent_path, map_location=self.device)
                self.opponent_model.load_state_dict(state_dict.get('model_state_dict', state_dict))
                self._log(f"Loaded opponent model: {os.path.basename(opponent_path)}")
            else:
                self.opponent_model.load_state_dict(self.latest_model.state_dict())
                self._log("Opponent pool is empty, using latest model as opponent.")
        except Exception as e:
            self._log(f"!!! EXCEPTION during opponent model loading: {e}")

    def _check_for_updates(self):
        self.request_counter += 1
        if self.request_counter % 100 != 0 and time.time() - self.last_version_check_time < 5:
            return
        self.last_version_check_time = time.time()
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, 'r') as f:
                    latest_version = int(f.read())
                if latest_version > self.model_version:
                    time.sleep(int(self.name.split('-')[-1]) * 0.1)
                    self._log(f"New model version detected ({latest_version}). Reloading models...")
                    self._load_models()
        except (IOError, ValueError) as e:
            self._log(f"Could not check for model update: {e}")

    def run(self):
        self._initialize()
        
        STREET_START_IDX = 9
        STREET_END_IDX = 14

        while not self.stop_event.is_set():
            try:
                try:
                    request_tuple = self.task_queue.get(timeout=1)
                    req_id, is_policy, infoset, action_vectors, is_traverser_turn = request_tuple
                except queue.Empty:
                    self._check_for_updates()
                    continue

                self._check_for_updates()
                
                with torch.inference_mode():
                    infoset_tensor = torch.tensor([infoset], dtype=torch.float32, device=self.device)
                    infoset_tensor = infoset_tensor.view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS)
                    
                    model_to_use = self.latest_model if is_traverser_turn else self.opponent_model

                    if is_policy:
                        if not action_vectors:
                            result = (req_id, True, [])
                        else:
                            num_actions = len(action_vectors)
                            body_out_single = model_to_use.forward_body(infoset_tensor)
                            body_out_batch = body_out_single.expand(num_actions, -1)
                            actions_tensor = torch.tensor(action_vectors, dtype=torch.float32, device=self.device)
                            street_vector = infoset_tensor[:, STREET_START_IDX:STREET_END_IDX, 0, 0].expand(num_actions, -1)
                            policy_logits = model_to_use.forward_policy_head(body_out_batch, actions_tensor, street_vector)
                            predictions = policy_logits.cpu().numpy().flatten().tolist()
                            result = (req_id, True, predictions)
                    else: # is_value
                        value = model_to_use.forward_value_head(model_to_use.forward_body(infoset_tensor))
                        result = (req_id, False, [value.item()])
                    
                    self.result_dict[req_id] = result

            except (KeyboardInterrupt, SystemExit):
                break
            except Exception:
                self._log(f"---!!! EXCEPTION IN {self.name} !!!---")
                exc_info = traceback.format_exc()
                for line in exc_info.split('\n'):
                    self._log(line)
        
        self._log("Stopped.")

def update_opponent_pool(model_version):
    if not os.path.exists(MODEL_PATH): return
    os.makedirs(LOCAL_OPPONENT_POOL_DIR, exist_ok=True)
    new_opponent_path = os.path.join(LOCAL_OPPONENT_POOL_DIR, f"paqn_model_v{model_version}.pth")
    try:
        shutil.copy2(MODEL_PATH, new_opponent_path)
        print(f"Added model version {model_version} to local opponent pool.")
    except Exception as e:
        print(f"Error updating opponent pool: {e}")
        return
    pool_files = sorted(glob.glob(os.path.join(LOCAL_OPPONENT_POOL_DIR, "*.pth")), key=os.path.getmtime)
    while len(pool_files) > MAX_OPPONENTS_IN_POOL:
        try:
            os.remove(pool_files.pop(0))
            print(f"Removed oldest opponent from pool.")
        except OSError as e:
            print(f"Warning: Could not remove old opponent file: {e}")

# Requirement 2.2: Differentiated learning rate and weight decay
def get_params_for_optimizer(model, base_lr, weight_decay, head_lr_mult=2.0, head_wd=0.0):
    head_names = ["value_head", "action_proj", "street_proj", "policy_head_fc", "body_ln", "action_ln", "street_ln"]
    
    params_body_decay = []
    params_body_no_decay = []
    params_head_decay = []
    params_head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_head = any(h_name in name for h_name in head_names)
        is_no_decay = param.dim() <= 1 or name.endswith(".bias") or "norm" in name

        if is_head:
            if is_no_decay:
                params_head_no_decay.append(param)
            else:
                params_head_decay.append(param)
        else: # is_body
            if is_no_decay:
                params_body_no_decay.append(param)
            else:
                params_body_decay.append(param)
    
    return [
        {'params': params_body_decay, 'weight_decay': weight_decay, 'lr': base_lr},
        {'params': params_body_no_decay, 'weight_decay': 0.0, 'lr': base_lr},
        {'params': params_head_decay, 'weight_decay': head_wd, 'lr': base_lr * head_lr_mult},
        {'params': params_head_no_decay, 'weight_decay': 0.0, 'lr': base_lr * head_lr_mult},
    ]

def main():
    aim_run = aim.Run(experiment="paqn_ofc_poker_fix")
    aim_run["hparams"] = {
        "num_cpp_workers": NUM_CPP_WORKERS,
        "num_inference_workers": NUM_INFERENCE_WORKERS,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
        "batch_size": BATCH_SIZE,
        "action_limit": ACTION_LIMIT,
        "policy_loss_weight": POLICY_LOSS_WEIGHT,
        "adv_clip_value": ADV_CLIP_VALUE,
        "value_clip_value": VALUE_CLIP_VALUE,
        "head_lr_mult": 2.0,
        "head_wd": 0.0,
        "head_warmup_steps": int(os.environ.get("HEAD_WARMUP_STEPS", "2000")),
    }

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
            except (psutil.NoSuchProcess, KeyboardInterrupt):
                break
    threading.Thread(target=monitor_resources, daemon=True).start()

    git_username = os.environ.get('GIT_USERNAME')
    git_token = os.environ.get('GIT_TOKEN')
    if not git_username or not git_token:
        print("ERROR: GIT_USERNAME and GIT_TOKEN environment variables must be set.")
        sys.exit(1)
    auth_repo_url = f"https://{git_username}:{git_token}@github.com/{GIT_REPO_OWNER}/{GIT_REPO_NAME}.git"
    run_git_command(["git", "config", "--global", "user.email", f"{git_username}@users.noreply.github.com"], project_root)
    run_git_command(["git", "config", "--global", "user.name", git_username], project_root)
    git_pull(project_root, auth_repo_url)
    
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    os.makedirs(LOCAL_OPPONENT_POOL_DIR, exist_ok=True)
    if os.path.exists(GIT_OPPONENT_POOL_DIR):
        print("Syncing opponent pool from Git...")
        for f in glob.glob(os.path.join(GIT_OPPONENT_POOL_DIR, "*.pth")):
            shutil.copy2(f, LOCAL_OPPONENT_POOL_DIR)
        print(f"Synced {len(os.listdir(LOCAL_OPPONENT_POOL_DIR))} opponents.")

    if os.path.exists(GIT_MODEL_PATH) and not os.path.exists(MODEL_PATH):
        shutil.copy2(GIT_MODEL_PATH, MODEL_PATH)
    
    print("Initializing C++ hand evaluator lookup tables...", flush=True)
    initialize_evaluator()
    print("C++ evaluator initialized successfully.", flush=True)

    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    
    model = OFC_CNN_Network().to(device)
    
    optimizer_grouped_parameters = get_params_for_optimizer(model, LEARNING_RATE, weight_decay=0.01, head_lr_mult=2.0, head_wd=0.0)
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    model_version, global_step = 0, 0
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            global_step = state_dict.get('global_step', 0)
            model_version = state_dict.get('model_version', 0)
            print(f"Loaded model, optimizer, and state. Resuming from step {global_step}, version {model_version}")
        except Exception as e:
            print(f"Could not load full state, starting from scratch. Error: {e}")
            model = OFC_CNN_Network().to(device)
            optimizer = optim.AdamW(get_params_for_optimizer(model, LEARNING_RATE, weight_decay=0.01))

    head_warmup_steps = int(os.environ.get("HEAD_WARMUP_STEPS", "2000"))
    if head_warmup_steps > 0:
        print(f"!!! HEAD-ONLY WARMUP ENABLED for {head_warmup_steps} steps !!!")
    
    policy_buffer = ReplayBuffer(BUFFER_CAPACITY)
    value_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    manager = mp.Manager()
    request_queue = manager.Queue(maxsize=NUM_CPP_WORKERS * 16)
    result_dict = manager.dict()
    log_queue = manager.Queue()
    stop_event = mp.Event()

    inference_workers = []
    for i in range(NUM_INFERENCE_WORKERS):
        worker = InferenceWorker(f"InferenceWorker-{i}", request_queue, result_dict, log_queue, stop_event)
        worker.start()
        inference_workers.append(worker)

    print(f"Creating C++ SolverManager with {NUM_CPP_WORKERS} workers...", flush=True)
    solver_manager = SolverManager(
        NUM_CPP_WORKERS, ACTION_LIMIT, policy_buffer, value_buffer,
        request_queue, result_dict, log_queue
    )
    
    solver_manager.start()
    print("C++ workers are running in the background.", flush=True)
    
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    skipped_policy_updates = deque(maxlen=100)
    last_stats_time = time.time()
    last_cleanup_time = time.time()
    
    STREET_START_IDX = 9
    STREET_END_IDX = 14
    training_started = False
    
    min_fill = BATCH_SIZE * 4 if global_step > 0 else MIN_BUFFER_FILL_SAMPLES
    print(f"Training will start when buffer size reaches {min_fill} samples.")
    
    last_save_step = global_step
    last_push_step = global_step
    
    result_timestamps = {}

    try:
        while True:
            if time.time() - last_stats_time > STATS_INTERVAL_SECONDS:
                while not log_queue.empty():
                    try:
                        print(log_queue.get(timeout=0.01), flush=True)
                    except queue.Empty:
                        break
                
                total_generated = policy_buffer.total_generated()
                avg_p_loss = np.mean(policy_losses) if policy_losses else float('nan')
                avg_v_loss = np.mean(value_losses) if value_losses else float('nan')
                
                print("\n" + "="*20 + " STATS UPDATE " + "="*20, flush=True)
                print(f"Time: {time.strftime('%H:%M:%S')}", flush=True)
                print(f"Model Version: {model_version}", flush=True)
                print(f"Global Step: {global_step}", flush=True)
                print(f"Total Generated: {total_generated:,}", flush=True)
                print(f"Buffer Fill -> Policy: {policy_buffer.size():,}/{BUFFER_CAPACITY:,} ({policy_buffer.size()/BUFFER_CAPACITY:.1%}) "
                      f"| Value: {value_buffer.size():,}/{BUFFER_CAPACITY:,} ({value_buffer.size()/BUFFER_CAPACITY:.1%})", flush=True)
                print(f"Avg Losses (last 100) -> Policy: {avg_p_loss:.6f} | Value: {avg_v_loss:.6f}", flush=True)
                print(f"Request Queue: {request_queue.qsize()} | Result Dict: {len(result_dict)}", flush=True)
                print("="*54, flush=True)
                
                if aim_run.active:
                    aim_run.track(total_generated, name="system/total_samples_generated", step=global_step)
                    aim_run.track(policy_buffer.size(), name="buffer/policy_buffer_size", step=global_step)
                    aim_run.track(value_buffer.size(), name="buffer/value_buffer_size", step=global_step)
                    aim_run.track(request_queue.qsize(), name="system/request_queue_size", step=global_step)
                    if policy_losses: aim_run.track(avg_p_loss, name="loss/policy_loss_avg", step=global_step)
                    if value_losses: aim_run.track(avg_v_loss, name="loss/value_loss_avg", step=global_step)
                
                last_stats_time = time.time()

            if time.time() - last_cleanup_time > 60:
                now = time.time()
                for key in result_dict.keys():
                    if key not in result_timestamps:
                        result_timestamps[key] = now
                
                keys_to_delete = [key for key, ts in result_timestamps.items() if now - ts > RESULT_TTL_SECONDS]
                
                if len(result_dict) > 10000 and keys_to_delete:
                    print(f"[CLEANUP] Deleting {len(keys_to_delete)} expired results from result_dict (current size: {len(result_dict)})...", flush=True)
                    for key in keys_to_delete:
                        result_dict.pop(key, None)
                        result_timestamps.pop(key, None)
                
                if len(result_timestamps) > len(result_dict) * 2:
                     current_keys = set(result_dict.keys())
                     ts_keys_to_delete = [key for key in result_timestamps if key not in current_keys]
                     for key in ts_keys_to_delete:
                         result_timestamps.pop(key, None)

                gc.collect()
                last_cleanup_time = time.time()

            if value_buffer.size() < min_fill or policy_buffer.size() < min_fill:
                print(f"Waiting for buffer... P: {policy_buffer.size()}/{min_fill} | V: {value_buffer.size()}/{min_fill}", end='\r', flush=True)
                time.sleep(1)
                continue

            if not training_started:
                print("\nBuffer ready. Starting training...")
                training_started = True

            model.train()
            
            # Requirement 2.2: Head warmup logic
            if head_warmup_steps > 0 and global_step < head_warmup_steps:
                head_names = ["value_head", "action_proj", "street_proj", "policy_head_fc", "body_ln", "action_ln", "street_ln"]
                for name, param in model.named_parameters():
                    if not any(h in name for h in head_names):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = True

            v_batch = value_buffer.sample(BATCH_SIZE)
            if not v_batch: continue
            v_infosets_np, _, v_targets_np = v_batch
            
            p_batch = policy_buffer.sample(BATCH_SIZE)
            if not p_batch: continue
            p_infosets_np, p_actions_np, p_advantages_np = p_batch
            
            # --- VALUE LOSS (Requirement 2.1) ---
            v_infosets = torch.from_numpy(v_infosets_np).view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS).to(device)
            v_targets = torch.from_numpy(v_targets_np).unsqueeze(1).to(device)
            
            # Train on raw targets, clipped for stability
            v_targets_clipped = torch.clamp(v_targets, -VALUE_CLIP_VALUE, VALUE_CLIP_VALUE)
            pred_values = model.forward_value_head(model.forward_body(v_infosets))
            loss_v = F.huber_loss(pred_values, v_targets_clipped, delta=1.0)
            
            # --- POLICY LOSS (Requirement 2.1) ---
            p_infosets = torch.from_numpy(p_infosets_np).view(-1, NUM_FEATURE_CHANNELS, NUM_SUITS, NUM_RANKS).to(device)
            p_actions = torch.from_numpy(p_actions_np).to(device)
            p_advantages = torch.from_numpy(p_advantages_np).unsqueeze(1).to(device)
            
            adv_mean, adv_std = p_advantages.mean(), p_advantages.std()
            
            # Requirement 2.3: Skip flat policy batches
            update_skipped = 0
            if adv_std.item() < 1e-6:
                loss_p = torch.tensor(0.0, device=device, requires_grad=False)
                update_skipped = 1
            else:
                # Z-score normalize advantages per batch
                p_advantages_normalized = torch.clamp((p_advantages - adv_mean) / adv_std, -ADV_CLIP_VALUE, ADV_CLIP_VALUE)
                p_street_vector = p_infosets[:, STREET_START_IDX:STREET_END_IDX, 0, 0]
                body_out = model.forward_body(p_infosets)
                pred_logits = model.forward_policy_head(body_out, p_actions, p_street_vector)
                loss_p = F.huber_loss(pred_logits, p_advantages_normalized, delta=1.0)

            optimizer.zero_grad()
            total_loss = loss_v + POLICY_LOSS_WEIGHT * loss_p
            total_loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            value_losses.append(loss_v.item())
            policy_losses.append(loss_p.item() if update_skipped == 0 else 0)
            skipped_policy_updates.append(update_skipped)
            global_step += 1
            
            # Requirement 3.1: Expanded Diagnostics
            if aim_run.active:
                aim_run.track(loss_v.item(), name="loss/value_loss", step=global_step)
                aim_run.track(loss_p.item(), name="loss/policy_loss", step=global_step)
                aim_run.track(grad_norm.item(), name="diagnostics/grad_norm", step=global_step)
                
                with torch.no_grad():
                    # Raw target stats
                    aim_run.track(float(v_targets.std().item()), name="targets/value_raw/std", step=global_step)
                    aim_run.track(float(p_advantages.std().item()), name="targets/advantage_raw/std", step=global_step)
                    
                    # Explained Variance
                    var_t = float(torch.var(v_targets_clipped))
                    ev = 1.0 - float(torch.var(v_targets_clipped - pred_values)) / max(var_t, 1e-6)
                    aim_run.track(ev, name="diagnostics/value_explained_var", step=global_step)
                    aim_run.track(var_t, name="diagnostics/value_target_var", step=global_step)

                    # Policy Correlation
                    if update_skipped == 0:
                        y = p_advantages_normalized.cpu().numpy().flatten()
                        yhat = pred_logits.cpu().numpy().flatten()
                        if y.std() > 1e-6 and yhat.std() > 1e-6:
                            corr = float(np.corrcoef(y, yhat)[0, 1])
                            aim_run.track(corr, name="diagnostics/policy_corr", step=global_step)
                    
                    # Prediction stats
                    aim_run.track(float(pred_values.std().item()), name="diagnostics/value_pred_std", step=global_step)
                    if update_skipped == 0:
                        aim_run.track(float(pred_logits.std().item()), name="diagnostics/policy_logit_std", step=global_step)

                # Skipped updates rate
                skip_rate = np.mean(skipped_policy_updates) if skipped_policy_updates else 0
                aim_run.track(skip_rate, name="diagnostics/skipped_policy_updates_rate", step=global_step)

            is_first_save = (global_step >= FIRST_SAVE_STEP) and (last_save_step < FIRST_SAVE_STEP)
            is_regular_save = (global_step - last_save_step) >= SAVE_INTERVAL_STEPS

            if training_started and (is_first_save or is_regular_save):
                print(f"\n--- Saving model at step {global_step} ---", flush=True)
                model_version += 1
                
                torch.save({
                    'global_step': global_step, 'model_version': model_version,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_PATH + ".tmp")
                os.rename(MODEL_PATH + ".tmp", MODEL_PATH)

                with open(VERSION_FILE, 'w') as f: f.write(str(model_version))
                update_opponent_pool(model_version)
                last_save_step = global_step
            
            if training_started and (global_step - last_push_step) >= GIT_PUSH_INTERVAL_STEPS:
                git_push(f"Periodic save: v{model_version}, step {global_step}", project_root, auth_repo_url)
                last_push_step = global_step

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("\n" + "="*15 + " НАЧАТА ПРОЦЕДУРА ЗАВЕРШЕНИЯ " + "="*15, flush=True)
        
        if 'aim_run' in locals() and aim_run.active:
            print("1. Закрытие сессии Aim для сохранения всех метрик...", flush=True)
            aim_run.close()
            print("   ✅ Сессия Aim успешно закрыта.", flush=True)

        print("2. Отправка сигнала остановки всем воркерам...", flush=True)
        stop_event.set()
        
        print("3. Остановка C++ воркеров...", flush=True)
        if 'solver_manager' in locals():
            solver_manager.stop()
        print("   ✅ C++ воркеры остановлены.", flush=True)
        
        print("4. Остановка Python воркеров...", flush=True)
        if 'inference_workers' in locals():
            for worker in inference_workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    print(f"   - Принудительное завершение {worker.name}...", flush=True)
                    worker.terminate()
        print("   ✅ Python воркеры остановлены.", flush=True)
        
        if training_started:
            print("5. Финальное сохранение и пуш модели...", flush=True)
            try:
                torch.save({
                    'global_step': global_step,
                    'model_version': model_version,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_PATH)
                print("   ✅ Модель сохранена локально.", flush=True)
                git_push(f"Final save on exit: v{model_version}, step {global_step}", project_root, auth_repo_url)
            except Exception as e:
                print(f"   ---! ❌ ОШИБКА при финальном сохранении/пуше: {e}", flush=True)
        
        print("="*58)
        print("✅ Процесс обучения корректно завершен.")

if __name__ == "__main__":
    main()
