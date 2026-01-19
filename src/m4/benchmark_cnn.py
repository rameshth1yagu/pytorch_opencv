import torch
import torchvision.models as models
import time
import csv
import platform
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Silences "UserWarning"

# --- CONFIGURATION ---
MODEL_NAME = "resnet18"  # Using a standard model for fair comparison
BATCH_SIZE = 64
NUM_ITERATIONS = 200
WARMUP = 20

# --- 1. DETECT HARDWARE ---
system_info = f"{platform.system()} {platform.processor()}"
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ DETECTED: NVIDIA GPU (CUDA) - Likely Jetson/Linux")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ DETECTED: Apple Silicon GPU (MPS) - Mac")
else:
    print(f"⚠️ DETECTED: CPU Only (This will be slow)")

# --- 2. LOAD MODEL ---
print(f"Loading {MODEL_NAME} to {device}...")
model = models.resnet18(weights=None).to(device)
model.eval()

# Create dummy input (Standard ImageNet size: 1x3x224x224)
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

# --- 3. WARMUP (Critical for GPU) ---
# GPUs need to "compile" kernels on the first few runs. We ignore these.
print("Warming up...")
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(dummy_input)

# --- 4. THE BENCHMARK ---
print(f"Running {NUM_ITERATIONS} iterations...")
latencies = []

with torch.no_grad():
    for i in range(NUM_ITERATIONS):
        # Sync GPU before start (for accurate timing)
        if device.type == 'cuda': torch.cuda.synchronize()
        if device.type == 'mps': torch.mps.synchronize()

        start_time = time.perf_counter()

        _ = model(dummy_input)

        # Sync GPU after end
        if device.type == 'cuda': torch.cuda.synchronize()
        if device.type == 'mps': torch.mps.synchronize()

        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to ms

# --- 5. CALCULATE STATS ---
avg_latency = sum(latencies) / len(latencies)
fps = 1000 / avg_latency

print(f"\nRESULTS for {device}:")
print(f"------------------------------------------------")
print(f"Avg Latency:   {avg_latency:.2f} ms")
print(f"FPS:           {fps:.2f} frames/sec")
print(f"------------------------------------------------")

# --- 6. SAVE REPORT ---
csv_file = "benchmark_report.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Device", "OS", "Model", "FPS", "Latency(ms)"]) # Header

    writer.writerow([device.type.upper(), system_info, MODEL_NAME, f"{fps:.2f}", f"{avg_latency:.2f}"])

print(f"📝 Stats saved to {csv_file}")