# GPU Acceleration Guide

## Overview

MCP Context Server uses [Ollama](https://ollama.com/) for local embedding generation and summary inference. Ollama supports GPU acceleration on NVIDIA, AMD, and Intel hardware. Since MCP Context Server communicates with Ollama via HTTP, GPU configuration is purely an Ollama infrastructure concern -- no application code changes are needed.

GPU acceleration primarily benefits larger models. For the default 0.6B parameter models (`qwen3-embedding:0.6b`, `qwen3:0.6b`), CPU-only performance is often adequate.

## GPU Support Matrix

| GPU Vendor     | Docker (Linux host) | Docker Desktop (Windows) | Kubernetes (Helm) | Native (Host) | Status       |
|----------------|---------------------|--------------------------|-------------------|---------------|--------------|
| NVIDIA (CUDA)  | Supported           | Supported (GPU-PV)       | Supported         | Supported     | Stable       |
| AMD (ROCm)     | Supported           | Not supported            | Supported         | Supported     | Stable       |
| Intel (Vulkan) | Experimental        | Not supported            | Not recommended   | Experimental  | Experimental |

## Windows Docker Desktop Limitations

Docker Desktop on Windows uses the WSL2 backend for GPU access. GPU passthrough in this environment is **NVIDIA-only** via the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and GPU Paravirtualization (GPU-PV).

**Key limitations:**

- **Intel and AMD GPUs are NOT supported** in Docker Desktop on Windows. The AMD/ROCm and Intel/Vulkan GPU sections in Docker Compose files require `/dev/dri` device nodes, which are Linux-native DRM (Direct Rendering Manager) devices. WSL2 exposes GPU via `/dev/dxg` (DirectX), not `/dev/dri`.
- **Intel/Vulkan acceleration requires native Ollama on Windows.** See [Hybrid Deployment Pattern (Windows)](#hybrid-deployment-pattern-windows) below for running Ollama natively while keeping the MCP Context Server in Docker.
- GPU acceleration in Docker containers **primarily benefits models larger than approximately 3B parameters**. For the default 0.6B models, CPU-only performance is adequate.

If you need Intel or AMD GPU acceleration with Docker, use a native Linux host where `/dev/dri` is available.

## Docker Compose

All Ollama-based Docker Compose files include commented GPU configuration sections for each vendor. Uncomment the section that matches your GPU.

All nine Ollama-based Compose files follow the same pattern:

- `docker-compose.sqlite.ollama.yml`
- `docker-compose.sqlite.ollama.local.yml`
- `docker-compose.sqlite.ollama-openai.yml`
- `docker-compose.postgresql.ollama.yml`
- `docker-compose.postgresql.ollama.local.yml`
- `docker-compose.postgresql.ollama-openai.yml`
- `docker-compose.postgresql-external.ollama.yml`
- `docker-compose.postgresql-external.ollama.local.yml`
- `docker-compose.postgresql-external.ollama-openai.yml`

### NVIDIA GPU

**Prerequisites:**
- NVIDIA GPU driver installed on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

**Steps:**

Uncomment the NVIDIA GPU section in the `ollama` service of your Docker Compose file:

```yaml
# --- NVIDIA GPU (requires NVIDIA Container Toolkit) ---
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Verify:**

```bash
docker compose -f deploy/docker/<your-compose-file> exec ollama nvidia-smi
```

### AMD GPU (ROCm)

**Prerequisites:**
- AMD GPU with ROCm support
- [ROCm driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) installed on the host

**Steps:**

1. Uncomment `OLLAMA_TAG: rocm` in the `build.args` section to use the ROCm-based Ollama image:

```yaml
build:
  context: ../..
  dockerfile: deploy/docker/ollama/Dockerfile
  args:
    OLLAMA_TAG: rocm
```

2. Uncomment the AMD GPU section:

```yaml
# --- AMD GPU (requires ROCm driver; also uncomment OLLAMA_TAG above) ---
devices:
  - /dev/kfd:/dev/kfd
  - /dev/dri:/dev/dri
group_add:
  - video
  - render
```

3. Rebuild the Ollama image:

```bash
docker compose -f deploy/docker/<your-compose-file> build ollama
docker compose -f deploy/docker/<your-compose-file> up -d
```

Alternatively, build with the `--build-arg` flag without editing the Compose file:

```bash
docker compose -f deploy/docker/<your-compose-file> build --build-arg OLLAMA_TAG=rocm ollama
```

**Verify:**

```bash
docker compose -f deploy/docker/<your-compose-file> exec ollama rocm-smi
```

### Intel/Vulkan GPU (Experimental)

> **Warning:** Intel Vulkan GPU support in Ollama is experimental. Known issues include gibberish or degraded output on Intel integrated GPUs, particularly on Alder Lake, Raptor Lake, Arrow Lake, and Meteor Lake architectures. **Test with Ollama outside Docker first** before deploying in containers. The AMD and Intel/Vulkan GPU sections require `/dev/dri` device nodes and are **Linux host only** -- they do not work on Docker Desktop for Windows (WSL2). See [Windows Docker Desktop Limitations](#windows-docker-desktop-limitations).

**Known issues:**
- Gibberish/degraded output on Intel iGPUs ([ollama#13086](https://github.com/ollama/ollama/issues/13086), [ollama#13108](https://github.com/ollama/ollama/issues/13108))
- Model quality degradation with larger models ([ollama#13964](https://github.com/ollama/ollama/issues/13964))
- Vulkan cannot be reliably disabled once enabled ([ollama#13212](https://github.com/ollama/ollama/issues/13212))
- Integer dot product issues on Intel GPUs ([llama.cpp#17106](https://github.com/ggml-org/llama.cpp/issues/17106))
- Integrated GPUs (Intel and AMD) with 0 bytes dedicated VRAM fail Vulkan device enumeration ([ollama#13023](https://github.com/ollama/ollama/issues/13023), [ollama#13103](https://github.com/ollama/ollama/issues/13103))
- AMD iGPUs (Radeon 760M, 860M) with shared VRAM also fail Vulkan detection ([ollama#14527](https://github.com/ollama/ollama/issues/14527), [ollama#14562](https://github.com/ollama/ollama/issues/14562))

**Steps:**

1. Uncomment `OLLAMA_VULKAN=1` in the `environment` section of the `ollama` service:

```yaml
environment:
  - OLLAMA_HOST=0.0.0.0
  - OLLAMA_KEEP_ALIVE=-1
  - EMBEDDING_MODEL=${EMBEDDING_MODEL:-qwen3-embedding:0.6b}
  - SUMMARY_MODEL=${SUMMARY_MODEL:-qwen3:0.6b}
  - OLLAMA_VULKAN=1
```

2. Uncomment the Intel/Vulkan GPU section:

```yaml
# --- Intel/Vulkan GPU (EXPERIMENTAL - see docs for known issues) ---
devices:
  - /dev/dri:/dev/dri
group_add:
  - video
  - render
```

3. If output is gibberish, also uncomment the workaround environment variable:

```yaml
  - GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
```

**Additional workarounds:**

| Environment Variable                    | Effect                                           |
|-----------------------------------------|--------------------------------------------------|
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | Disables integer dot product optimization        |
| `GGML_VK_VISIBLE_DEVICES=-1`            | Disables all Vulkan GPUs (fallback to CPU)       |
| `OLLAMA_NUM_GPU=0`                      | Forces CPU-only inference                        |
| `OLLAMA_FLASH_ATTENTION=0`              | Disables flash attention (may help small models) |

## Kubernetes (Helm)

### NVIDIA GPU

Requires the [NVIDIA device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin).

In `values.yaml` or via `--set`:

```yaml
ollama:
  enabled: true
  resources:
    limits:
      nvidia.com/gpu: "1"
```

### AMD GPU

Requires the [AMD GPU device plugin for Kubernetes](https://github.com/ROCm/k8s-device-plugin).

In `values.yaml` or via `--set`:

```yaml
ollama:
  enabled: true
  image:
    tag: "rocm"
  resources:
    limits:
      amd.com/gpu: "1"
```

### Intel/Vulkan GPU

Intel/Vulkan GPUs are **not recommended** for Kubernetes deployments due to the experimental Vulkan quality issues described above and the lack of a widely adopted device plugin. Use CPU-only or consider [IPEX-LLM](https://github.com/intel/ipex-llm) as an alternative.

## Native (Non-Docker) Setup

### NVIDIA

CUDA is automatically detected by Ollama when NVIDIA drivers are installed. No environment variables needed.

```bash
# Verify GPU detection
ollama run qwen3:0.6b "test"
# Check GPU utilization
nvidia-smi
```

### AMD (ROCm)

Install [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) and use the ROCm-compatible Ollama build.

```bash
# Verify ROCm installation
rocm-smi
# Verify device nodes
ls -la /dev/kfd /dev/dri/renderD*
```

### Intel/Vulkan

Set `OLLAMA_VULKAN=1` before starting Ollama. The same quality warnings apply.

```bash
OLLAMA_VULKAN=1 ollama serve
```

### Hybrid Deployment Pattern (Windows)

On Windows, run Ollama natively to access GPU acceleration while keeping MCP Context Server in Docker:

1. **Install Ollama on Windows** from [ollama.com](https://ollama.com/download)

2. **Start Ollama with Vulkan** (if Intel/AMD GPU):

```powershell
$env:OLLAMA_VULKAN = "1"
$env:OLLAMA_HOST = "0.0.0.0"
ollama serve
```

3. **Update Docker Compose** -- point to host Ollama instead of sidecar:

```yaml
services:
  mcp-context-server:
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
    # Remove or comment out: depends_on: ollama

  # Comment out or remove the ollama service entirely
  # ollama:
  #   ...
```

4. **Pull required models** on the host:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:0.6b
```

> **Note:** `host.docker.internal` resolves to the host machine from within Docker Desktop containers on Windows and macOS.

## Intel iGPU Considerations

Integrated GPUs share system RAM and have no dedicated VRAM. Key considerations:

- For small models (0.6B parameters), CPU-only may perform comparably to iGPU inference
- Vulkan quality risks make CPU-only the safer choice for production workloads
- [IPEX-LLM](https://github.com/intel/ipex-llm) is a third-party Intel-specific fork with potentially better Intel GPU support (not integrated in this project)

## Troubleshooting

### SELinux Blocking GPU Access (RHEL/Fedora/CentOS)

```bash
sudo setsebool container_use_devices=1
```

### Device Permission Errors

Ensure the user is in the `video` and `render` groups:

```bash
sudo usermod -aG video,render $USER
# Log out and back in for group changes to take effect
```

### NVIDIA Container Toolkit Not Installed

Follow the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Verify installation:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### AMD ROCm Not Detected

```bash
# Verify ROCm installation
rocm-smi

# Check device nodes exist
ls -la /dev/kfd /dev/dri/renderD*

# If missing, install ROCm drivers
# See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
```

### Vulkan Producing Gibberish Output

1. Try adding `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` to the environment
2. Try adding `OLLAMA_FLASH_ATTENTION=0` (may help with very small models)
3. Fall back to CPU: set `OLLAMA_NUM_GPU=0`
4. Test outside Docker first: `OLLAMA_VULKAN=1 ollama serve`

### GPU Not Detected in Container

```bash
# NVIDIA: check runtime
docker info | grep -i nvidia

# AMD: check device nodes are passed through
docker compose exec ollama ls -la /dev/kfd /dev/dri/

# General: check Ollama logs for GPU detection
docker compose -f deploy/docker/<your-compose-file> logs ollama | grep -i gpu
```

### GPU Discovery Diagnostics

To diagnose GPU detection issues, run Ollama with debug logging:

```bash
# Linux/macOS
OLLAMA_VULKAN=1 OLLAMA_DEBUG=1 ollama serve

# Windows (PowerShell)
$env:OLLAMA_VULKAN = "1"
$env:OLLAMA_DEBUG = "1"
ollama serve
```

**What to look for in logs:**

- `runner enumerated devices` -- GPU discovery completed
- `total_vram=0 B` -- No GPU VRAM detected (CPU fallback)
- `devices=[]` -- No Vulkan-compatible devices found
- `offloaded N/M layers` -- N of M model layers are on GPU

**Common diagnostic outcomes:**

- `devices=[]` + `total_vram=0 B`: Vulkan device enumeration failed. Common with integrated GPUs that report 0 bytes dedicated VRAM.
- `offloaded 0/N layers`: GPU detected but model not offloaded. Try increasing available VRAM or use a smaller model.
- No GPU-related log lines: Ollama did not attempt GPU discovery. Verify `OLLAMA_VULKAN=1` is set.

For deeper investigation, use `OLLAMA_DEBUG=2` for trace-level logging.

### Intel/AMD iGPU Not Detected

Integrated GPUs sharing system RAM (no dedicated VRAM) commonly fail Ollama's Vulkan device enumeration. This affects:

- Intel UHD (Alder Lake, Raptor Lake, Arrow Lake, Meteor Lake)
- AMD Radeon integrated (760M, 860M, and similar RDNA 3 iGPUs)

**Expected behavior:** `devices=[]`, `total_vram=0 B`, CPU-only fallback. This is a known upstream Ollama limitation, not a configuration error.

**Workaround:** For 0.6B parameter models, CPU-only inference is adequate. For larger models requiring GPU acceleration, use a discrete NVIDIA GPU.

## Performance Expectations

| Configuration       | Inference Speed | Recommendation                 |
|---------------------|-----------------|--------------------------------|
| NVIDIA GPU (CUDA)   | Fastest         | Recommended for production     |
| AMD GPU (ROCm)      | Fast            | Good alternative               |
| Intel iGPU (Vulkan) | Variable        | Not recommended for production |
| CPU-only            | Slower          | Safe, reliable default         |

For 0.6B parameter models (`qwen3-embedding:0.6b`, `qwen3:0.6b`), CPU-only performance is reasonable. GPU acceleration provides the most benefit for larger models.

## Related Documentation

- [Docker Deployment Guide](docker.md) - Docker Compose deployment with GPU sections
- [Helm Deployment Guide](helm.md) - Kubernetes deployment with GPU resources
- [Ollama GPU Documentation](https://docs.ollama.com/gpu) - Upstream Ollama GPU reference
