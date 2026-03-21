# GPU Acceleration Guide

## Overview

MCP Context Server uses [Ollama](https://ollama.com/) for local embedding generation and summary inference. Ollama supports GPU acceleration on NVIDIA, AMD, and Intel hardware. Since MCP Context Server communicates with Ollama via HTTP, GPU configuration is purely an Ollama infrastructure concern -- no application code changes are needed.

GPU acceleration primarily benefits larger models. For the default 0.6B parameter models (`qwen3-embedding:0.6b`, `qwen3:0.6b`), CPU-only performance is often adequate.

## GPU Support Matrix

| GPU Vendor     | Docker Compose | Kubernetes (Helm) | Native (Host) | Status       |
|----------------|----------------|-------------------|---------------|--------------|
| NVIDIA (CUDA)  | Supported      | Supported         | Supported     | Stable       |
| AMD (ROCm)     | Supported      | Supported         | Supported     | Stable       |
| Intel (Vulkan) | Supported      | Not recommended   | Supported     | Experimental |

## Docker Compose

All Ollama-based Docker Compose files include commented GPU configuration sections for each vendor. Uncomment the section that matches your GPU.

The six Ollama Compose files follow the same pattern:

- `docker-compose.sqlite.ollama.yml`
- `docker-compose.sqlite.ollama.dev.yml`
- `docker-compose.postgresql.ollama.yml`
- `docker-compose.postgresql.ollama.dev.yml`
- `docker-compose.postgresql-external.ollama.yml`
- `docker-compose.postgresql-external.ollama.dev.yml`

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

> **Warning:** Intel Vulkan GPU support in Ollama is experimental. Known issues include gibberish or degraded output on Intel integrated GPUs, particularly on Alder Lake, Arrow Lake, and Meteor Lake architectures. **Test with Ollama outside Docker first** before deploying in containers.

**Known issues:**
- Gibberish/degraded output on Intel iGPUs ([ollama#13086](https://github.com/ollama/ollama/issues/13086), [ollama#13108](https://github.com/ollama/ollama/issues/13108))
- Model quality degradation with larger models ([ollama#13964](https://github.com/ollama/ollama/issues/13964))
- Vulkan cannot be reliably disabled once enabled ([ollama#13212](https://github.com/ollama/ollama/issues/13212))
- Integer dot product issues on Intel GPUs ([llama.cpp#17106](https://github.com/ggml-org/llama.cpp/issues/17106))

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
