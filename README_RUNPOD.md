# Qwen-Image-Edit RunPod Serverless Deployment

This repository is optimized for deployment on **RunPod Serverless** using **RTX 4090 (24GB VRAM)**. It includes support for Cloudflare R2 image uploading and dynamic LoRA loading.

## Features

- **RunPod Serverless**: Scalable API for image editing.
- **RTX 4090 Optimized**: Uses `bfloat16`, CPU offloading, and Flash Attention 3 to fit within 24GB VRAM.
- **R2 Upload**: Automatically uploads generated images to Cloudflare R2.
- **Base64 Fallback**: Returns base64 encoded images if R2 upload fails.
- **Dynamic LoRA**: Support for built-in adapters and custom Hugging Face LoRA repositories.

## Deployment Instructions

### 1. Prerequisites

- A RunPod account.
- A Cloudflare account with R2 bucket created.
- A Hugging Face account (and token for private/gated models).

### 2. Build and Push Docker Image

```bash
# Build the image
docker build -t your-docker-user/qwen-image-edit-serverless:latest .

# Push to Docker Hub
docker push your-docker-user/qwen-image-edit-serverless:latest
```

### 3. Setup RunPod Serverless Endpoint

1. Go to **Serverless** -> **Endpoints** on RunPod.
2. Click **New Endpoint**.
3. Use your Docker image: `your-docker-user/qwen-image-edit-serverless:latest`.
4. Select **RTX 4090** or **RTX 5090** as the GPU.
5. Set the following **Environment Variables**:

| Variable | Description |
| --- | --- |
| `R2_ACCOUNT_ID` | Your Cloudflare Account ID |
| `R2_ACCESS_KEY_ID` | R2 Access Key ID |
| `R2_SECRET_ACCESS_KEY` | R2 Secret Access Key |
| `R2_BUCKET_NAME` | Your R2 Bucket Name |
| `R2_PUBLIC_URL` | Public URL for your R2 bucket (e.g., https://pub-xxx.r2.dev) |
| `HF_TOKEN` | (Optional) Hugging Face token for private/gated models |
| `HF_HOME` | Set to `/runpod-volume/huggingface` if using Network Volume |

### 4. API Usage

#### Request Format

```json
{
  "input": {
    "images": [
      {
        "image_url": "https://example.com/input.jpg" 
      }
    ],
    "prompt": "transform into anime",
    "lora_adapter": "Photo-to-Anime",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 42
  }
}
```

#### LoRA Options

- **Built-in**: Use `lora_adapter` with one of the predefined names (e.g., `Photo-to-Anime`, `Anime-V2`, `Upscaler`, etc.).
- **Custom**: Use `lora_repo` (Hugging Face repo ID) and optionally `lora_weight_name`.

```json
{
  "input": {
    "images": [{"image_url": "..."}],
    "prompt": "...",
    "lora_repo": "user/custom-lora-repo",
    "lora_weight_name": "adapter_model.safetensors",
    "lora_scale": 0.8
  }
}
```

#### Response Format

```json
{
  "images": [
    {
      "image_url": "https://your-r2-public-url/videos/images/job-id/scene-0.png",
      "seed": 42
    }
  ]
}
```

## Local Testing

You can test the handler locally using `test_handler.py`. Make sure to set the environment variables or mock them in the script.

```bash
python3 test_handler.py
```

## Memory Management

- **CPU Offloading**: Enabled for the text encoder to save VRAM.
- **LoRA Unloading**: Automatically unloads the previous LoRA when a new one is requested.
- **Cache Clearing**: `torch.cuda.empty_cache()` is called after every inference.
