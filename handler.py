import runpod
import torch
import base64
import os
import logging
import sys
import gc
import random
import requests
from io import BytesIO
from PIL import Image
import boto3
from botocore.config import Config

# Import custom pipeline and transformer
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

# ==============================
# CONFIG & LOGGING
# ==============================
HF_CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ==============================
# GLOBAL STATE
# ==============================
pipe = None
current_lora_key = None  # format: "adapter:name" or "repo:repo_id"

ADAPTER_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles"
    },
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "photo-to-anime"
    },
    "Anime-V2": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
        "adapter_name": "anime-v2"
    },
    "Light-Migration": {
        "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
        "weights": "参考色调.safetensors",
        "adapter_name": "light-migration"
    },
    "Upscaler": {
        "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weights": "qwen_image_edit_2511_upscale.safetensors",
        "adapter_name": "upscale-2k"
    },
    "Style-Transfer": {
        "repo": "zooeyy/Style-Transfer",
        "weights": "Style Transfer-Alpha-V0.1.safetensors",
        "adapter_name": "style-transfer"
    },
    "Manga-Tone": {
        "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        "weights": "tone001.safetensors",
        "adapter_name": "manga-tone"
    },
    "Anything2Real": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601.safetensors",
        "adapter_name": "anything2real"
    },
    "Fal-Multiple-Angles": {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "adapter_name": "fal-multiple-angles"
    },
    "Polaroid-Photo": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        "adapter_name": "polaroid-photo"
    },
    "Unblur-Anything": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        "adapter_name": "unblur-anything"
    },
    "Midnight-Noir-Eyes-Spotlight": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        "adapter_name": "midnight-noir-eyes-spotlight"
    },
    "Hyper-Realistic-Portrait": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
       "weights": "HRP_20.safetensors",
       "adapter_name": "hyper-realistic-portrait"
   },     
    "Ultra-Realistic-Portrait": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
       "weights": "URP_20.safetensors",
       "adapter_name": "ultra-realistic-portrait"
   },     
    "Pixar-Inspired-3D": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
       "weights": "PI3_20.safetensors",
       "adapter_name": "pi3"
   },
    "Noir-Comic-Book": {
       "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
       "weights": "Noir-Comic-Book-Panel_20.safetensors",
       "adapter_name": "ncb"
   },  
    "Any-light": {
       "repo": "lilylilith/QIE-2511-MP-AnyLight",
       "weights": "QIE-2511-AnyLight_.safetensors",
       "adapter_name": "any-light"
   }, 
    "Studio-DeLight": {
       "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
       "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
       "adapter_name": "studio-delight"
   }, 
    "Cinematic-FlatLog": {
       "repo": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
       "weights": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
       "adapter_name": "flat-log"
   },   
}

# ==============================
# MODEL LOADER
# ==============================
def get_pipe():
    global pipe
    if pipe is None:
        logger.info("🚀 Loading Qwen-Image-Edit pipeline...")
        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                transformer=QwenImageTransformer2DModel.from_pretrained(
                    "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
                    torch_dtype=dtype,
                    device_map='cuda',
                    cache_dir=HF_CACHE_DIR
                ),
                torch_dtype=dtype,
                cache_dir=HF_CACHE_DIR
            ).to(device)
            
            # Enable memory optimizations
            pipe.enable_model_cpu_offload()
            
            # Try Flash Attention 3
            try:
                pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
                logger.info("✅ Flash Attention 3 Processor set successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not set FA3 processor: {e}")
            
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}", exc_info=True)
            raise e
    return pipe

# ==============================
# R2 CONFIG
# ==============================
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL")

r2_client = None
if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    r2_endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    r2_client = boto3.client(
        "s3",
        endpoint_url=r2_endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )
    logger.info(f"✅ R2 client initialized with bucket: {R2_BUCKET_NAME}")
else:
    logger.warning("⚠️ R2 credentials not set. Images will be returned as base64.")

# ==============================
# HELPERS
# ==============================
def _round_size(x, base=8):
    return max(base, (x // base) * base)

def upload_to_r2(image_bytes: bytes, job_id: str, index: int, content_type: str = "image/png") -> str:
    if not r2_client:
        raise ValueError("R2 client not initialized")
    if not R2_PUBLIC_URL:
        raise ValueError("R2_PUBLIC_URL not set")
    
    filename = f"videos/images/{job_id}/scene-{index}.png"
    
    try:
        r2_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=filename,
            Body=image_bytes,
            ContentType=content_type,
        )
        return f"{R2_PUBLIC_URL.rstrip('/')}/{filename}"
    except Exception as e:
        logger.error(f"❌ R2 upload failed: {e}")
        raise

def load_image(item):
    if "image_url" in item:
        response = requests.get(item["image_url"], timeout=30)
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif "image_base64" in item:
        image_data = base64.b64decode(item["image_base64"])
        return Image.open(BytesIO(image_data)).convert("RGB")
    return None

def update_dimensions(image, target_width=None, target_height=None):
    if target_width and target_height:
        return _round_size(target_width), _round_size(target_height)
        
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
    return _round_size(new_width), _round_size(new_height)

# ==============================
# MAIN HANDLER
# ==============================
@torch.inference_mode()
def handler(job):
    global current_lora_key
    
    pipeline = get_pipe()
    input_data = job.get("input", {})
    job_id = job.get("id", "unknown")
    
    # -------- Validation --------
    images_input = input_data.get("images")
    if not images_input or not isinstance(images_input, list):
        return {"error": "Input 'images' list is required"}
    
    prompt = input_data.get("prompt", "")
    lora_adapter = input_data.get("lora_adapter")
    lora_repo = input_data.get("lora_repo")
    lora_weight_name = input_data.get("lora_weight_name")
    lora_scale = float(input_data.get("lora_scale", 1.0))
    
    steps = int(input_data.get("num_inference_steps", 4))
    guidance = float(input_data.get("guidance_scale", 1.0))
    seed = input_data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    try:
        # -------- LoRA Management --------
        new_lora_key = None
        lora_spec = None
        
        if lora_repo:
            new_lora_key = f"repo:{lora_repo}"
        elif lora_adapter and lora_adapter in ADAPTER_SPECS:
            lora_spec = ADAPTER_SPECS[lora_adapter]
            new_lora_key = f"adapter:{lora_adapter}"
            
        if new_lora_key != current_lora_key:
            if current_lora_key:
                logger.info(f"🧹 Unloading LoRA: {current_lora_key}")
                pipeline.unload_lora_weights()
            
            if lora_repo:
                logger.info(f"📦 Loading custom LoRA: {lora_repo}")
                pipeline.load_lora_weights(
                    lora_repo,
                    weight_name=lora_weight_name,
                    adapter_name="lora_active",
                    cache_dir=HF_CACHE_DIR
                )
                current_lora_key = new_lora_key
            elif lora_spec:
                logger.info(f"📦 Loading built-in LoRA: {lora_adapter}")
                pipeline.load_lora_weights(
                    lora_spec["repo"],
                    weight_name=lora_spec["weights"],
                    adapter_name="lora_active",
                    cache_dir=HF_CACHE_DIR
                )
                current_lora_key = new_lora_key
            else:
                current_lora_key = None

        if current_lora_key:
            pipeline.set_adapters(["lora_active"], adapter_weights=[lora_scale])

        # -------- Load Images --------
        pil_images = []
        for item in images_input:
            img = load_image(item)
            if img:
                pil_images.append(img)
        
        if not pil_images:
            return {"error": "Could not load any input images"}

        # -------- Inference --------
        width, height = update_dimensions(
            pil_images[0], 
            input_data.get("width"), 
            input_data.get("height")
        )
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

        logger.info(f"🎨 Generating with prompt: {prompt}")
        output = pipeline(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance,
        ).images[0]

        # -------- Output Processing --------
        buf = BytesIO()
        output.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        
        result = {"seed": seed}
        if r2_client and R2_PUBLIC_URL:
            try:
                url = upload_to_r2(image_bytes, job_id, 0)
                result["image_url"] = url
                logger.info(f"✅ Uploaded to R2: {url}")
            except Exception as e:
                logger.warning(f"⚠️ R2 upload failed, falling back to base64")
                result["image_base64"] = base64.b64encode(image_bytes).decode()
        else:
            result["image_base64"] = base64.b64encode(image_bytes).decode()

        return {"images": [result]}

    except Exception as e:
        logger.error(f"❌ Inference error: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
