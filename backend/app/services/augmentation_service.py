import os
import io
import uuid
from typing import List, Dict, Any
from pathlib import Path
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import Image as DBImage
from app.services.file_handler import resolve_filepath, save_image_to_storage, calculate_phash, get_image_dimensions, get_file_format, calculate_md5_bytes

logger = logging.getLogger(__name__)

# Lazy initialization of diffusers pipeline to avoid loading when not used
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import AutoPipelineForImage2Image
        from peft import PeftModel

        logger.info("Loading Qwen-Image-Edit-2509 pipeline...")
        # Check if CUDA is available, else use CPU (which will be extremely slow)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load base model across multiple GPUs
        _pipeline = AutoPipelineForImage2Image.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=dtype,
            variant=None,
            device_map="balanced"
        )

        logger.info("Loading Multiple-angles LoRA...")
        # Load multiple angles LoRA
        _pipeline.load_lora_weights("dx8152/Qwen-Edit-2509-Multiple-angles")
        
        # Multi-GPU distribution is active (device_map="balanced")
        # PyTorch 2.0+ built-in Scaled Dot Product Attention (SDPA) will be used automatically.

        # Fix black image issue (VAE NaN in FP16)
        if hasattr(_pipeline, "vae") and _pipeline.vae is not None:
            logger.info("Casting VAE to float32 to prevent black image (NaN) issue...")
            _pipeline.vae.to(dtype=torch.float32)

        # Disable safety checker to prevent false positive black images
        if hasattr(_pipeline, "safety_checker"):
            _pipeline.safety_checker = None
            
    return _pipeline

async def run_augmentation_task(
    db: AsyncSession,
    dataset_id: int,
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Dict[str, Any]:
    """
    Run the augmentation process on all images in a dataset using Qwen-Edit.
    """
    try:
        from PIL import Image as PILImage
        import torch
        
        import asyncio
        
        # Fetch all images for the dataset
        result = await db.execute(
            select(DBImage).where(DBImage.dataset_id == dataset_id)
        )
        images = result.scalars().all()
        
        if not images:
            return {"added": 0, "error": "No images found in dataset."}

        logger.info(f"Starting augmentation on {len(images)} images for dataset {dataset_id}")
        
        pipe = await asyncio.to_thread(_get_pipeline)
        
        added_count = 0
        errors = []

        # Iterate over all images and augment
        for img in images:
            try:
                abs_path = resolve_filepath(img.filepath)
                if not os.path.exists(abs_path):
                    continue
                
                # Load original image
                init_image = PILImage.open(abs_path).convert("RGB")
                
                # Resize if it's too large to prevent OOM
                init_image.thumbnail((1024, 1024))
                
                # Run pipeline
                def _run_inference(p, image):
                    with torch.no_grad():
                        return p(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=image,
                            # strength=strength,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps
                        ).images[0]
                        
                augmented_pil = await asyncio.to_thread(_run_inference, pipe, init_image)
                
                # Save augmented image to bytes
                buf = io.BytesIO()
                augmented_pil.save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()
                
                # Save to storage
                filename = f"aug_{uuid.uuid4().hex[:8]}.jpg"
                rel_path = save_image_to_storage(img_bytes, filename, dataset_id)[1]
                abs_new_path = resolve_filepath(rel_path)
                
                # Add new image to DB
                md5 = calculate_md5_bytes(img_bytes)
                w, h = get_image_dimensions(abs_new_path)
                phash = calculate_phash(abs_new_path)
                fmt = get_file_format(abs_new_path)
                
                new_db_img = DBImage(
                    dataset_id=dataset_id,
                    filename=Path(abs_new_path).name,
                    filepath=rel_path,
                    width=w, height=h, format=fmt,
                    file_hash=md5, phash=phash,
                )
                db.add(new_db_img)
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error augmenting image {img.id}: {e}")
                errors.append(str(e))
                
        await db.commit()
        return {"added": added_count, "errors": errors}
        
    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        await db.rollback()
        return {"added": 0, "error": str(e), "errors": [str(e)]}
    finally:
        # Optionally free up memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
