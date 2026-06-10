import os
import io
import uuid
from typing import List, Dict, Any
from pathlib import Path
import logging
import threading

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import Image as DBImage
from app.services.file_handler import resolve_filepath, save_image_to_storage, calculate_phash, get_image_dimensions, get_file_format, calculate_md5_bytes

logger = logging.getLogger(__name__)
# 스레드 안전성을 위한 Lock 객체 생성
_pipeline_lock = threading.Lock()

# Lazy initialization of diffusers pipeline to avoid loading when not used
_pipeline = None

def _get_pipeline():
    global _pipeline
    
    # 여러 요청이 동시에 초기화를 시도하지 못하도록 Lock 처리
    with _pipeline_lock:
        if _pipeline is None:
            import torch
            from diffusers import QwenImageEditPipeline
            from transformers import BitsAndBytesConfig

            # ── 속도 최적화: TF32 허용
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # ── 로딩 전 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Loading Qwen-Image-Edit pipeline...")

            # ── device_map 없이 로드 후 enable_model_cpu_offload() 사용
            #    device_map="balanced" 시 bitsandbytes 가 초기화 시점에
            #    "CPU에 배치된 4-bit 레이어"를 감지해 에러를 냄.
            #    device_map 자체를 제거하면 이 검사가 아예 실행되지 않음.
            #    enable_model_cpu_offload() 는 컴포넌트 단위(text_encoder/transformer/vae)로
            #    GPU ↔ CPU 를 forward hook 으로 관리하므로 bitsandbytes 와 충돌하지 않음.
            _pipeline = QwenImageEditPipeline.from_pretrained(
                "ovedrive/Qwen-Image-Edit-2511-4bit",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=False,
                use_safetensors=True
            )

            # 컴포넌트별 GPU 스케줄링 — 24GB 단일 GPU 로 동작
            _pipeline.enable_model_cpu_offload()

            # ── Attention Slicing: 8.23GiB 추론 오버헤드를 잘게 나눠 처리
            #    1024x1024 이미지의 Attention 행렬을 한 번에 계산하면 ~8.23GiB 필요.
            #    슬라이싱하면 Peak 사용량을 ~2-3GiB 수준으로 낮출 수 있음. (속도는 조금 느려짐)
            _pipeline.enable_attention_slicing(slice_size=1)

            logger.info("Loading Multiple-angles LoRA...")
            _pipeline.load_lora_weights("dx8152/Qwen-Edit-2509-Multiple-angles")
            _pipeline.set_progress_bar_config(disable=True)

            # ── VAE 메모리 절약
            if hasattr(_pipeline.vae, "enable_tiling"):
                _pipeline.vae.enable_tiling()
            if hasattr(_pipeline.vae, "enable_slicing"):
                _pipeline.vae.enable_slicing()

            if hasattr(_pipeline, "safety_checker"):
                _pipeline.safety_checker = None

            logger.info("Pipeline ready.")

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
                
                # Resize: 512 이상이면 OOM. Attention 메모리는 해상도의 제곱에 비례.
                # 1024→512 축소 시 Attention 메모리 4배 감소.
                init_image.thumbnail((512, 512))
                
                # Run pipeline
                def _run_inference(p, image):
                    def step_callback(pipe, step_index, timestep, callback_kwargs):
                        if (step_index + 1) % 5 == 0 or (step_index + 1) == num_inference_steps:
                            logger.info(f"Augmentation progress: Step {step_index + 1}/{num_inference_steps} completed")
                        return callback_kwargs

                    # Set inputs based on user's reference script
                    inputs = {
                        "image": image,
                        "prompt": prompt,
                        "generator": torch.manual_seed(0),
                        "true_cfg_scale": 4.0,  # Qwen-Image-Edit uses true_cfg_scale instead of guidance_scale
                        "negative_prompt": negative_prompt if negative_prompt else " ",
                        "num_inference_steps": num_inference_steps,
                        "callback_on_step_end": step_callback,
                    }
                    with torch.inference_mode():
                        return p(**inputs).images[0]
                        
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
            finally:
                # [추가됨] 매 이미지 처리마다 사용 끝난 객체 정리 및 VRAM 확보
                if 'augmented_pil' in locals():
                    augmented_pil.close()
                if 'init_image' in locals():
                    init_image.close()
                
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
