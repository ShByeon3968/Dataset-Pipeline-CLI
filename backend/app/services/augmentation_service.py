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

# Lazy initialization of diffusers pipelines to avoid loading when not used
_qwen_pipeline = None
_flux_pipeline = None

def _get_qwen_pipeline():
    global _qwen_pipeline
    
    # 여러 요청이 동시에 초기화를 시도하지 못하도록 Lock 처리
    with _pipeline_lock:
        if _qwen_pipeline is None:
            import torch
            
            # transformers 5.10.x 이상과 PyTorch 2.5.1의 float8 속성 충돌 방지
            if not hasattr(torch, "float8_e8m0fnu"):
                torch.float8_e8m0fnu = None

            # Qwen2.5-VL config의 dict object has no attribute 'to_dict' 에러 방지를 위한 몽키패치
            try:
                from transformers.generation.configuration_utils import GenerationConfig
                from transformers import PretrainedConfig
                
                _orig_from_model_config = GenerationConfig.from_model_config
                
                @classmethod
                def patched_from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
                    orig_get_text_config = model_config.get_text_config
                    def patched_get_text_config(*args, **kwargs):
                        config = orig_get_text_config(*args, **kwargs)
                        if isinstance(config, dict):
                            class DictWrapper:
                                def __init__(self, d):
                                    self.d = d
                                def to_dict(self):
                                    return self.d
                            return DictWrapper(config)
                        return config
                    model_config.get_text_config = patched_get_text_config
                    try:
                        return _orig_from_model_config(model_config)
                    finally:
                        model_config.get_text_config = orig_get_text_config
                        
                GenerationConfig.from_model_config = patched_from_model_config
                logger.info("Successfully patched GenerationConfig.from_model_config for Qwen2.5-VL")
            except Exception as patch_err:
                logger.warning(f"Failed to patch GenerationConfig: {patch_err}")

            # Qwen2.5-VL lm_head quantization을 강제하기 위한 몽키패치
            try:
                import transformers.integrations
                import transformers.integrations.bitsandbytes as bnb_integration
                orig_replace = bnb_integration.replace_with_bnb_linear
                
                def patched_replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
                    if modules_to_not_convert is not None:
                        # modules_to_not_convert 리스트에서 lm_head를 완전히 제외시켜 양자화되도록 유도
                        modules_to_not_convert = [m for m in modules_to_not_convert if m != "lm_head"]
                    return orig_replace(model, modules_to_not_convert, current_key_name, quantization_config)
                    
                transformers.integrations.replace_with_bnb_linear = patched_replace_with_bnb_linear
                bnb_integration.replace_with_bnb_linear = patched_replace_with_bnb_linear
                logger.info("Successfully patched replace_with_bnb_linear in both namespaces")
            except Exception as e:
                logger.warning(f"Failed to patch replace_with_bnb_linear: {e}")

            # Qwen2.5-VL lm_head weight tying 방지를 위한 몽키패치
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                Qwen2_5_VLForConditionalGeneration.tie_weights = lambda self: None
                logger.info("Successfully bypassed Qwen2_5_VLForConditionalGeneration.tie_weights")
            except Exception as e:
                logger.warning(f"Failed to patch tie_weights: {e}")

            # Bnb4BitHfQuantizer.check_quantized_param 및 create_quantized_param 교정 패치
            try:
                from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
                import bitsandbytes as bnb
                from transformers.modeling_utils import get_module_from_name
                
                orig_check = Bnb4BitHfQuantizer.check_quantized_param
                orig_create = Bnb4BitHfQuantizer.create_quantized_param
                
                def patched_check_quantized_param(self, model, param_value, param_name, state_dict, **kwargs):
                    module, tensor_name = get_module_from_name(model, param_name)
                    if isinstance(module, bnb.nn.Linear4bit) and tensor_name == "weight":
                        return True
                    return orig_check(self, model, param_value, param_name, state_dict, **kwargs)
                    
                def patched_create_quantized_param(self, model, param_value, param_name, target_device, state_dict, unexpected_keys=None):
                    module, tensor_name = get_module_from_name(model, param_name)
                    if isinstance(module, bnb.nn.Linear4bit) and tensor_name == "weight":
                        old_p = module._parameters.get(tensor_name, None)
                        if old_p is not None and not isinstance(old_p, bnb.nn.Params4bit):
                            logger.info(f"Converting non-Params4bit weight of {param_name} to Params4bit on meta device...")
                            # create a dummy Params4bit on meta device
                            dummy_data = torch.empty(old_p.shape, device="meta")
                            kwargs = getattr(old_p, "__dict__", {})
                            new_p = bnb.nn.Params4bit(dummy_data, requires_grad=old_p.requires_grad, **kwargs)
                            module._parameters[tensor_name] = new_p
                    return orig_create(self, model, param_value, param_name, target_device, state_dict, unexpected_keys)
                    
                Bnb4BitHfQuantizer.check_quantized_param = patched_check_quantized_param
                Bnb4BitHfQuantizer.create_quantized_param = patched_create_quantized_param
                logger.info("Successfully patched Bnb4BitHfQuantizer check and create quantized param methods")
            except Exception as e:
                logger.warning(f"Failed to patch Bnb4BitHfQuantizer: {e}")
                 
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
            _qwen_pipeline = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=False,
                use_safetensors=True
            )

            # 컴포넌트별 GPU 스케줄링 — 24GB 단일 GPU 로 동작
            _qwen_pipeline.enable_model_cpu_offload()

            # ── Attention Slicing: 8.23GiB 추론 오버헤드를 잘게 나눠 처리
            #    1024x1024 이미지의 Attention 행렬을 한 번에 계산하면 ~8.23GiB 필요.
            #    슬라이싱하면 Peak 사용량을 ~2-3GiB 수준으로 낮출 수 있음. (속도는 조금 느려짐)
            _qwen_pipeline.enable_attention_slicing(slice_size=1)

            logger.info("Loading Multiple-angles LoRA...")
            _qwen_pipeline.load_lora_weights("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA")
            _qwen_pipeline.set_progress_bar_config(disable=True)

            # ── VAE 메모리 절약
            if hasattr(_qwen_pipeline.vae, "enable_tiling"):
                _qwen_pipeline.vae.enable_tiling()
            if hasattr(_qwen_pipeline.vae, "enable_slicing"):
                _qwen_pipeline.vae.enable_slicing()

            if hasattr(_qwen_pipeline, "safety_checker"):
                _qwen_pipeline.safety_checker = None

            logger.info("Qwen Pipeline ready.")

    return _qwen_pipeline

def _get_flux_pipeline():
    global _flux_pipeline
    
    with _pipeline_lock:
        if _flux_pipeline is None:
            import torch
            
            # transformers 5.10.x 이상과 PyTorch 2.5.1의 float8 속성 충돌 방지
            if not hasattr(torch, "float8_e8m0fnu"):
                torch.float8_e8m0fnu = None
                
            from diffusers import AutoPipelineForImage2Image
            
            # ── 속도 최적화: TF32 허용
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Loading FLUX.2-klein-9B AutoPipelineForImage2Image...")
            
            _flux_pipeline = AutoPipelineForImage2Image.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # VRAM 최적화
            _flux_pipeline.enable_model_cpu_offload()
            _flux_pipeline.set_progress_bar_config(disable=True)
            
            logger.info("FLUX Pipeline ready.")
            
    return _flux_pipeline


async def run_augmentation_task(
    db: AsyncSession,
    dataset_id: int,
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    model_type: str = "qwen",
    progress_callback=None
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

        total_images = len(images)
        logger.info(f"Starting augmentation on {total_images} images for dataset {dataset_id} using {model_type}")
        
        if model_type == "flux":
            pipe = await asyncio.to_thread(_get_flux_pipeline)
        else:
            pipe = await asyncio.to_thread(_get_qwen_pipeline)
        
        added_count = 0
        errors = []

        # Iterate over all images and augment
        for img_idx, img in enumerate(images):
            try:
                # 이미 증강된 이미지 건너뛰기 (파일명이 'aug_'로 시작하거나 '_aug_'가 포함된 경우)
                if img.filename and (img.filename.startswith("aug_") or "_aug_" in img.filename):
                    logger.info(f"Skipping already augmented image: {img.filename}")
                    continue
                    
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
                        if progress_callback:
                            progress_callback(img_idx + 1, total_images, step_index + 1, num_inference_steps)
                        if (step_index + 1) % 5 == 0 or (step_index + 1) == num_inference_steps:
                            logger.info(f"Augmentation progress: Step {step_index + 1}/{num_inference_steps} completed")
                        return callback_kwargs

                    # Set inputs based on user's reference script
                    if model_type == "flux":
                        inputs = {
                            "image": image,
                            "prompt": prompt,
                            "generator": torch.manual_seed(0),
                            "guidance_scale": guidance_scale,
                            # "strength": strength,
                            "num_inference_steps": num_inference_steps,
                            "callback_on_step_end": step_callback,
                        }
                    else:
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
        logger.exception("Augmentation failed")
        await db.rollback()
        return {"added": 0, "error": str(e), "errors": [str(e)]}
    finally:
        # Optionally free up memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
