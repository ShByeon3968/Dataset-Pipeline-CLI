import time
import typer

from cli import client, display

app = typer.Typer(help="합성 데이터 증강(Augmentation) 파이프라인")

@app.command("run")
def run(
    dataset_id: int = typer.Option(..., "--dataset-id", "-d", help="대상 데이터셋 ID"),
    model: str = typer.Option("qwen", "--model", "-m", help="사용할 모델 (qwen 또는 flux)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="증강 명령어 (예: 'Move the camera left')"),
    negative_prompt: str = typer.Option("", "--negative-prompt", "-n", help="네거티브 프롬프트"),
    strength: float = typer.Option(0.8, "--strength", "-s", help="변환 강도 (0.0 ~ 1.0)"),
    steps: int = typer.Option(50, "--steps", help="추론 스텝 수")
):
    """
    지정된 모델(Qwen 또는 FLUX)을 사용해 데이터셋 이미지들을 변환하여 합성 이미지를 생성합니다.
    """
    display.info(f"데이터셋 {dataset_id}에 대해 {model} 모델을 사용한 증강 파이프라인을 시작합니다.")
    
    # 프리셋 처리: "45,left" -> "Rotate the camera 45 degrees to the left"
    if "," in prompt:
        parts = [p.strip().lower() for p in prompt.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            degrees, direction = parts
            expanded_prompt = f"Rotate the camera {degrees} degrees to the {direction}"
            display.info(f"💡 프리셋 감지: '{prompt}' -> '{expanded_prompt}'")
            prompt = expanded_prompt
            
    display.info(f"최종 프롬프트: '{prompt}'")
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "num_inference_steps": steps,
        "guidance_scale": 7.5,
        "model_type": model
    }
    
    try:
        res = client.post(f"/api/v1/datasets/{dataset_id}/augment", json=payload)
        task_id = res.get("task_id")
        if not task_id:
            display.error("Task ID를 받지 못했습니다.")
            raise typer.Exit(1)
            
        display.info("백그라운드에서 증강 작업이 시작되었습니다.")
        
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=display.console
        ) as progress:
            task_gui = progress.add_task("[cyan]이미지 증강 시작 대기 중...", total=100)
            
            while True:
                status_res = client.get(f"/api/v1/datasets/{dataset_id}/images/tasks/{task_id}")
                status = status_res.get("status")
                
                prog_data = status_res.get("progress")
                if prog_data:
                    c_img = prog_data.get("current_image", 0)
                    t_imgs = prog_data.get("total_images", 0)
                    c_step = prog_data.get("current_step", 0)
                    t_steps = prog_data.get("total_steps", 0)
                    
                    if t_imgs > 0 and t_steps > 0:
                        total_work = t_imgs * t_steps
                        current_work = (c_img - 1) * t_steps + c_step
                        
                        progress.update(
                            task_gui, 
                            completed=current_work, 
                            total=total_work,
                            description=f"[cyan]이미지 증강 중... ({c_img}/{t_imgs}) - 스텝 ({c_step}/{t_steps})"
                        )
                
                if status == "done":
                    progress.update(task_gui, completed=100, total=100, description="[green]증강 완료!")
                    result = status_res.get("result", {})
                    added = result.get("added", 0)
                    errors = result.get("errors", [])
                    display.success(f"증강 완료! {added}개의 이미지가 추가되었습니다.")
                    if errors:
                        display.warn(f"{len(errors)}개의 에러 발생. 첫번째 에러: {errors[0]}")
                    break
                elif status == "error":
                    progress.update(task_gui, description="[red]증강 실패")
                    display.error(f"증강 작업 실패: {status_res.get('error')}")
                    break
                    
                time.sleep(1.0)
                
    except client.PipelineError as e:
        display.error(str(e))
        raise typer.Exit(1)
