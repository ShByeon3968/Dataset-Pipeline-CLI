import time
import typer

from cli import client, display

app = typer.Typer(help="합성 데이터 증강(Augmentation) 파이프라인")

@app.command("run")
def run(
    dataset_id: int = typer.Option(..., "--dataset-id", "-d", help="대상 데이터셋 ID"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="증강 명령어 (예: 'Move the camera left')"),
    negative_prompt: str = typer.Option("", "--negative-prompt", "-n", help="네거티브 프롬프트"),
    strength: float = typer.Option(0.8, "--strength", "-s", help="변환 강도 (0.0 ~ 1.0)"),
    steps: int = typer.Option(50, "--steps", help="추론 스텝 수")
):
    """
    Qwen-Edit 모델을 사용해 데이터셋 이미지들의 구도를 변환하여 합성 이미지를 생성합니다.
    """
    display.info(f"데이터셋 {dataset_id}에 대해 증강 파이프라인을 시작합니다.")
    
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
        "guidance_scale": 7.5
    }
    
    try:
        res = client.post(f"/api/v1/datasets/{dataset_id}/augment", json=payload)
        task_id = res.get("task_id")
        if not task_id:
            display.error("Task ID를 받지 못했습니다.")
            raise typer.Exit(1)
            
        display.info("백그라운드에서 증강 작업이 시작되었습니다.")
        
        with display.console.status("[bold cyan]이미지 증강 중... (시간이 오래 걸릴 수 있습니다)"):
            while True:
                status_res = client.get(f"/api/v1/datasets/{dataset_id}/images/tasks/{task_id}")
                status = status_res.get("status")
                
                if status == "done":
                    result = status_res.get("result", {})
                    added = result.get("added", 0)
                    errors = result.get("errors", [])
                    display.success(f"증강 완료! {added}개의 이미지가 추가되었습니다.")
                    if errors:
                        display.warn(f"{len(errors)}개의 에러 발생. 첫번째 에러: {errors[0]}")
                    break
                elif status == "error":
                    display.error(f"증강 작업 실패: {status_res.get('error')}")
                    break
                    
                time.sleep(2.0)
                
    except client.PipelineError as e:
        display.error(str(e))
        raise typer.Exit(1)
