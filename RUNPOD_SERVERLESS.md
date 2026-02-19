# Runpod Serverless Setup

이 문서는 현재 저장소를 Runpod Serverless에서 실행하기 위한 최소 설정 절차입니다.

## 1) 준비된 파일

- `dockerfile.serverless`
- `DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/runpod_handler.py`

`runpod_handler.py`는 다음을 지원합니다.

- 입력:
  - `input.pdf_base64` (권장)
  - `input.pdf_url` (대안)
  - `input.filename` (선택)
- 출력:
  - 기본: `mmd_text`, `total_pages`
  - 옵션:
    - `include_mmd_det_text: true`
    - `include_layout_pdf_base64: true`
    - `include_output_files_base64: true`

## 2) 이미지 빌드/푸시

아래 `<registry>/<image>`를 본인 레지스트리로 바꿔서 실행하세요.

```bash
docker build --platform linux/amd64 -t <registry>/<image>:serverless -f dockerfile.serverless .
docker push <registry>/<image>:serverless
```

## 3) Runpod Endpoint 생성

Runpod 콘솔에서:

1. **Serverless Template** 생성
   - Container Image: `<registry>/<image>:serverless`
   - GPU: L4/A10 이상 권장
   - Container Disk: 모델 다운로드 고려해서 여유 있게 설정
2. **Serverless Endpoint** 생성 후 Template 연결
3. 필요 시 환경변수 추가
   - `HF_TOKEN` (필요한 경우)
   - `RUNPOD_PRELOAD_MODEL=1` (기본값, 콜드스타트 시 모델 선로딩)

## 4) 요청 예시

### A. `pdf_base64` 방식

```json
{
  "input": {
    "filename": "sample.pdf",
    "pdf_base64": "<BASE64_PDF>"
  }
}
```

### B. `pdf_url` 방식

```json
{
  "input": {
    "pdf_url": "https://example.com/sample.pdf",
    "filename": "sample.pdf",
    "include_mmd_det_text": true
  }
}
```

## 5) 로컬에서 base64 만들기

```bash
python3 - <<'PY'
import base64, json
with open("test.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
payload = {"input": {"filename": "test.pdf", "pdf_base64": b64}}
print(json.dumps(payload))
PY
```

## 6) 응답 예시

```json
{
  "status": "completed",
  "filename": "sample.pdf",
  "total_pages": 12,
  "mmd_text": "# ... markdown ..."
}
```

실패 시:

```json
{
  "status": "failed",
  "error": "...",
  "traceback": "..."
}
```

## 7) 운영 팁

- 긴 문서는 응답이 커질 수 있으므로 `layout_pdf_base64` 옵션은 필요할 때만 사용하세요.
- OOM이 나면 `DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/config.py`의
  `MAX_CONCURRENCY`, `NUM_WORKERS` 값을 낮추세요.
- 대용량 PDF를 자주 처리하면 payload 크기 때문에 `pdf_url` 방식이 더 안정적입니다.
