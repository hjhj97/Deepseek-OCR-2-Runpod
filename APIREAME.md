# DeepSeek OCR2 API 사용법

## 서버 실행
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name deepseek-ocr2-api \
  deepseek-ocr2-api
```

## API 엔드포인트

### 1. OCR 요청

**POST** `/ocr`

PDF 파일을 업로드하여 OCR 작업을 시작합니다.
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@문서.pdf" | jq
```

응답:
```json
{
  "job_id": "0aba6f3a-a8a1-4348-99a9-c7775c79327d",
  "status": "queued"
}
```

### 2. 상태 확인

**GET** `/ocr/status/{job_id}`

작업 진행 상태를 확인합니다.
```bash
curl "http://localhost:8000/ocr/status/{job_id}" | jq
```

응답:
```json
{
  "status": "completed",
  "filename": "문서.pdf",
  "total_pages": 15,
  "result": {
    "mmd_path": "/tmp/ocr_outputs/{job_id}/문서.mmd",
    "mmd_det_path": "/tmp/ocr_outputs/{job_id}/문서_det.mmd",
    "layout_pdf_path": "/tmp/ocr_outputs/{job_id}/문서_layouts.pdf"
  }
}
```

| status | 설명 |
|--------|------|
| queued | 대기 중 |
| processing | 처리 중 |
| completed | 완료 |
| failed | 실패 |

### 3. 파일 다운로드

**GET** `/ocr/download/{job_id}/{file_type}`

| file_type | 설명 |
|-----------|------|
| mmd | OCR 결과 (정리됨) |
| mmd_det | OCR 결과 (좌표 포함) |
| layout_pdf | 레이아웃 시각화 PDF |
```bash
# mmd 파일 다운로드
curl -OJ "http://localhost:8000/ocr/download/{job_id}/mmd"

# 레이아웃 PDF 다운로드
curl -OJ "http://localhost:8000/ocr/download/{job_id}/layout_pdf"
```

### 4. 서버 상태 확인

**GET** `/health`
```bash
curl "http://localhost:8000/health"
```

## 전체 사용 예시
```bash
# 1. OCR 요청
JOB_ID=$(curl -s -X POST "http://localhost:8000/ocr" \
  -F "file=@문서.pdf" | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# 2. 완료 대기
while true; do
  STATUS=$(curl -s "http://localhost:8000/ocr/status/$JOB_ID" | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "completed" ]; then
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "작업 실패"
    exit 1
  fi
  
  sleep 3
done

# 3. 결과 다운로드
curl -OJ "http://localhost:8000/ocr/download/$JOB_ID/mmd"
curl -OJ "http://localhost:8000/ocr/download/$JOB_ID/layout_pdf"
```

## Python 클라이언트 예시
```python
import requests
import time

# OCR 요청
with open("문서.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr",
        files={"file": f}
    )
job_id = response.json()["job_id"]

# 완료 대기
while True:
    status = requests.get(f"http://localhost:8000/ocr/status/{job_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(3)

# 결과 다운로드
mmd = requests.get(f"http://localhost:8000/ocr/download/{job_id}/mmd")
with open("결과.mmd", "wb") as f:
    f.write(mmd.content)
```