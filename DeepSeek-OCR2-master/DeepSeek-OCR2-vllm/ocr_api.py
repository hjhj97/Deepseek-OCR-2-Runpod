import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager

from run_dpsk_ocr2_pdf import init_model, run_ocr

# 작업 상태 저장
jobs = {}
OUTPUT_DIR = "/tmp/ocr_outputs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    print("🛑 서버 종료")


app = FastAPI(title="DeepSeek OCR2 API", lifespan=lifespan)


def background_ocr(job_id: str, pdf_bytes: bytes, filename: str):
    try:
        jobs[job_id]["status"] = "processing"
        output_path = f"{OUTPUT_DIR}/{job_id}"
        base_name = filename.replace('.pdf', '')
        
        result = run_ocr(pdf_bytes, output_path, base_name)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["total_pages"] = result["total_pages"]
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/ocr")
async def ocr(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "PDF 파일만 지원됩니다")
    
    job_id = str(uuid.uuid4())
    pdf_bytes = await file.read()
    
    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "total_pages": 0,
        "result": None
    }
    
    background_tasks.add_task(background_ocr, job_id, pdf_bytes, file.filename)
    
    return {"job_id": job_id, "status": "queued"}


@app.get("/ocr/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return jobs[job_id]


@app.get("/ocr/download/{job_id}/{file_type}")
async def download(job_id: str, file_type: str):
    if job_id not in jobs or jobs[job_id]["status"] != "completed":
        raise HTTPException(400, "작업이 완료되지 않았습니다")
    
    result = jobs[job_id]["result"]
    path_map = {
        "mmd": result["mmd_path"],
        "mmd_det": result["mmd_det_path"],
        "layout_pdf": result["layout_pdf_path"]
    }
    
    if file_type not in path_map:
        raise HTTPException(400, "file_type: mmd, mmd_det, layout_pdf")
    
    return FileResponse(path_map[file_type], filename=os.path.basename(path_map[file_type]))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)