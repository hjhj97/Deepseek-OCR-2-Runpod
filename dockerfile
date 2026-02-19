FROM vllm/vllm-openai:v0.8.5

# pip 업그레이드
RUN pip install --upgrade pip --root-user-action=ignore

# requirements.txt 복사 및 설치
COPY DeepSeek-OCR2-master/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --root-user-action=ignore

# FastAPI 관련 패키지 설치
RUN pip install fastapi uvicorn python-multipart --root-user-action=ignore

# flash-attn 설치
RUN pip install flash-attn==2.7.3 --no-build-isolation --root-user-action=ignore

# 소스 코드 복사
COPY DeepSeek-OCR2-master/ /DeepSeek-OCR2-master/

# 작업 디렉토리 설정 (ocr_api.py가 있는 곳)
WORKDIR /DeepSeek-OCR2-master/DeepSeek-OCR2-vllm

# API 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_USE_V1=0

# API 서버 실행
ENTRYPOINT ["python3", "ocr_api.py"]