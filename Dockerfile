# CUDA 12.8 + PyTorch 환경 기반 이미지
FROM nvidia/cuda:12.8.0-cudnn8-runtime-ubuntu22.04

# 필수 패키지 설치
RUN apt update && apt install -y \
    python3 python3-pip git curl nano

# pip 업그레이드 및 ML 관련 패키지 설치
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio \
    pandas scikit-learn matplotlib \
    mlflow fastapi uvicorn

# 작업 디렉토리 지정
WORKDIR /app

# Mac에서 만든 코드 복사 예정 위치
COPY . /app

# 컨테이너 실행 시 기본 명령어
CMD ["python3", "train.py"]