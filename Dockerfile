FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 코드 복사
COPY . /app

# 패키지 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 실행 명령
CMD ["python", "main.py"]




#docker build --no-cache -t titan-trainer .

#docker run --rm --gpus all -p 5000:5000 -v "C:/Users/USER/PycharmProjects/research/data:/app/data" titan-trainer
#pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime