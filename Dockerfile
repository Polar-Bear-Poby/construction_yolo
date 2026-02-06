# YOLO26 건축 도면 객체 탐지 시스템 - Docker 이미지
# Base: NVIDIA CUDA + PyTorch
FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/root/.local/bin:$PATH"

# 시스템 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 버전 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# uv 설치 (빠른 Python 패키지 매니저)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 작업 디렉토리 설정
WORKDIR /workspace

# 프로젝트 파일 복사 (로그, PNG 제외)
COPY pyproject.toml uv.lock .python-version ./
COPY requirements.txt ./
COPY .env.example ./
COPY dataset/data.yml ./dataset/
COPY dataset/train/labels/*.txt ./dataset/train/labels/
COPY dataset/train/labels.cache ./dataset/train/
COPY dataset/valid/labels/*.txt ./dataset/valid/labels/
COPY dataset/valid/labels.cache ./dataset/valid/
COPY code/ ./code/
COPY fine_tuning_weights/*.pt ./fine_tuning_weights/
COPY run_all_training_cli.sh ./
COPY README.md ./

# 실행 권한 부여
RUN chmod +x run_all_training_cli.sh

# uv로 의존성 설치
RUN ~/.local/bin/uv sync --no-dev

# 디렉토리 생성 (런타임에 사용)
RUN mkdir -p /workspace/training_logs \
    /workspace/results \
    /workspace/inference_results \
    /workspace/dataset/train/images \
    /workspace/dataset/valid/images

# GPU 사용 확인 헬스체크
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 기본 명령어 (컨테이너 실행 시)
CMD ["/bin/bash"]

# 사용법:
# 1. 이미지 빌드: docker build -t yolo26-construction:latest .
# 2. 컨테이너 실행: docker run --gpus all -it -v $(pwd)/dataset/train/images:/workspace/dataset/train/images -v $(pwd)/dataset/valid/images:/workspace/dataset/valid/images yolo26-construction:latest
# 3. 학습 시작: ./run_all_training_cli.sh
