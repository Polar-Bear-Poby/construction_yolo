#!/bin/bash

# YOLO26 Python 래퍼 기반 단계별 학습 스크립트
# W&B 자동 연결, 고급 로깅, 에러 처리
# 1단계: s(GPU 0,1) + m(GPU 2,3) 동시 학습
# 2단계: l(GPU 0,1,2,3) + x(GPU 0,1,2,3) 순차 학습
# 사용법: ./run_all_training_cli.sh

# .env 파일 로드 (WANDB_API_KEY 등)
if [ -f "/home/themiraclesoft/wishket/.env" ]; then
    echo "📄 .env 파일 로드 중..."
    export $(grep -v '^#' /home/themiraclesoft/wishket/.env | xargs)
    echo "✅ 환경변수 로드 완료"
fi

echo "======================================"
echo "YOLO26 Python 래퍼 학습 시스템"
echo "======================================"
echo "1단계: s + m 모델 동시 학습 (GPU 분할)"
echo "2단계: l + x 모델 순차 학습 (GPU 전체)"
echo "W&B 자동 연결 및 Fitness 모니터링"
echo "======================================"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# 로그 디렉토리 설정
LOG_DIR="/home/themiraclesoft/wishket/training_logs"
mkdir -p $LOG_DIR

echo "📁 로그 저장 위치: $LOG_DIR"
echo ""

# W&B 프로젝트명
WANDB_PROJECT="yolo26-construction"

# ===== 1단계: s + m 모델 동시 학습 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 1단계: YOLO26s + YOLO26m 동시 학습 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "시작 시간: $(date)"

# YOLO26s 백그라운드 학습 (GPU 0,1)
echo "🔥 YOLO26s 학습 시작 (GPU 0,1)..."
uv run python train_yolo26_cli.py \
    --model_size s \
    --epochs 300 \
    --batch_size 32 \
    --device "0,1" \
    --optimizer MuSGD \
    --lr 0.001 \
    --patience 50 \
    --run_name "yolo26s_stage1_$(date +%Y%m%d_%H%M%S)" > $LOG_DIR/yolo26s_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_S=$!
echo "📝 YOLO26s PID: $PID_S"

# YOLO26m 백그라운드 학습 (GPU 2,3)
echo "🔥 YOLO26m 학습 시작 (GPU 2,3)..."
uv run python train_yolo26_cli.py \
    --model_size m \
    --epochs 300 \
    --batch_size 32 \
    --device "2,3" \
    --optimizer MuSGD \
    --lr 0.001 \
    --patience 50 \
    --run_name "yolo26m_stage1_$(date +%Y%m%d_%H%M%S)" > $LOG_DIR/yolo26m_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID_M=$!
echo "📝 YOLO26m PID: $PID_M"

echo ""
echo "⏳ s + m 모델 학습 완료 대기 중..."

# s 모델 완료 대기
wait $PID_S
RESULT_S=$?
if [ $RESULT_S -eq 0 ]; then
    echo "✅ YOLO26s 학습 완료! - $(date)"
else
    echo "❌ YOLO26s 학습 실패! (종료 코드: $RESULT_S) - $(date)"
fi

# m 모델 완료 대기
wait $PID_M
RESULT_M=$?
if [ $RESULT_M -eq 0 ]; then
    echo "✅ YOLO26m 학습 완료! - $(date)"
else
    echo "❌ YOLO26m 학습 실패! (종료 코드: $RESULT_M) - $(date)"
fi

echo ""
echo "🎯 1단계 완료: s + m 모델 학습 종료"

# GPU 메모리 정리
echo "🧹 GPU 메모리 정리 중..."
nvidia-smi --gpu-reset-ecc=0,1,2,3 2>/dev/null || true
sleep 10

# ===== 2단계: l + x 모델 순차 학습 =====
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 2단계: YOLO26l + YOLO26x 순차 학습 시작"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# YOLO26l 학습 (GPU 0,1,2,3)
echo "🔥 YOLO26l 학습 시작 (GPU 0,1,2,3)..."
echo "시작 시간: $(date)"
uv run python train_yolo26_cli.py \
    --model_size l \
    --epochs 300 \
    --batch_size 32 \
    --device "0,1,2,3" \
    --optimizer MuSGD \
    --lr 0.0008 \
    --patience 50 \
    --run_name "yolo26l_stage2_$(date +%Y%m%d_%H%M%S)"

RESULT_L=$?
if [ $RESULT_L -eq 0 ]; then
    echo "✅ YOLO26l 학습 완료! - $(date)"
else
    echo "❌ YOLO26l 학습 실패! (종료 코드: $RESULT_L) - $(date)"
fi

# GPU 메모리 정리
echo "🧹 GPU 메모리 정리 중..."
nvidia-smi --gpu-reset-ecc=0,1,2,3 2>/dev/null || true
sleep 10

# YOLO26x 학습 (GPU 0,1,2,3)
echo "🔥 YOLO26x 학습 시작 (GPU 0,1,2,3)..."
echo "시작 시간: $(date)"
uv run python train_yolo26_cli.py \
    --model_size x \
    --epochs 300 \
    --batch_size 24 \
    --device "0,1,2,3" \
    --optimizer MuSGD \
    --lr 0.0006 \
    --patience 50 \
    --run_name "yolo26x_stage2_$(date +%Y%m%d_%H%M%S)"

RESULT_X=$?
if [ $RESULT_X -eq 0 ]; then
    echo "✅ YOLO26x 학습 완료! - $(date)"
else
    echo "❌ YOLO26x 학습 실패! (종료 코드: $RESULT_X) - $(date)"
fi

echo ""
echo "🎯 2단계 완료: l + x 모델 학습 종료"

# 종료 시간 기록 및 결과 요약
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "======================================"
echo "🎉 모든 YOLO26 모델 학습 완료!"
echo "======================================"
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분"
echo "완료 시간: $(date)"
echo ""
echo "📊 학습 결과 요약:"
echo "  YOLO26s: $([ $RESULT_S -eq 0 ] && echo '✅ 성공' || echo '❌ 실패')"
echo "  YOLO26m: $([ $RESULT_M -eq 0 ] && echo '✅ 성공' || echo '❌ 실패')"
echo "  YOLO26l: $([ $RESULT_L -eq 0 ] && echo '✅ 성공' || echo '❌ 실패')"
echo "  YOLO26x: $([ $RESULT_X -eq 0 ] && echo '✅ 성공' || echo '❌ 실패')"
echo ""
echo "📁 결과 확인:"
echo "  로컬: /home/themiraclesoft/wishket/results/"
echo "  로그: $LOG_DIR/"
echo "  W&B: https://wandb.ai/ (프로젝트: $WANDB_PROJECT)"
echo ""
echo "💡 사용법:"
echo "  기본 (Python+W&B): ./run_all_training_cli.sh --use-python"
echo "  CLI 직접: ./run_all_training_cli.sh"
echo "  백그라운드: nohup ./run_all_training_cli.sh --use-python > training_output.log 2>&1 &"
echo "  W&B 대시보드: https://wandb.ai/your-username/yolo26-construction"
echo "======================================"