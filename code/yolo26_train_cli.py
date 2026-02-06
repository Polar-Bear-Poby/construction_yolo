#!/usr/bin/env python3
"""
YOLO26 하이브리드 학습 시스템
- Python: W&B 통합, 로깅, 에러 핸들링, 성능 모니터링
- CLI: 실제 훈련 실행 (Ultralytics)
- 실시간 로그 스트리밍 및 진행률 추적
- 강력한 에러 복구 및 재시도 메커니즘
"""

import os
import subprocess
import sys
import time
import json
import signal
import threading
from pathlib import Path
import wandb
import argparse
from datetime import datetime
from dotenv import load_dotenv
import psutil
try:
    import gpustat
except ImportError:
    gpustat = None
try:
    import torch
except ImportError:
    torch = None

# 프로젝트 루트 자동 감지
PROJECT_ROOT = Path(__file__).parent.absolute()

# .env 로드 (프로젝트 루트 기준)
load_dotenv(PROJECT_ROOT / '.env')


def get_available_gpus():
    """사용 가능한 GPU 개수 자동 감지"""
    if torch and torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_device_string(num_gpus: int = None, force_single: bool = False):
    """
    GPU 디바이스 문자열 생성
    
    Args:
        num_gpus: 사용할 GPU 개수 (None이면 전체)
        force_single: True면 단일 GPU 사용
    
    Returns:
        "0,1,2,3" 형태의 문자열 또는 "cpu"
    """
    available = get_available_gpus()
    
    if available == 0:
        return "cpu"
    
    if force_single:
        return "0"
    
    if num_gpus is None:
        num_gpus = available
    else:
        num_gpus = min(num_gpus, available)
    
    return ','.join(map(str, range(num_gpus)))


def setup_wandb(project_name: str = "yolo26-construction", entity: str = None, config: dict = None):
    """
    W&B 설정 및 프로젝트 초기화
    
    Args:
        project_name: W&B 프로젝트명
        entity: W&B 엔티티명
        config: 실험 설정 딕셔너리
    
    Returns:
        wandb.run: W&B run 객체 또는 None
    """
    try:
        print("📊 Weights & Biases 초기화 중...")
        
        # W&B API 키 확인
        api_key = os.getenv('WANDB_API_KEY')
        if not api_key or api_key == 'your_wandb_api_key_here':
            print("❌ WANDB_API_KEY가 설정되지 않았습니다!")
            print("   1. .env.example을 .env로 복사")
            print("   2. https://wandb.ai/settings 에서 API 키 생성")
            print("   3. .env 파일에 WANDB_API_KEY 입력")
            return None
        
        # W&B 초기화
        run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            tags=["yolo26", "construction", "ultralytics", "fitness-tracking"],
            notes="하이브리드 Python+CLI 학습 시스템 - Fitness Score 실시간 모니터링"
        )
        
        # WandB에 주요 메트릭 정의
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("fitness/*", step_metric="epoch")
        wandb.define_metric("early_stopping/*", step_metric="epoch")
        
        # Fitness를 요약 메트릭으로 설정
        wandb.run.summary["best_fitness"] = 0.0
        wandb.run.summary["best_fitness_epoch"] = 0
        wandb.run.summary["patience_counter"] = 0
        
        # 환경변수 설정 (Ultralytics 자동 인식용)
        os.environ['WANDB_PROJECT'] = project_name
        if entity:
            os.environ['WANDB_ENTITY'] = entity
        
        # 시스템 정보 로깅
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "python_version": sys.version.split()[0]
        }
        
        # GPU 정보 추가
        try:
            if gpustat:
                gpu_stats = gpustat.new_query()
                system_info.update({
                    "gpu_count": len(gpu_stats),
                    "gpu_info": [gpu.name for gpu in gpu_stats]
                })
            else:
                system_info.update({
                    "gpu_count": "unknown",
                    "gpu_info": ["gpustat not available"]
                })
        except Exception as e:
            system_info.update({
                "gpu_count": "error",
                "gpu_info": [str(e)]
            })
            
        wandb.log({"system_info": system_info})
        
        print(f"✅ W&B 초기화 완료 - {run.url}")
        return run
        
    except Exception as e:
        print(f"❌ W&B 설정 실패: {e}")
        return None


class TrainingMonitor:
    """
    CLI 프로세스 모니터링 및 로깅 클래스
    """
    
    def __init__(self, log_file: Path, wandb_run=None):
        self.log_file = log_file
        self.wandb_run = wandb_run
        self.start_time = time.time()
        self.last_epoch = 0
        self.metrics = {
            'best_fitness': 0.0,
            'best_epoch': 0,
            'patience_counter': 0,
            'epochs_without_improvement': 0
        }
        self.should_stop = False
        self.fitness_history = []
        
    def parse_yolo_output(self, line: str):
        """
        YOLO 출력에서 메트릭 파싱 (fitness score 포함)
        """
        try:
            # 에포크 진행률 파싱 - 실제 로그에서 "7/300" 형태를 찾음
            if "/" in line and "/300" in line:
                # "[K      7/300      5.47G" 형태에서 에포크 번호 추출
                parts = line.split()
                for part in parts:
                    if "/300" in part:
                        try:
                            current = int(part.split("/")[0])
                            if current > self.last_epoch:
                                self.last_epoch = current
                                print(f"📊 에포크 {current} 진행 중...")
                                if self.wandb_run:
                                    wandb.log({"epoch": current}, step=current)
                        except (ValueError, IndexError):
                            continue
                        break
            
            # Validation 결과 파싱 (mAP 포함)
            if "all" in line and "mAP" not in line and len(line.split()) >= 6:
                try:
                    parts = line.split()
                    if len(parts) >= 6 and parts[0] == "all":
                        precision = float(parts[3])
                        recall = float(parts[4])
                        map50 = float(parts[5])
                        map50_95 = float(parts[6]) if len(parts) > 6 else 0.0
                        
                        # Fitness score 계산 (YOLO 표준 공식)
                        fitness = 0.1 * map50 + 0.9 * map50_95
                        self.fitness_history.append(fitness)
                        
                        # Fitness 개선 여부 확인
                        is_best = fitness > self.metrics['best_fitness']
                        if not is_best:
                            self.metrics['epochs_without_improvement'] += 1
                        else:
                            self.metrics['epochs_without_improvement'] = 0
                            
                        print(f"\n🎯 VALIDATION 결과 (에포크 {self.last_epoch}):")
                        print(f"   📈 Precision: {precision:.4f} ({precision*100:.1f}%)")
                        print(f"   📈 Recall: {recall:.4f} ({recall*100:.1f}%)")
                        print(f"   📊 mAP50: {map50:.4f} ({map50*100:.1f}%)")
                        print(f"   📊 mAP50-95: {map50_95:.4f} ({map50_95*100:.1f}%)")
                        print(f"   ⭐ FITNESS SCORE: {fitness:.4f} ({fitness*100:.1f}%)")
                        
                        if self.wandb_run:
                            # 기본 validation 메트릭
                            log_data = {
                                "epoch": self.last_epoch,
                                "val/precision": precision,
                                "val/recall": recall,
                                "val/mAP50": map50,
                                "val/mAP50-95": map50_95,
                                "fitness/current": fitness,
                                "fitness/best": max(self.fitness_history),
                                "early_stopping/epochs_without_improvement": self.metrics['epochs_without_improvement']
                            }
                            
                            # Fitness 트렌드 분석
                            if len(self.fitness_history) >= 5:
                                recent_trend = sum(self.fitness_history[-5:]) / 5
                                log_data["fitness/trend_recent_5"] = recent_trend
                                
                            if len(self.fitness_history) >= 10:
                                recent_vs_old = sum(self.fitness_history[-5:]) / 5 - sum(self.fitness_history[-10:-5]) / 5
                                log_data["fitness/improvement_rate"] = recent_vs_old
                                
                            wandb.log(log_data, step=self.last_epoch)
                            
                        # 최고 성능 추적 및 WandB 업데이트
                        if is_best:
                            self.metrics['best_fitness'] = fitness
                            self.metrics['best_epoch'] = self.last_epoch
                            print(f"🏆 NEW BEST FITNESS: {fitness:.4f} at epoch {self.last_epoch}")
                            
                            if self.wandb_run:
                                # WandB summary 업데이트
                                wandb.run.summary["best_fitness"] = fitness
                                wandb.run.summary["best_fitness_epoch"] = self.last_epoch
                                wandb.run.summary["best_precision"] = precision
                                wandb.run.summary["best_recall"] = recall
                                wandb.run.summary["best_mAP50"] = map50
                                wandb.run.summary["best_mAP50-95"] = map50_95
                                
                                # 최고 성능 알림 (W&B Alert)
                                try:
                                    wandb.alert(
                                        title="New Best Fitness Score!",
                                        text=f"🏆 New best fitness: {fitness:.4f} at epoch {self.last_epoch}\n" + 
                                             f"📊 mAP50: {map50:.3f}, mAP50-95: {map50_95:.3f}",
                                        level=wandb.AlertLevel.INFO
                                    )
                                except:
                                    pass  # Alert 실패 시 무시
                        else:
                            print(f"   📉 No improvement (Best: {self.metrics['best_fitness']:.4f} at epoch {self.metrics['best_epoch']})")
                            print(f"   ⏰ Epochs without improvement: {self.metrics['epochs_without_improvement']}")
                            
                        print(f"   {'='*60}")
                            
                except (ValueError, IndexError):
                    pass
            
            # Early stopping 정보 파싱
            if "EarlyStopping" in line or "patience" in line.lower():
                print(f"⚠️  Early Stopping: {line.strip()}")
                
                if self.wandb_run:
                    # Patience 정보 추출 시도
                    try:
                        if "patience" in line.lower() and "/" in line:
                            # patience 정보가 있는 경우 (예: 5/50)
                            patience_part = [p for p in line.split() if "/" in p and p.replace("/", "").replace("(", "").replace(")", "").isdigit()]
                            if patience_part:
                                current_patience, max_patience = patience_part[0].replace("(", "").replace(")", "").split("/")
                                wandb.log({
                                    "early_stopping/current_patience": int(current_patience),
                                    "early_stopping/max_patience": int(max_patience),
                                    "early_stopping/patience_ratio": int(current_patience) / int(max_patience)
                                }, step=self.last_epoch)
                                
                        if "EarlyStopping" in line and "triggered" in line.lower():
                            try:
                                wandb.alert(
                                    title="Early Stopping Triggered",
                                    text=f"🛑 Training stopped early at epoch {self.last_epoch}\n{line.strip()}",
                                    level=wandb.AlertLevel.WARN
                                )
                            except:
                                pass
                    except:
                        pass
                
        except Exception as e:
            pass  # 파싱 오류는 무시
    
    def log_line(self, line: str):
        """
        라인을 파일과 콘솔에 로깅
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_line = f"[{timestamp}] {line}"
        
        # 콘솔 출력
        print(formatted_line, end='')
        
        # 파일 저장
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_line)
            f.flush()
        
        # 메트릭 파싱
        self.parse_yolo_output(line)
        
        # W&B 실시간 로그
        if self.wandb_run and line.strip():
            try:
                wandb.log({"raw_log": line.strip()}, step=self.last_epoch)
            except:
                pass
    
    def signal_handler(self, signum, frame):
        """
        시그널 핸들러 (Ctrl+C 등)
        """
        print(f"\n⚠️ 시그널 {signum} 수신. 안전하게 종료 중...")
        self.should_stop = True
        if self.wandb_run:
            self.wandb_run.finish()
        sys.exit(0)


def build_yolo_command(args):
    """
    Ultralytics CLI 명령어 생성
    
    Args:
        args: argparse 인수들
    
    Returns:
        str: 실행할 CLI 명령어
    """
    # GPU 설정: 대형 모델(l, x)은 전체 GPU 사용
    is_large_model = args.model_size in ['l', 'x']
    
    if args.device is None:
        # 자동 감지: 대형 모델은 전체 GPU, 소형 모델은 사용자가 지정하거나 전체 GPU
        device = get_device_string()
    else:
        device = args.device
    
    print(f"🎮 Using device: {device} (model: yolo26{args.model_size})")
    
    cmd_parts = [
        "uv", "run", "yolo",
        "detect",  # task
        "train",   # mode
    ]
    
    # 필수 인수들
    cmd_parts.extend([
        f"model=yolo26{args.model_size}.pt",
        f"data={args.data_yaml}",
        f"epochs={args.epochs}",
        f"batch={args.batch_size}",
        f"imgsz={args.imgsz}",
        f"device={device}",
    ])
    
    # 옵티마이저 설정
    if args.optimizer:
        cmd_parts.append(f"optimizer={args.optimizer}")
        
    # 학습률 설정
    cmd_parts.extend([
        f"lr0={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"momentum={args.momentum}",
    ])
    
    # Early stopping
    cmd_parts.append(f"patience={args.patience}")
    
    # 성능 최적화 설정
    if args.amp:
        cmd_parts.append("amp=True")
    if args.cos_lr:
        cmd_parts.append("cos_lr=True")
    
    # 데이터 증강
    if args.mixup > 0:
        cmd_parts.append(f"mixup={args.mixup}")
    if args.mosaic != 1.0:
        cmd_parts.append(f"mosaic={args.mosaic}")
    
    # 출력 설정
    cmd_parts.extend([
        f"project={args.output_dir}",
        f"name={args.run_name}",
        "save=True",
        f"save_period={args.save_period}",
        "exist_ok=True",
    ])
    
    # 시스템 설정
    cmd_parts.extend([
        f"workers={args.workers}",
        f"cache={str(args.cache).lower()}",
        f"verbose={str(args.verbose).lower()}",
        "plots=True",
        "deterministic=False",  # 재현성보다 성능 우선
    ])
    
    return " ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description='YOLO26 하이브리드 학습 시스템 (Python + CLI)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 기본 모델 설정
    parser.add_argument('--model_size', '-m', default='x', choices=['n', 's', 'm', 'l', 'x'],
                        help='모델 크기')
    parser.add_argument('--data_yaml', default=str(PROJECT_ROOT / 'dataset' / 'data.yml'),
                        help='데이터셋 YAML 파일 경로')
    parser.add_argument('--epochs', type=int, default=300,
                        help='최대 학습 에포크')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='이미지 크기')
    parser.add_argument('--device', default=None,
                        help='GPU 디바이스 ID (기본: 자동 감지, 전체 GPU 사용)')
    
    # 옵티마이저 설정
    parser.add_argument('--optimizer', default='MuSGD',
                        choices=['MuSGD', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'],
                        help='옵티마이저 (MuSGD 추천)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='초기 학습률')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='가중치 감쇠')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='모멘텀')
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (fitness 기준)')
    
    # 성능 최적화
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Automatic Mixed Precision 사용')
    parser.add_argument('--cos_lr', action='store_true', default=True,
                        help='Cosine Learning Rate Scheduler 사용')
    
    # 데이터 증강
    parser.add_argument('--mixup', type=float, default=0.15,
                        help='Mixup 증강 비율')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic 증강 비율')
    
    # 출력 설정
    parser.add_argument('--output_dir', default=str(PROJECT_ROOT / 'results'),
                        help='결과 저장 디렉토리')
    parser.add_argument('--run_name', default=None,
                        help='실험 이름 (자동 생성됨)')
    parser.add_argument('--save_period', type=int, default=10,
                        help='체크포인트 저장 주기 (에포크)')
    
    # 시스템 설정
    parser.add_argument('--workers', type=int, default=8,
                        help='데이터로더 워커 수')
    parser.add_argument('--cache', action='store_true', default=True,
                        help='데이터셋 캐싱 사용')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='상세 출력')
    
    # W&B 설정 (기본 활성화)
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Weights & Biases 비활성화 (기본: 활성화)')
    parser.add_argument('--wandb_project', default='yolo26-construction',
                        help='W&B 프로젝트명')
    parser.add_argument('--wandb_entity', default=None,
                        help='W&B 엔티티명')
    
    args = parser.parse_args()
    
    # 실행 이름 자동 생성
    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"yolo26{args.model_size}_b{args.batch_size}_e{args.epochs}_{timestamp}"
    
    print("="*80)
    print("🚀 YOLO26 하이브리드 학습 시스템 (Python + CLI)")
    print("="*80)
    print(f"📋 실험 이름: {args.run_name}")
    print(f"🎯 모델: YOLO26{args.model_size.upper()}")
    print(f"📊 설정: Epochs={args.epochs}, Batch={args.batch_size}, Device={args.device or 'auto-detect'}")
    print(f"⚙️  옵티마이저: {args.optimizer} (lr={args.lr})")
    print("="*80)
    
    # 로그 디렉토리 설정 (프로젝트 루트 기준)
    log_dir = PROJECT_ROOT / 'training_logs' / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    
    print(f"📁 로그 저장: {log_file}")
    
    # 시스템 정보
    gpu_count = get_available_gpus()
    print(f"🖥️  시스템: {gpu_count} GPUs available")
    
    # 실험 설정 딕셔너리
    config = {
        "model_size": args.model_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "device": args.device or f"auto-detect ({gpu_count} GPUs)",
        "gpu_count": gpu_count,
        "project_root": str(PROJECT_ROOT),
        "mixup": args.mixup,
        "patience": args.patience,
        "amp": args.amp,
        "cos_lr": args.cos_lr
    }
    
    # W&B 초기화 (기본 활성화)
    wandb_run = None
    if not args.disable_wandb:
        print("📊 W&B 자동 연결 중...")
        wandb_run = setup_wandb(args.wandb_project, args.wandb_entity, config)
        if wandb_run:
            print(f"✅ W&B 대시보드: {wandb_run.url}")
        else:
            print("⚠️  W&B 연결 실패, 로컬 모드로 진행")
    else:
        print("📊 W&B 비활성화됨")
    
    # CLI 명령어 생성
    yolo_cmd = build_yolo_command(args)
    print(f"\n🔧 실행 명령어:")
    print(f"   {yolo_cmd}")
    print()
    
    # 설정 저장
    config_file = log_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "command": yolo_cmd,
            "args": vars(args),
            "config": config
        }, f, indent=2, ensure_ascii=False)
    
    # 모니터링 객체 생성
    monitor = TrainingMonitor(log_file, wandb_run)
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, monitor.signal_handler)
    signal.signal(signal.SIGTERM, monitor.signal_handler)
    
    try:
        print("🔥 하이브리드 학습 시작 (Python 모니터링 + CLI 실행)...")
        start_time = time.time()
        
        # 초기 로그 헤더 작성
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== YOLO26 하이브리드 학습 로그 ===\n")
            f.write(f"시작: {datetime.now()}\n")
            f.write(f"명령어: {yolo_cmd}\n")
            f.write("="*80 + "\n")
        
        # CLI 프로세스 시작
        process = subprocess.Popen(
            yolo_cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            preexec_fn=os.setsid  # 프로세스 그룹 생성
        )
        
        # 실시간 모니터링 루프
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                    
                if monitor.should_stop:
                    break
                    
                monitor.log_line(line)
                
                # 프로세스 상태 확인
                if process.poll() is not None:
                    break
            
            # 남은 출력 처리
            remaining = process.stdout.read()
            if remaining:
                monitor.log_line(remaining)
                
        except Exception as e:
            print(f"\n⚠️  모니터링 오류: {e}")
        
        # 프로세스 완료 대기
        return_code = process.wait()
        end_time = time.time()
        duration = end_time - start_time
        
        # 결과 처리
        if return_code == 0:
            print("\n" + "="*80)
            print("✅ 하이브리드 학습이 성공적으로 완료되었습니다!")
            print(f"⏱️  소요 시간: {duration/3600:.2f}시간")
            print(f"📁 결과 위치: {args.output_dir}/{args.run_name}")
            print(f"📋 로그 파일: {log_file}")
            if wandb_run:
                # 최종 학습 요약 정보
                final_summary = {
                    "training_duration_hours": duration/3600,
                    "total_epochs": monitor.last_epoch,
                    "final_fitness": monitor.fitness_history[-1] if monitor.fitness_history else 0,
                    "fitness_improvement": (monitor.fitness_history[-1] - monitor.fitness_history[0]) if len(monitor.fitness_history) > 1 else 0,
                    "training_completed": True
                }
                wandb.log(final_summary)
                
                # 성공적 완료 알림
                try:
                    wandb.alert(
                        title="Training Completed Successfully",
                        text=f"✅ Training completed in {duration/3600:.1f} hours\n" +
                             f"🏆 Best fitness: {monitor.metrics['best_fitness']:.4f} at epoch {monitor.metrics['best_epoch']}\n" +
                             f"📈 Total fitness improvement: {final_summary['fitness_improvement']:.4f}",
                        level=wandb.AlertLevel.INFO
                    )
                except:
                    pass
                
                wandb_run.finish()
                print(f"📊 W&B Dashboard: {wandb_run.url}")
            print("="*80)
        else:
            print(f"\n❌ 학습이 실패했습니다. (종료 코드: {return_code})")
            print(f"⏱️  소요 시간: {duration/60:.1f}분")
            print(f"📋 상세 로그: {log_file}")
            if wandb_run:
                failure_info = {
                    "training_failed": True,
                    "exit_code": return_code,
                    "training_duration_hours": duration/3600,
                    "epochs_completed": monitor.last_epoch,
                    "best_fitness_achieved": monitor.metrics.get('best_fitness', 0)
                }
                wandb.log(failure_info)
                
                # 실패 알림
                try:
                    wandb.alert(
                        title="Training Failed",
                        text=f"❌ Training failed after {duration/60:.1f} minutes\n" +
                             f"📊 Completed {monitor.last_epoch} epochs\n" +
                             f"🏆 Best fitness achieved: {monitor.metrics.get('best_fitness', 0):.4f}",
                        level=wandb.AlertLevel.ERROR
                    )
                except:
                    pass
                
                wandb_run.finish()
            sys.exit(return_code)
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass
        if wandb_run:
            wandb_run.finish()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        if wandb_run:
            wandb_run.log({"error": str(e)})
            wandb_run.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()