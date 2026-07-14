#!/usr/bin/env bash
#
# run_multi_gpu.sh — 在单机多卡上并行跑 QGF 复现实验（每张卡最多 1 个实验）。
#
# 用哪几张卡由环境变量 GPUS 指定（逗号分隔），例如：
#     GPUS=0,1,2,3 OGBENCH_DATA_DIR=/path/to/ogbench/data bash run_multi_gpu.sh
#
# 调度逻辑：把所有待跑实验排成队列，只要有空闲 GPU 就派发下一个，
# 每张 GPU 同时只跑 1 个进程（靠 CUDA_VISIBLE_DEVICES 绑定单卡）。
#
# 实验内容默认按 README「QGF (our method)」的训练命令生成，扫描维度可用环境变量覆盖：
#     ENVS   —— 环境族（默认 "cube-triple"），可选：cube-triple cube-quadruple puzzle-4x4 scene
#     TASKS  —— 任务号（默认 "1 2 3 4 5"）
#     SEEDS  —— 随机种子（默认 "1"）
#
# 也可以自带命令列表：把每行一条完整命令（python main.py ...，不用写 MUJOCO_GL 前缀）
# 放到一个文件里，作为第一个参数传入，本脚本只负责把它们分发到各张卡：
#     GPUS=0,1 bash run_multi_gpu.sh my_commands.txt
#
# 其它可选环境变量：
#     DRY_RUN=1          只打印将要执行的命令，不真正启动
#     LOG_DIR=...        日志目录（默认 logs/<RUN_GROUP>）
#     OFFLINE_STEPS=...  训练步数（默认 500000）
#     RUN_GROUP=.. run group（默认 qgf_repro）
#     GUIDANCE_WEIGHTS=..评估用的 guidance 权重列表
#     POLL_SECONDS=...   轮询空闲 GPU 的间隔秒数（默认 5）

set -uo pipefail

# 先按调用时的工作目录把命令文件解析成绝对路径，再切到仓库根目录
CMD_FILE="${1:-}"
if [ -n "$CMD_FILE" ] && [ "${CMD_FILE#/}" = "$CMD_FILE" ]; then
  CMD_FILE="$PWD/$CMD_FILE"   # 相对路径 -> 绝对路径
fi

# 切到脚本所在目录（即仓库根目录），保证能找到 main.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------ #
# 参数与环境变量
# ------------------------------------------------------------------ #
: "${GPUS:?请设置 GPUS，例如 GPUS=0,1,2,3}"

ENVS="${ENVS:-cube-triple}"
TASKS="${TASKS:-1 2 3 4 5}"
SEEDS="${SEEDS:-1}"
OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
GUIDANCE_WEIGHTS="${GUIDANCE_WEIGHTS:-0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12}"
RUN_GROUP="${RUN_GROUP:-qgf_repro}"
LOG_DIR="${LOG_DIR:-logs/${RUN_GROUP}}"
POLL_SECONDS="${POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

# 解析 GPU 列表
IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -eq 0 ]; then
  echo "错误：GPUS 为空。" >&2
  exit 1
fi

# ------------------------------------------------------------------ #
# 构建实验队列： CMDS[i] 是完整命令， NAMES[i] 用于日志文件名
# ------------------------------------------------------------------ #
declare -a CMDS=()
declare -a NAMES=()

# 环境族 -> 100M 数据集子目录名。例：cube-triple -> cube-triple-play-100m-v0
dataset_dir_name() {
  echo "${1}-play-100m-v0"
}

build_default_sweep() {
  : "${OGBENCH_DATA_DIR:?请设置 OGBENCH_DATA_DIR（OGBench 100M 数据集根目录）}"
  local env task seed env_name ds cmd name
  for env in $ENVS; do
    ds="$(dataset_dir_name "$env")"
    for task in $TASKS; do
      env_name="${env}-play-singletask-task${task}-v0"
      for seed in $SEEDS; do
        # 对应 README「QGF (our method)」的训练命令；tuple 值用 \" 保留引号，
        # 经 bash -c 重新解析后 (1024,...) 会作为字面串传给 absl。
        cmd="python main.py \
--agent=agents/qgf.py \
--agent.denoised_action_approx=one_euler_step_approx \
--agent.apply_jacobian=False \
--agent.action_chunking=True \
--agent.horizon_length=5 \
--agent.batch_size=1024 \
--agent.value_network_kwargs.hidden_dims=\"(1024,1024,1024,1024)\" \
--agent.actor_hidden_dims=\"(1024,1024,1024,1024)\" \
--agent.discount=0.999 \
--env_name=${env_name} \
--ogbench_dataset_dir=${OGBENCH_DATA_DIR}/${ds}/ \
--offline_steps=${OFFLINE_STEPS} \
--seed=${seed} \
--run_group=${RUN_GROUP} \
--guidance_weights=${GUIDANCE_WEIGHTS}"
        name="${env}-task${task}-seed${seed}"
        CMDS+=("$cmd")
        NAMES+=("$name")
      done
    done
  done
}

build_from_file() {
  local line idx=0 pending_name=""
  while IFS= read -r line || [ -n "$line" ]; do
    # 以 # 开头的注释行：若形如「# name」，记为下一条命令的作业名；否则忽略
    if [[ "$line" =~ ^[[:space:]]*#[[:space:]]*(.+)$ ]]; then
      pending_name="${BASH_REMATCH[1]}"
      pending_name="${pending_name%%[[:space:]]}"   # 去尾部空白
      continue
    fi
    [[ -z "${line//[[:space:]]/}" ]] && continue
    idx=$((idx + 1))
    CMDS+=("$line")
    if [ -n "$pending_name" ]; then
      NAMES+=("$pending_name")
    else
      NAMES+=("$(printf 'run_%03d' "$idx")")
    fi
    pending_name=""
  done < "$CMD_FILE"
}

if [ -n "$CMD_FILE" ]; then
  if [ ! -f "$CMD_FILE" ]; then
    echo "错误：命令文件不存在：$CMD_FILE" >&2
    exit 1
  fi
  build_from_file
else
  build_default_sweep
fi

N=${#CMDS[@]}
if [ "$N" -eq 0 ]; then
  echo "没有可运行的实验，退出。" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "=================================================================="
echo " 待运行实验数 : $N"
echo " 使用的 GPU   : ${GPU_ARR[*]}  (每卡最多 1 个实验)"
echo " 日志目录     : $LOG_DIR"
[ -n "$CMD_FILE" ] && echo " 命令来源     : $CMD_FILE" || echo " 命令来源     : 默认 QGF 训练扫描 (ENVS=$ENVS TASKS=$TASKS SEEDS=$SEEDS)"
echo "=================================================================="

if [ "$DRY_RUN" = "1" ]; then
  for ((i = 0; i < N; i++)); do
    echo "--- [${NAMES[$i]}] ---"
    echo "MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> ${CMDS[$i]}"
    echo
  done
  echo "(DRY_RUN=1，未真正启动任何实验)"
  exit 0
fi

# ------------------------------------------------------------------ #
# 调度器
# ------------------------------------------------------------------ #
declare -A GPU_PID    # gpu -> 当前运行进程 pid（空表示空闲）
declare -A GPU_JOB    # gpu -> 当前运行的实验名（用于日志汇报）
N_OK=0
N_FAIL=0

# Ctrl-C / TERM 时杀掉所有在跑的子进程
cleanup() {
  echo
  echo "收到中断信号，正在停止所有实验..."
  for g in "${GPU_ARR[@]}"; do
    local pid="${GPU_PID[$g]:-}"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  exit 130
}
trap cleanup INT TERM

# 回收某张 GPU 上已结束的进程，统计成败
reap() {
  local g="$1" pid="${GPU_PID[$g]:-}"
  [ -n "$pid" ] || return 0
  if wait "$pid" 2>/dev/null; then
    echo "[完成] gpu $g  ${GPU_JOB[$g]}  [OK]"
    N_OK=$((N_OK + 1))
  else
    echo "[完成] gpu $g  ${GPU_JOB[$g]}  [FAIL] 见 $LOG_DIR/${GPU_JOB[$g]}.log"
    N_FAIL=$((N_FAIL + 1))
  fi
  GPU_PID[$g]=""
}

# 阻塞直到有空闲 GPU，结果写入全局变量 FREE_GPU
FREE_GPU=""
find_free_gpu() {
  FREE_GPU=""
  while [ -z "$FREE_GPU" ]; do
    for g in "${GPU_ARR[@]}"; do
      local pid="${GPU_PID[$g]:-}"
      if [ -z "$pid" ]; then
        FREE_GPU="$g"; return
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        reap "$g"          # 进程已结束，回收后即空闲
        FREE_GPU="$g"; return
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

launch() {
  local g="$1" name="$2" cmd="$3"
  local log="$LOG_DIR/${name}.log"
  echo "[启动] gpu $g  $name  -> $log"
  CUDA_VISIBLE_DEVICES="$g" MUJOCO_GL=egl bash -c "$cmd" > "$log" 2>&1 &
  GPU_PID[$g]="$!"
  GPU_JOB[$g]="$name"
}

# 逐个派发
for ((i = 0; i < N; i++)); do
  find_free_gpu
  launch "$FREE_GPU" "${NAMES[$i]}" "${CMDS[$i]}"
done

echo "全部实验已派发，等待剩余任务结束..."
for g in "${GPU_ARR[@]}"; do
  reap "$g"
done

echo "=================================================================="
echo " 结束： 成功 $N_OK  失败 $N_FAIL  （共 $N）"
echo "=================================================================="
[ "$N_FAIL" -eq 0 ]
