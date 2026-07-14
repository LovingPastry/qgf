#!/usr/bin/env bash
#
# run_compare.sh —— 一键跑「同一 env + 同一 task」下 QGF 与 README 中各 demo 的对比实验。
#
# 默认 8 张卡，每张卡同时最多 1 个训练任务，多任务自动排队（复用 run_multi_gpu.sh 调度器）。
#
# 对比方法：
#   Phase 1（从零训练，互相独立，排队并行）：
#     qgf（本仓库方法）、cfgrl、fql、edp、qam、dac、qsm_bc、robust_q、bc_iql（测试期方法的基座）
#   Phase 2（eval_only，从 bc_iql 基座恢复，Phase 1 完成后自动生成并运行）：
#     qgf_test_time_eval、qgf_jacobian_test_time_eval、qfql_test_time_eval、grad_step
#
# 用法（最简）：
#     OGBENCH_DATA_DIR=/path/to/ogbench/data bash run_compare.sh
#
# 常用可选环境变量：
#     GPUS=0,1,2,3,4,5,6,7   使用哪些卡（默认 0-7 共 8 张）
#     ENV=cube-triple        env 族：cube-triple / cube-quadruple / puzzle-4x4 / scene
#     TASK=1                 任务号
#     SEED=1                 随机种子
#     OFFLINE_STEPS=500000   训练步数
#     EVAL_EPISODES=30       评估回合数
#     SAVE_INTERVAL=100000   checkpoint 保存间隔（Phase 2 依赖此 checkpoint）
#     ONLY_PHASE=1|2         只跑某一个 phase（默认两个都跑）
#     SKIP_PHASE2=1          只跑 Phase 1（训练所有基座/训练期方法，不做测试期评估）
#     P1_ONLY="qgf,fql"      Phase 1 只跑这些方法（逗号分隔）
#     P2_ONLY="grad_step"    Phase 2 只跑这些方法
#     DRY_RUN=1              只打印将要执行的命令，不真正启动
#
# 日志：logs/compare_<ENV>-task<TASK>-seed<SEED>/{phase1,phase2}/<方法>.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------ #
# 参数
# ------------------------------------------------------------------ #
: "${OGBENCH_DATA_DIR:?请设置 OGBENCH_DATA_DIR（OGBench 100M 数据集根目录）}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
ENV="${ENV:-cube-triple}"
TASK="${TASK:-1}"
SEED="${SEED:-1}"
OFFLINE_STEPS="${OFFLINE_STEPS:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
ONLY_PHASE="${ONLY_PHASE:-}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"
P1_ONLY="${P1_ONLY:-}"
P2_ONLY="${P2_ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
POLL_SECONDS="${POLL_SECONDS:-5}"

PY="${PYTHON:-python}"

TAG="${ENV}-task${TASK}-seed${SEED}"
BASE_LOG_DIR="logs/compare_${TAG}"
CMD_DIR="${BASE_LOG_DIR}/cmds"
mkdir -p "$CMD_DIR"

echo "=================================================================="
echo " 方法对比实验"
echo "   场景/任务 : ${ENV}  task${TASK}  seed${SEED}"
echo "   GPU       : ${GPUS}  (每卡最多 1 个任务，自动排队)"
echo "   训练步数  : ${OFFLINE_STEPS}   评估回合: ${EVAL_EPISODES}"
echo "   日志目录  : ${BASE_LOG_DIR}"
echo "=================================================================="

# 复用现有的多卡排队调度器把一个命令文件跑完
run_phase() {
  local phase_name="$1" cmd_file="$2" log_dir="$3"
  echo
  echo ">>> 运行 ${phase_name}：命令文件 ${cmd_file}"
  GPUS="$GPUS" LOG_DIR="$log_dir" DRY_RUN="$DRY_RUN" POLL_SECONDS="$POLL_SECONDS" \
    bash run_multi_gpu.sh "$cmd_file"
}

# ------------------------------------------------------------------ #
# Phase 1：从零训练所有基座/训练期方法
# ------------------------------------------------------------------ #
do_phase1() {
  local cmd_file="${CMD_DIR}/phase1.txt"
  local extra=()
  [ -n "$P1_ONLY" ] && extra+=(--only "$P1_ONLY")
  "$PY" scripts/gen_compare_cmds.py \
    --phase 1 --env "$ENV" --task "$TASK" --seed "$SEED" \
    --offline_steps "$OFFLINE_STEPS" --eval_episodes "$EVAL_EPISODES" \
    --save_interval "$SAVE_INTERVAL" --ogbench_dir "$OGBENCH_DATA_DIR" \
    --out "$cmd_file" "${extra[@]}" || return 1
  run_phase "Phase 1 (训练)" "$cmd_file" "${BASE_LOG_DIR}/phase1"
}

# ------------------------------------------------------------------ #
# Phase 2：测试期方法，从 Phase 1 训好的 bc_iql 基座恢复
# ------------------------------------------------------------------ #
do_phase2() {
  local cmd_file="${CMD_DIR}/phase2.txt"
  local extra=()
  [ -n "$P2_ONLY" ] && extra+=(--only "$P2_ONLY")
  # 此处才生成 Phase 2 命令：需要 bc_iql checkpoint 已经存在
  if ! "$PY" scripts/gen_compare_cmds.py \
        --phase 2 --env "$ENV" --task "$TASK" --seed "$SEED" \
        --eval_episodes "$EVAL_EPISODES" --train_run_group bc_iql \
        --ogbench_dir "$OGBENCH_DATA_DIR" \
        --out "$cmd_file" "${extra[@]}"; then
    echo "!!! Phase 2 命令生成失败（多半是 bc_iql 基座 checkpoint 未就绪），跳过测试期评估。" >&2
    return 1
  fi
  run_phase "Phase 2 (测试期评估)" "$cmd_file" "${BASE_LOG_DIR}/phase2"
}

# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
rc=0
case "$ONLY_PHASE" in
  1)
    do_phase1 || rc=$?
    ;;
  2)
    do_phase2 || rc=$?
    ;;
  *)
    do_phase1 || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "!!! Phase 1 有任务失败（rc=$rc）。仍尝试用已生成的 bc_iql 基座跑 Phase 2。" >&2
    fi
    if [ "$SKIP_PHASE2" = "1" ]; then
      echo ">>> SKIP_PHASE2=1，跳过 Phase 2。"
    elif [ "$DRY_RUN" = "1" ]; then
      echo ">>> DRY_RUN=1：Phase 2 依赖 Phase 1 的真实 checkpoint，dry-run 下跳过其命令生成。"
    else
      do_phase2 || true
    fi
    ;;
esac

echo
echo "=================================================================="
echo " 全部结束。日志见 ${BASE_LOG_DIR}/ ；用 tensorboard --logdir exp/ 查看指标与视频。"
echo "=================================================================="
exit "$rc"
