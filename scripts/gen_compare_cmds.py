#!/usr/bin/env python3
"""为「同一 env + 同一 task」的方法对比实验生成 main.py 命令列表。

对比集合（与 README / scripts/exp_*.py 完全一致的超参）：

  Phase 1 —— 从零训练，彼此无依赖，可并行排队：
    qgf       : 本仓库方法（README「QGF (our method)」）
    cfgrl fql edp qam dac qsm_bc robust_q : 训练期基线
    bc_iql    : 测试期方法共享的 BC+IQL 基座（Phase 2 依赖它）

  Phase 2 —— eval_only，从 bc_iql 基座 checkpoint 恢复：
    qgf_test_time_eval qgf_jacobian_test_time_eval qfql_test_time_eval grad_step

每行输出一条完整命令（`python main.py ...`，不含 MUJOCO_GL / CUDA_VISIBLE_DEVICES 前缀），
交给 run_multi_gpu.sh 分发到各张卡。

用法：
    python scripts/gen_compare_cmds.py --phase 1 --env cube-triple --task 1 --seed 1 \
        --offline_steps 500000 --out phase1.txt
    python scripts/gen_compare_cmds.py --phase 2 --env cube-triple --task 1 --seed 1 \
        --train_run_group bc_iql --out phase2.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import _flags_from_args  # 复用与 sbatch 相同的 flag 引号处理

# ------------------------------------------------------------------ #
# env 相关工具
# ------------------------------------------------------------------ #
def env_dir_name(env_name: str) -> str:
    """cube-triple-play-singletask-task1-v0 -> cube-triple-play-100m-v0"""
    splits = env_name.split("-")
    if "singletask" not in splits:
        raise ValueError(f"Expected singletask env id, got {env_name!r}")
    pos = splits.index("singletask")
    prefix = "-".join(splits[:pos])
    ver = splits[-1]
    return f"{prefix}-100m-{ver}"


def env_short(env_name: str) -> str:
    return re.sub(r"-(singletask|v0|v1|v2)", "", env_name)


# 公共大网络 + action chunking 配置（所有方法共享）。value 维度的 flag 名因 agent 而异。
COMMON = {
    "agent.batch_size": 1024,
    "agent.actor_hidden_dims": "(1024,1024,1024,1024)",
    "agent.discount": 0.999,
    "agent.action_chunking": True,
    "agent.horizon_length": 5,
}
VALUE_NET = {"agent.value_network_kwargs.hidden_dims": "(1024,1024,1024,1024)"}
VALUE_FLAT = {"agent.value_hidden_dims": "(1024,1024,1024,1024)"}

# 部分方法的 per-env 调优超参（取自各 scripts/exp_*.py）
FQL_ALPHA = {"cube-triple": 1000.0, "cube-quadruple": 1000.0, "puzzle-4x4": 1000.0, "scene": 300.0}
EDP_BC_WEIGHT = {"cube-triple": 100.0, "cube-quadruple": 300.0, "puzzle-4x4": 300.0, "scene": 30.0}
DAC_ALPHA = {"cube-triple": 100.0, "cube-quadruple": 100.0, "puzzle-4x4": 300.0, "scene": 100.0}
GRAD_STEP = {  # (step_size, num_steps)
    "cube-triple": (0.01, 3), "cube-quadruple": (0.01, 3),
    "puzzle-4x4": (0.01, 3), "scene": (0.01, 5),
}

QGF_GUIDANCE = "0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12"


# ------------------------------------------------------------------ #
# Phase 1：从零训练的方法（run_group 用方法名，与 exp_*.py 对齐）
# ------------------------------------------------------------------ #
def phase1_methods(env_type, task, seed, offline_steps, eval_episodes, save_interval, ogbench_dir):
    env_name = f"{env_type}-play-singletask-task{task}-v0"
    ds = f"{ogbench_dir}/{env_dir_name(env_name)}/"
    base = {
        "env_name": env_name,
        "seed": seed,
        "online_steps": 0,
        "offline_steps": offline_steps,
        "eval_episodes": eval_episodes,
        "save_interval": save_interval,
        "ogbench_dataset_dir": ds,
    }

    methods = {}

    # QGF —— 本仓库方法（README「QGF (our method)」）
    methods["qgf"] = {
        **base, "run_group": "qgf", "agent": "agents/qgf.py",
        "agent.denoised_action_approx": "one_euler_step_approx",
        "agent.apply_jacobian": False,
        **COMMON, **VALUE_NET,
        "guidance_weights": QGF_GUIDANCE,
    }

    # CFGRL
    methods["cfgrl"] = {
        **base, "run_group": "cfgrl", "agent": "agents/cfgrl.py",
        "agent.denoise_steps": 10, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET,
        "guidance_weights": "1.0,2.0,3.0,5.0,10.0,15.0",
    }

    # FQL
    methods["fql"] = {
        **base, "run_group": "fql", "agent": "agents/fql.py",
        "agent.critic_loss_type": "iql", "agent.alpha": FQL_ALPHA[env_type],
        "agent.expectile": 0.9,
        **COMMON, **VALUE_NET,
    }

    # EDP
    methods["edp"] = {
        **base, "run_group": "edp", "agent": "agents/edp.py",
        "agent.denoise_steps": 10, "agent.bc_weight": EDP_BC_WEIGHT[env_type],
        "agent.expectile": 0.9,
        **COMMON, **VALUE_NET,
    }

    # QAM （注意用 value_hidden_dims）
    methods["qam"] = {
        **base, "run_group": "qam", "agent": "agents/qam.py",
        "agent.critic_loss_type": "iql", "agent.num_qs": 2, "agent.inv_temp": 0.1,
        "agent.expectile": 0.9,
        **COMMON, **VALUE_FLAT,
    }

    # DAC （dcgql, actor_loss_type=dac，用 value_hidden_dims）
    methods["dac"] = {
        **base, "run_group": "dac", "agent": "agents/dcgql.py",
        "agent.actor_loss_type": "dac", "agent.critic_loss_type": "iql",
        "agent.num_qs": 2, "agent.rho": 0.0, "agent.alpha": DAC_ALPHA[env_type],
        "agent.actor_cond_hidden_dims": "(32,32)", "agent.expectile": 0.9,
        **COMMON, **VALUE_FLAT,
    }

    # QSM-BC （dcgql, actor_loss_type=qsm）
    methods["qsm_bc"] = {
        **base, "run_group": "qsm_bc", "agent": "agents/dcgql.py",
        "agent.actor_loss_type": "qsm", "agent.critic_loss_type": "iql",
        "agent.num_qs": 2, "agent.rho": 0.0, "agent.inv_temp": 0.1, "agent.alpha": 10.0,
        "agent.actor_cond_hidden_dims": "(32,32)", "agent.expectile": 0.9,
        **COMMON, **VALUE_FLAT,
    }

    # Robust-Q （训练期方法，自带 guidance 评估）
    methods["robust_q"] = {
        **base, "run_group": "robust_q", "agent": "agents/robust_q.py",
        "agent.denoise_steps": 10, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET,
        "guidance_weights": QGF_GUIDANCE,
    }

    # BC + IQL 基座（Phase 2 依赖），agent 用 qgf.py，guidance_weights=0.0 即无引导基线
    methods["bc_iql"] = {
        **base, "run_group": "bc_iql", "agent": "agents/qgf.py",
        "agent.denoise_steps": 10, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET,
        "guidance_weights": "0.0",
    }

    return env_name, methods


# ------------------------------------------------------------------ #
# Phase 2：eval_only，从 bc_iql 基座恢复
# ------------------------------------------------------------------ #
def find_base_checkpoint(env_name, seed, train_run_group, save_dir="exp", tb_project="qgf"):
    """定位 bc_iql 基座的 run 目录，并返回 (run_dir, restore_epoch)。

    restore_epoch 取该目录下实际存在的最大 params_<step>.pkl，稳健对付非默认 offline_steps。
    找不到返回 (None, None)。
    """
    pattern = os.path.join(
        save_dir, tb_project, train_run_group,
        f"{train_run_group}_qgf_{env_short(env_name)}_seed{seed:02d}_*",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None, None
    run_dir = matches[0]
    ckpts = glob.glob(os.path.join(run_dir, "params_*.pkl"))
    steps = []
    for c in ckpts:
        m = re.search(r"params_(\d+)\.pkl$", c)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        return None, None
    return run_dir, max(steps)


def phase2_methods(env_type, task, seed, eval_episodes, train_run_group, ogbench_dir):
    env_name = f"{env_type}-play-singletask-task{task}-v0"
    ds = f"{ogbench_dir}/{env_dir_name(env_name)}/"
    run_dir, restore_epoch = find_base_checkpoint(env_name, seed, train_run_group)
    if run_dir is None:
        return env_name, {}, None

    base = {
        "env_name": env_name,
        "seed": seed,
        "offline_steps": 0,
        "online_steps": 0,
        "eval_episodes": eval_episodes,
        "eval_only": True,
        "restore_path": run_dir,
        "restore_epoch": restore_epoch,
        "ogbench_dataset_dir": ds,
    }
    gw = QGF_GUIDANCE

    methods = {}

    methods["qgf_test_time_eval"] = {
        **base, "run_group": "qgf_test_time_eval", "agent": "agents/qgf.py",
        "agent.denoise_steps": 10, "agent.denoised_action_approx": "one_euler_step_approx",
        "agent.apply_jacobian": False, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET, "guidance_weights": gw,
    }

    methods["qgf_jacobian_test_time_eval"] = {
        **base, "run_group": "qgf_jacobian_test_time_eval", "agent": "agents/qgf.py",
        "agent.denoise_steps": 10, "agent.denoised_action_approx": "one_euler_step_approx",
        "agent.apply_jacobian": True, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET, "guidance_weights": gw,
    }

    methods["qfql_test_time_eval"] = {
        **base, "run_group": "qfql_test_time_eval", "agent": "agents/qgf.py",
        "agent.denoise_steps": 10, "agent.denoised_action_approx": "noisy",
        "agent.apply_jacobian": False, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET, "guidance_weights": gw,
    }

    step_size, steps = GRAD_STEP[env_type]
    methods["grad_step"] = {
        **base, "run_group": "grad_step", "agent": "agents/grad_step.py",
        "agent.qgrad_step_size": step_size, "agent.qgrad_steps": steps,
        "agent.denoise_steps": 10, "agent.expectile": 0.9,
        **COMMON, **VALUE_NET,
    }

    return env_name, methods, run_dir


# ------------------------------------------------------------------ #
def build_command(kwargs: dict) -> str:
    return " ".join(["python main.py", *_flags_from_args(kwargs)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", type=int, choices=[1, 2], required=True)
    p.add_argument("--env", required=True, help="env 族，如 cube-triple / cube-quadruple / puzzle-4x4 / scene")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--offline_steps", type=int, default=500_000)
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--save_interval", type=int, default=100_000)
    p.add_argument("--train_run_group", default="bc_iql", help="Phase 2 恢复所用的基座 run_group")
    p.add_argument("--only", default="", help="逗号分隔，仅生成这些方法（默认全部）")
    p.add_argument("--ogbench_dir", default=os.environ.get("OGBENCH_DATA_DIR", ""))
    p.add_argument("--out", required=True, help="输出命令文件路径")
    args = p.parse_args()

    if not args.ogbench_dir:
        sys.exit("错误：未提供 --ogbench_dir，也未设置 OGBENCH_DATA_DIR。")

    only = {m.strip() for m in args.only.split(",") if m.strip()}

    if args.phase == 1:
        env_name, methods = phase1_methods(
            args.env, args.task, args.seed, args.offline_steps,
            args.eval_episodes, args.save_interval, args.ogbench_dir,
        )
    else:
        env_name, methods, run_dir = phase2_methods(
            args.env, args.task, args.seed, args.eval_episodes,
            args.train_run_group, args.ogbench_dir,
        )
        if not methods:
            sys.exit(
                f"错误：找不到 {args.train_run_group} 基座 checkpoint（env={env_name} "
                f"seed={args.seed}）。请先完成 Phase 1（bc_iql 训练）。"
            )
        print(f"[phase2] 使用基座 checkpoint: {run_dir}", file=sys.stderr)

    if only:
        methods = {k: v for k, v in methods.items() if k in only}
        missing = only - set(methods)
        if missing:
            print(f"[warn] --only 里有未知/不适用于该 phase 的方法: {sorted(missing)}", file=sys.stderr)

    lines = []
    for name, kwargs in methods.items():
        lines.append(f"# {name}")
        lines.append(build_command(kwargs))

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[phase{args.phase}] env={env_name} 生成 {len(methods)} 条命令 -> {args.out}", file=sys.stderr)
    print(f"[phase{args.phase}] 方法: {', '.join(methods)}", file=sys.stderr)


if __name__ == "__main__":
    main()
