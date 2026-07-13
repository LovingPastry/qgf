import os
import subprocess
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
from PIL import Image, ImageEnhance
from tensorboardX import SummaryWriter


def _is_media(v):
    """Return True for non-scalar array-like values (e.g. videos) that cannot go in a CSV cell."""
    return isinstance(v, np.ndarray) and v.ndim > 0


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None

    def log(self, row, step):
        row["step"] = step
        if self.file is None:
            self.file = open(self.path, "w")
            if self.header is None:
                self.header = [k for k, v in row.items() if not _is_media(v)]
                self.file.write(",".join(self.header) + "\n")
            filtered_row = {k: v for k, v in row.items() if not _is_media(v)}
            self.file.write(
                ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
            )
        else:
            filtered_row = {k: v for k, v in row.items() if not _is_media(v)}
            self.file.write(
                ",".join([str(filtered_row.get(k, "")) for k in self.header]) + "\n"
            )
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def log_scalars(writer, metrics, step):
    """Write all scalar-valued entries of ``metrics`` to TensorBoard.

    Non-scalar values (arrays, videos, strings that are not numbers) are skipped so
    that only proper scalars land on the SCALARS dashboard.
    """
    for k, v in metrics.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        writer.add_scalar(k, fv, step)
    writer.flush()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ""
    exp_name += f"sd{seed:03d}_"
    if "SLURM_JOB_ID" in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if "SLURM_PROCID" in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if "." not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_tensorboard(
    log_dir,
    hyperparam_dict=None,
    disabled=False,
    log_code=False,
):
    """Set up a TensorBoard ``SummaryWriter`` for logging.

    Args:
        log_dir: Directory to write event files to. Point ``tensorboard --logdir`` at
            its parent to browse all runs.
        hyperparam_dict: Extra hyperparameters to record alongside the absl flags.
        disabled: If True (e.g. debug runs), write to a throwaway temp dir so the
            experiment directory stays clean.
        log_code: If True, additionally record the current git commit and diff as text.

    Returns:
        A ``tensorboardX.SummaryWriter``.
    """
    if disabled:
        log_dir = tempfile.mkdtemp()
    writer = SummaryWriter(logdir=log_dir)

    # Combine flag dict with hyperparameters and record them on the TEXT dashboard.
    config_dict = get_flag_dict()
    if hyperparam_dict is not None:
        config_dict.update(hyperparam_dict)
    config_md = "\n".join(
        f"- **{k}**: `{config_dict[k]}`" for k in sorted(config_dict, key=str)
    )
    writer.add_text("config", config_md, 0)

    if log_code:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            diff = subprocess.check_output(["git", "diff"], text=True)
        except Exception:
            sha, diff = "no-git", ""
        writer.add_text("git", f"commit: `{sha}`\n\n```\n{diff}\n```", 0)

    return writer


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[
            None,
        ]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_tensorboard_video(renders=None, n_cols=None):
    """Return a video tensor ready for ``SummaryWriter.add_video``.

    It takes a list of videos and reshapes them into a single grid video. The returned
    array has shape (1, t, c, h, w) and dtype uint8, which TensorBoard renders as an
    animation on the IMAGES dashboard (viewable in the browser).

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(
            renders[i],
            ((0, 0), (1, 1), (1, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    # add_video expects (N, T, C, H, W) uint8; add a singleton batch dimension.
    renders = renders.astype(np.uint8)[None]

    return renders
