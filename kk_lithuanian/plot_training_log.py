import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER_PREFIX_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)

TITLE_FONT_SIZE = 13
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def coerce_value(value: str):
    value = value.strip()
    lowered = value.lower()
    if lowered.startswith("nan"):
        return float("nan")
    if lowered.startswith("-inf"):
        return float("-inf")
    if lowered.startswith("inf"):
        return float("inf")

    match = NUMBER_PREFIX_RE.match(value)
    if not match:
        return value

    number_text = match.group(0)
    if re.search(r"[.eE]", number_text):
        return float(number_text)
    return int(number_text)


def parse_log(log_path: Path):
    rows = []
    clamp_metrics = {
        "critic/rewards/mean",
        "val-aux/kk_logic_lithuanian/reward/mean@1",
    }

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = strip_ansi(raw_line.strip())
            if not line:
                continue

            if " step:" not in line or " - " not in line:
                continue

            before, rest = line.split(" step:", 1)
            step_str, metrics_str = rest.split(" - ", 1)
            try:
                step = int(step_str.strip())
            except ValueError:
                continue

            row = {"step": step}
            for item in metrics_str.split(" - "):
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                key = key.strip()
                parsed_value = coerce_value(value)
                if key in clamp_metrics and isinstance(parsed_value, (int, float)):
                    if parsed_value > 1.8:
                        parsed_value = 1.8
                    elif parsed_value < -1.8:
                        parsed_value = -1.8
                row[key] = parsed_value

            rows.append(row)

    if not rows:
        raise RuntimeError("No step metrics found in the log file.")

    return pd.DataFrame(rows)


def normalize_series_0_1(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return series
    if min_val >= 0.0 and max_val <= 1.0:
        return series
    if min_val == max_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def plot_metric(ax, df, metric, title, is_val_metric=False, transform=None, y_label=None):
    if metric not in df.columns:
        ax.set_visible(False)
        return

    df = df.sort_values("step")
    x = df["step"]
    y = df[metric]
    if transform is not None:
        y = transform(y)
    if is_val_metric:
        ax.scatter(x, y, s=18)
    else:
        ax.plot(x, y)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("treniravimo žingsnis", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label or metric, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_response_lengths(ax, df):
    min_key = "response_length/min"
    max_key = "response_length/max"
    mean_key = "response_length/mean"

    if not any(key in df.columns for key in (min_key, max_key, mean_key)):
        ax.set_visible(False)
        return

    df = df.sort_values("step")
    x = df["step"]
    if min_key in df.columns:
        ax.plot(x, df[min_key], label="min")
    if mean_key in df.columns:
        ax.plot(x, df[mean_key], label="mean")
    if max_key in df.columns:
        ax.plot(x, df[max_key], label="max")

    ax.set_title("Sugeneruoto atsakymo ilgis", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("treniravimo žingsnis", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("tokenai", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LABEL_FONT_SIZE,
    )


def plot_response_length_clip_ratio(ax, df):
    metric = "response_length/clip_ratio"
    if metric not in df.columns:
        ax.set_visible(False)
        return

    df = df.sort_values("step")
    ax.plot(df["step"], df[metric])
    ax.set_title("Atsakymo ilgio ribojimo santykis", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("treniravimo žingsnis", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("santykis", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.3)


def hide_axes(axes):
    for ax in axes:
        ax.set_visible(False)


def main():
    parser = argparse.ArgumentParser(
        description="Parse verl training logs and plot key metrics."
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to the training log file")
    parser.add_argument(
        "--method",
        type=str,
        default="grpo",
        choices=["grpo", "dapo", "reinforce++"],
        help="Training method to enable method-specific plots",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory path (default: <log filename>/)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for parsed metrics",
    )
    args = parser.parse_args()

    df = parse_log(args.log)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)

    output_dir = args.out_dir or args.log.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)

    plot_metric(
        axes[0, 0],
        df,
        "critic/rewards/mean",
        "Vidutinis atlygis treniravimo metu",
        is_val_metric=False,
        y_label="atlygis",
    )
    plot_metric(
        axes[0, 1],
        df,
        "val-aux/kk_logic_lithuanian/reward/mean@1",
        "Vidutinis atlygis validavimo metu",
        is_val_metric=True,
        y_label="atlygis",
    )
    plot_metric(
        axes[0, 2],
        df,
        "actor/ppo_kl",
        "Kullbacko-Leiblerio (KL) nuokrypis",
        is_val_metric=False,
        y_label="KL",
    )
    plot_metric(
        axes[1, 1],
        df,
        "actor/entropy",
        "Entropija",
        is_val_metric=False,
        y_label="entropija",
    )
    plot_metric(
        axes[1, 0],
        df,
        "actor/grad_norm",
        "Gradientų norma",
        is_val_metric=False,
        y_label="grad. norma",
    )
    plot_metric(
        axes[0, 3],
        df,
        "perf/throughput",
        "Duomenų pralaidumas",
        is_val_metric=False,
        y_label="tokenai/sek.",
    )
    plot_response_lengths(axes[1, 2], df)
    plot_metric(
        axes[1, 3],
        df,
        "timing_s/step",
        "Žingsnio trukmė",
        is_val_metric=False,
        y_label="sekundės",
    )

    fig.suptitle("Mokymo proceso metrikos pagal treniravimo žingsnį")
    combined_path = output_dir / "training_metrics.png"
    fig.savefig(combined_path, dpi=160)
    print(f"Saved combined plot to {combined_path}")

    metric_specs = [
        (
            "critic/rewards/mean",
            "training_reward_mean",
            "Vidutinis atlygis treniravimo metu",
            "atlygis",
        ),
        (
            "val-aux/kk_logic_lithuanian/reward/mean@1",
            "val_reward_mean",
            "Vidutinis atlygis validavimo metu",
            "atlygis",
        ),
        (
            "actor/ppo_kl",
            "actor_ppo_kl",
            "Kullbacko-Leiblerio (KL) nuokrypis",
            "KL",
        ),
        ("actor/entropy", "actor_entropy", "Entropija", "entropija"),
        (
            "actor/grad_norm",
            "actor_grad_norm",
            "Gradientų norma",
            "grad. norma",
        ),
        (
            "perf/throughput",
            "throughput",
            "Duomenų pralaidumas",
            "tokenai/sek.",
        ),
        (
            "timing_s/step",
            "step_time",
            "Žingsnio trukmė",
            "sekundės",
        ),
    ]

    for metric, stem, title, y_label in metric_specs:
        fig_metric, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        plot_metric(ax, df, metric, title, is_val_metric=False, y_label=y_label)
        if ax.get_visible():
            out_path = output_dir / f"{stem}.png"
            fig_metric.savefig(out_path, dpi=160)
        plt.close(fig_metric)

    fig_resp, ax_resp = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    plot_response_lengths(ax_resp, df)
    if ax_resp.get_visible():
        fig_resp.savefig(output_dir / "response_length.png", dpi=160)
    plt.close(fig_resp)

    if args.method == "dapo":
        fig_clip, ax_clip = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        plot_response_length_clip_ratio(ax_clip, df)
        if ax_clip.get_visible():
            fig_clip.savefig(output_dir / "response_length_clip_ratio.png", dpi=160)
        plt.close(fig_clip)


if __name__ == "__main__":
    main()
