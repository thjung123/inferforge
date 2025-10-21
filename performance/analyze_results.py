import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

RESULTS_ROOT = Path("./perf_out")
OUTPUT_ROOT = Path("./perf_image")
CONFIG_PATH = Path("./../performance/config_matrix.json")


def parse_filename(filename: str):
    match = re.match(r".*_b(\d+)_c(\d+)_(.*?)\.csv", filename)
    if match:
        batch, conc, tag = match.groups()
        return int(batch), int(conc), tag
    return None, None, None


def load_results():
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"{RESULTS_ROOT} not found")
    all_rows = []
    for csv_path in RESULTS_ROOT.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            batch, conc, tag = parse_filename(csv_path.name)
            if batch is None:
                continue
            row = df.iloc[-1]
            throughput = row.get("Inferences/Second")
            latency = row.get(
                "p95 latency", row.get("p95 Latency (µs)", row.get("p95 Latency (ms)"))
            )
            gpu_util = None
            util_match = re.findall(r":([\d.]+);", open(csv_path).read())
            if util_match:
                try:
                    gpu_util = float(util_match[-3])
                except Exception as e:
                    gpu_util = None
                    print("Occur Exception:", e)
            if "no_dynamic" in tag:
                mode, delay = "no_dynamic", None
            elif "dynamic" in tag:
                mode = "dynamic"
                delay = re.findall(r"dynamic_(\d+)", tag)
                delay = int(delay[0]) if delay else None
            else:
                mode, delay = "unknown", None
            all_rows.append(
                {
                    "batch": batch,
                    "concurrency": conc,
                    "mode": mode,
                    "delay": delay,
                    "throughput": throughput,
                    "p95_latency": latency,
                    "gpu_util": gpu_util,
                    "tag": tag,
                    "csv_file": csv_path.name,
                }
            )
        except Exception as e:
            print(f"[SKIP] {csv_path.name} ({e})")
    df = pd.DataFrame(all_rows)
    df.sort_values(by=["batch", "concurrency", "delay"], inplace=True)
    print(f"[LOAD] {len(df)} CSV loaded.")
    return df


def plot_batch_scaling(df, out_dir, concurrency=1):
    subset = df[(df["mode"] == "dynamic") & (df["concurrency"] == 1)]
    if subset.empty:
        print("[WARN] No dynamic data for batch scaling (concurrency=1)")
        return

    subset = (
        subset.groupby("batch", as_index=False)[["throughput", "p95_latency"]]
        .mean()
        .sort_values("batch")
    )

    fig, ax1 = plt.subplots(figsize=(9, 6))
    color1 = "tab:blue"
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (infer/sec)", color=color1)
    ax1.plot(subset["batch"], subset["throughput"], "o-", color=color1)
    for _, row in subset.iterrows():
        ax1.text(
            row["batch"],
            row["throughput"] * 1.01,
            f"{int(row['throughput'])}",
            color=color1,
            fontsize=8,
            ha="center",
        )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("p95 Latency (µs, log scale)", color=color2)
    ax2.plot(subset["batch"], subset["p95_latency"], "s--", color=color2)
    for _, row in subset.iterrows():
        ax2.text(
            row["batch"],
            row["p95_latency"] * 1.1,
            f"{int(row['p95_latency']):,}",
            color=color2,
            fontsize=8,
            ha="center",
        )
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Batch Scaling (Dynamic, Concurrency=1)")
    fig.tight_layout()
    out_path = out_dir / "01_batch_scaling_dynamic_b1.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Batch scaling plot saved: {out_path}")


def plot_dynamic_vs_static(df, out_dir, batch=64, concurrency=1):
    base = df[
        (df["batch"] == batch)
        & (df["concurrency"] == concurrency)
        & (df["mode"] == "no_dynamic")
    ]
    dyn = df[
        (df["batch"] == batch)
        & (df["concurrency"] == concurrency)
        & (df["mode"] == "dynamic")
    ]
    if base.empty or dyn.empty:
        print("[WARN] Missing dynamic/static comparison data")
        return
    compare_df = pd.concat([base, dyn])
    compare_df["efficiency"] = compare_df["throughput"] / compare_df["p95_latency"]
    color_map = {"dynamic": "#2ca02c", "no_dynamic": "#9467bd"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x_labels = compare_df["mode"] + compare_df["delay"].fillna("").astype(str)

    axes[0].bar(
        x_labels,
        compare_df["throughput"],
        color=[color_map[m] for m in compare_df["mode"]],
    )
    axes[0].set_title("Throughput (infer/sec)")

    axes[1].bar(
        x_labels,
        compare_df["p95_latency"],
        color=[color_map[m] for m in compare_df["mode"]],
    )
    axes[1].set_title("p95 Latency (µs)")

    axes[2].bar(
        x_labels,
        compare_df["efficiency"],
        color=[color_map[m] for m in compare_df["mode"]],
    )
    axes[2].set_title("Efficiency (Throughput / Latency)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(ls="--", alpha=0.4)
    fig.suptitle(f"Dynamic vs Static — Batch={batch}, Concurrency={concurrency}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / f"02_dynamic_vs_static_b{batch}_c{concurrency}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[PLOT] Dynamic vs Static plot saved.")


def plot_concurrency_scaling(df, out_dir):
    subset = df[(df["mode"] == "dynamic") & (df["batch"] == 64)]
    if subset.empty:
        print("[WARN] No dynamic data for concurrency scaling (batch=64)")
        return

    subset = (
        subset.groupby("concurrency", as_index=False)["p95_latency"]
        .mean()
        .sort_values("concurrency")
    )
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        subset["concurrency"],
        subset["p95_latency"],
        "o-",
        color="orange",
        label="Batch 64",
    )
    for _, row in subset.iterrows():
        ax.text(
            row["concurrency"],
            row["p95_latency"] * 1.05,
            f"{int(row['p95_latency']):,}",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Concurrency Level")
    ax.set_ylabel("p95 Latency (µs)")
    ax.set_title("Concurrency vs Latency (Dynamic, Batch=64)")
    ax.legend(title="Batch Size")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()

    out_path = out_dir / "03_concurrency_vs_latency_dynamic_b64.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Concurrency scaling plot saved: {out_path}")


def plot_gpu_util_vs_throughput(df, out_dir):
    valid = df.dropna(subset=["gpu_util"])
    if valid.empty:
        print("[WARN] No GPU utilization data found")
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(valid["gpu_util"], valid["throughput"], alpha=0.7, color="#1f77b4")
    plt.xlabel("Avg GPU Utilization (%)")
    plt.ylabel("Throughput (infer/sec)")
    plt.title("GPU Utilization vs Throughput")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / "04_gpu_util_vs_throughput.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[PLOT] GPU util vs throughput plot saved.")


def export_summary(df, out_dir):
    summary_cols = [
        "csv_file",
        "batch",
        "concurrency",
        "mode",
        "delay",
        "throughput",
        "p95_latency",
        "gpu_util",
    ]
    summary_path = out_dir / "summary.csv"
    df[summary_cols].to_csv(summary_path, index=False)
    print(f"[SAVE] Summary CSV exported → {summary_path}")


def main():
    df = load_results()
    OUTPUT_ROOT.mkdir(exist_ok=True)

    export_summary(df, OUTPUT_ROOT)
    plot_batch_scaling(df, OUTPUT_ROOT, concurrency=1)
    plot_dynamic_vs_static(df, OUTPUT_ROOT, batch=64, concurrency=1)
    plot_concurrency_scaling(df, OUTPUT_ROOT)
    plot_gpu_util_vs_throughput(df, OUTPUT_ROOT)

    print("\n[DONE] Core plots saved under", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
