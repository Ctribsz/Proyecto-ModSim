import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import EvacuationModel

def run_once(N=200, width=20, height=20, num_exits=2, seed=42, max_steps=5000):
    model = EvacuationModel(width=width, height=height, N=N, num_exits=num_exits, seed=seed)

    steps = 0
    # Avanzar hasta que todos evacuen o se alcance max_steps
    while model.running and steps < max_steps:
        model.step()
        steps += 1

    # Construir DataFrame de tiempos individuales (si existen)
    exit_times = getattr(model, "exit_times", [])
    df = pd.DataFrame({"t_exit": exit_times})
    df.sort_values("t_exit", inplace=True)

    makespan = df["t_exit"].max() if not df.empty else np.nan
    p50 = df["t_exit"].quantile(0.5) if not df.empty else np.nan
    p90 = df["t_exit"].quantile(0.9) if not df.empty else np.nan

    # Serie % evacuado vs tiempo (por step)
    # Cada step vale model.time_step segundos
    ts = np.arange(0, steps + 1) * model.time_step
    evac_counts = []
    for s in range(steps + 1):
        t = s * model.time_step
        evac_counts.append(np.sum(df["t_exit"] <= t) if not df.empty else 0)
    perc = np.array(evac_counts) / max(N, 1) * 100.0

    metrics = {
        "N": N,
        "width": width,
        "height": height,
        "num_exits": num_exits,
        "seed": seed,
        "steps": steps,
        "time_step": model.time_step,
        "makespan": makespan,
        "p50": p50,
        "p90": p90,
        "evacuados": int(np.sum(~df["t_exit"].isna())) if not df.empty else 0
    }

    return df, ts, perc, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--num_exits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df, ts, perc, metrics = run_once(
        N=args.agents,
        width=args.width,
        height=args.height,
        num_exits=args.num_exits,
        seed=args.seed,
        max_steps=args.max_steps
    )

    # Guardar CSV de tiempos
    csv_path = os.path.join(args.outdir, "baseline_times.csv")
    df.to_csv(csv_path, index=False)

    # Guardar métricas
    met_path = os.path.join(args.outdir, "baseline_metrics.csv")
    pd.DataFrame([metrics]).to_csv(met_path, index=False)

    # Gráfica % evacuado vs tiempo
    plt.figure()
    plt.plot(ts, perc)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("% evacuado")
    plt.title("Curva de evacuación - Baseline")
    png_path = os.path.join(args.outdir, "curva_baseline.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")

    print(f"✅ Guardado: {csv_path}")
    print(f"✅ Guardado: {met_path}")
    print(f"✅ Guardado: {png_path}")
    print("✅ Listo.")

if __name__ == "__main__":
    main()
