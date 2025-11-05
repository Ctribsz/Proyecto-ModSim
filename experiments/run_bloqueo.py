import argparse, os
import pandas as pd
from src.scenarios import bloqueo
from src.metrics import save_times, save_metrics, plot_curva

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=300)
    p.add_argument("--width", type=int, default=25)
    p.add_argument("--height", type=int, default=25)
    p.add_argument("--num_exits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t_bloqueo", type=float, default=60.0, help="segundos para bloquear")
    p.add_argument("--exit_index", type=int, default=0, help="índice de salida a bloquear (0..n-1)")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df, ts, perc, metrics = bloqueo(
        N=args.agents, width=args.width, height=args.height,
        num_exits=args.num_exits, seed=args.seed,
        t_bloqueo=args.t_bloqueo, exit_index=args.exit_index,
        max_steps=args.max_steps
    )

    base = f"bloqueo_e{args.exit_index}_t{int(args.t_bloqueo)}"
    save_times(df, os.path.join(args.outdir, f"{base}_times.csv"))
    save_metrics({
        **metrics,
        "N": args.agents, "width": args.width, "height": args.height,
        "num_exits": args.num_exits, "seed": args.seed
    }, os.path.join(args.outdir, f"{base}_metrics.csv"))
    plot_curva(ts, perc, f"Bloqueo: salida {args.exit_index} @ {args.t_bloqueo}s", os.path.join(args.outdir, f"{base}_curva.png"))

    print("✅ Listo.")

if __name__ == "__main__":
    main()