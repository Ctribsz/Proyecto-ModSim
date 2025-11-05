import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.scenarios import anchos
from src.metrics import plot_curvas_comparadas

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=300)
    p.add_argument("--width", type=int, default=25)
    p.add_argument("--height", type=int, default=25)
    p.add_argument("--anchos", nargs="+", type=int, default=[1,2,3], help="proxy: número de salidas")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    resultados = anchos(
        N=args.agents, width=args.width, height=args.height,
        lista_anchos=args.anchos, seed=args.seed, max_steps=args.max_steps
    )

    # Guardar resumen + curvas
    series = []
    rows = []
    for a, df, ts, perc, met in resultados:
        series.append((f"ancho{a}", ts, perc))
        rows.append({**met, "N": args.agents, "width": args.width, "height": args.height, "seed": args.seed})

        # export por ancho
        pref = os.path.join(args.outdir, f"anchos_{a}")
        df.to_csv(pref + "_times.csv", index=False)

    # CSV de métricas
    resumo = pd.DataFrame(rows)
    resumo.to_csv(os.path.join(args.outdir, "anchos_metrics.csv"), index=False)

    # Curvas comparadas
    plot_curvas_comparadas(series, "Curvas de evacuación por 'ancho' (proxy)", os.path.join(args.outdir, "anchos_curvas.png"))

    # Gráfica ancho vs makespan
    try:
        import matplotlib
        try:
            matplotlib.get_backend()
        except Exception:
            matplotlib.use("Agg")
        plt.figure()
        xs = [r["ancho_proxy"] for r in rows]
        ys = [r["makespan"] for r in rows]
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Ancho (proxy = número de salidas)")
        plt.ylabel("Makespan (s)")
        plt.title("Efecto del 'ancho' en el tiempo total")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.outdir, "anchos_vs_makespan.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    print("✅ Listo.")

if __name__ == "__main__":
    main()
