import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_model(model, max_steps=5000):
    """
    Ejecuta un modelo Mesa hasta que termine o llegue a max_steps.
    Devuelve: df (t_exit), ts (tiempos), perc (%evacuado), metrics (dict)
    """
    steps = 0
    while model.running and steps < max_steps:
        model.step()
        steps += 1

    exit_times = getattr(model, "exit_times", [])
    df = pd.DataFrame({"t_exit": exit_times})
    if not df.empty:
        df.sort_values("t_exit", inplace=True)

    makespan = df["t_exit"].max() if not df.empty else np.nan
    p50 = df["t_exit"].quantile(0.5) if not df.empty else np.nan
    p90 = df["t_exit"].quantile(0.9) if not df.empty else np.nan

    ts = np.arange(0, steps + 1) * model.time_step
    evac_counts = []
    for s in range(steps + 1):
        t = s * model.time_step
        evac_counts.append(np.sum(df["t_exit"] <= t) if not df.empty else 0)
    perc = np.array(evac_counts) / max(len(model.schedule.agents) + len(exit_times), 1) * 100.0

    metrics = {
        "steps": steps,
        "time_step": model.time_step,
        "makespan": makespan,
        "p50": p50,
        "p90": p90,
        "evacuados": 0 if df.empty else int((~df["t_exit"].isna()).sum())
    }
    return df, ts, perc, metrics


def save_times(df, path_csv):
    df.to_csv(path_csv, index=False)


def save_metrics(metrics_dict, path_csv):
    import pandas as pd
    pd.DataFrame([metrics_dict]).to_csv(path_csv, index=False)


def plot_curva(ts, perc, title, out_png):
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")

    plt.figure()
    plt.plot(ts, perc)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("% evacuado")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_curvas_comparadas(series, title, out_png):
    """
    series: lista de tuplas (label, ts, perc)
    """
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")

    plt.figure()
    for label, ts, perc in series:
        plt.plot(ts, perc, label=label)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("% evacuado")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()