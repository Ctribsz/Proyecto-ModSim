import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.agents import PersonAgent

def run_model(model, max_steps=5000):
    """
    Ejecuta un modelo Mesa hasta que termine o llegue a max_steps.
    Devuelve: df (t_exit), ts (tiempos), perc (%evacuado), metrics (dict)
    """
    steps = 0
    while model.running and steps < max_steps:
        model.step()
        steps += 1

    # --- Obtener datos de evacuación ---
    exit_events = getattr(model, "exit_events", [])
    if exit_events:
        df = pd.DataFrame(exit_events)
        df = df.sort_values("t_exit").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["id", "t_exit"])

    makespan = df["t_exit"].max() if not df.empty else np.nan
    p50 = df["t_exit"].quantile(0.5) if not df.empty else np.nan
    p90 = df["t_exit"].quantile(0.9) if not df.empty else np.nan

    ts = np.arange(0, steps + 1) * model.time_step
    evac_counts = []
    for s in range(steps + 1):
        t = s * model.time_step
        evac_counts.append(np.sum(df["t_exit"] <= t) if not df.empty else 0)
    
    # Calcular porcentaje usando el número inicial de personas
    initial_population = len(model.person_data) if hasattr(model, "person_data") else max(1, model.N)
    perc = np.array(evac_counts) / initial_population * 100.0

    metrics = {
        "steps": steps,
        "time_step": model.time_step,
        "makespan": makespan,
        "p50": p50,
        "p90": p90,
        "evacuados": len(df) if not df.empty else 0,
        "total_agentes": initial_population
    }

    # --- Métricas adicionales: reelecciones y throughput ---
    person_agents = [a for a in model.schedule.agents if isinstance(a, PersonAgent)]
    N = len(person_agents)
    
    if N > 0:
        reelecciones_totales = sum(getattr(a, "reelecciones", 0) for a in person_agents)
        metrics["reelecciones_totales"] = reelecciones_totales
        metrics["reelecciones_promedio"] = reelecciones_totales / N

        # Throughput por salida
        for i, ex in enumerate(model.exits):
            key = f"throughput_exit_{i}"
            t = metrics["makespan"] if not np.isnan(metrics["makespan"]) else (steps * model.time_step)
            if t > 0 and hasattr(ex, "exit_count"):
                metrics[key] = ex.exit_count / t
            else:
                metrics[key] = 0.0

    # --- Análisis por tipo de persona ---
    if hasattr(model, "person_data") and model.person_data:
        df_persons = pd.DataFrame(model.person_data)
        
        if not df.empty and not df_persons.empty and "id" in df.columns and "id" in df_persons.columns:
            # Combinar datos de evacuación con atributos demográficos
            df_full = df_persons.merge(df[["id", "t_exit"]], on="id", how="left")
            
            # Marcar no evacuados como infinito para cálculos de percentiles
            df_full["t_exit"] = df_full["t_exit"].fillna(np.inf)
            
            # Calcular métricas por tipo
            if "tipo" in df_full.columns:
                for tipo in df_full["tipo"].dropna().unique():
                    cleaned_tipo = tipo.replace(" ", "_").replace("-", "_")
                    mask = df_full["tipo"] == tipo
                    evacs = df_full.loc[mask, "t_exit"]
                    evacs_ok = evacs[evacs < np.inf]
                    
                    total_grupo = mask.sum()
                    evacuados_grupo = len(evacs_ok)
                    
                    if evacuados_grupo > 0:
                        metrics[f"p90_{cleaned_tipo}"] = evacs_ok.quantile(0.9)
                        metrics[f"p50_{cleaned_tipo}"] = evacs_ok.quantile(0.5)
                        metrics[f"evacuados_{cleaned_tipo}"] = evacuados_grupo
                    else:
                        metrics[f"p90_{cleaned_tipo}"] = np.nan
                        metrics[f"p50_{cleaned_tipo}"] = np.nan
                        metrics[f"evacuados_{cleaned_tipo}"] = 0
                    
                    # Porcentaje evacuado de este grupo
                    metrics[f"pct_evac_{cleaned_tipo}"] = (evacuados_grupo / total_grupo * 100) if total_grupo > 0 else 0.0

    return df, ts, perc, metrics


def save_times(df, path_csv):
    """Guarda los tiempos de evacuación individuales en CSV"""
    df.to_csv(path_csv, index=False)


def save_metrics(metrics_dict, path_csv):
    """Guarda las métricas de la simulación en CSV"""
    pd.DataFrame([metrics_dict]).to_csv(path_csv, index=False)


def plot_curva(ts, perc, title, out_png):
    """Genera y guarda la curva de evacuación"""
    import matplotlib
    try:
        matplotlib.get_backend()
    except Exception:
        matplotlib.use("Agg")

    plt.figure(figsize=(10, 6))
    plt.plot(ts, perc, linewidth=2, color='steelblue')
    plt.xlabel("Tiempo (s)", fontsize=12)
    plt.ylabel("% evacuado", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
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

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
    
    for (label, ts, perc), color in zip(series, colors):
        plt.plot(ts, perc, label=label, linewidth=2, color=color)
    
    plt.xlabel("Tiempo (s)", fontsize=12)
    plt.ylabel("% evacuado", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()