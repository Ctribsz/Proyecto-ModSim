# src/scenarios.py
from .model import EvacuationModel
from .metrics import run_model

def baseline(N=300, width=25, height=25, num_exits=3, seed=42, max_steps=5000):
    model = EvacuationModel(width=width, height=height, N=N, num_exits=num_exits, seed=seed)
    return run_model(model, max_steps=max_steps)


def bloqueo(N=300, width=25, height=25, num_exits=3, seed=42, t_bloqueo=60.0, exit_index=0, max_steps=5000):
    """
    Bloquea una salida (exit_index) en t >= t_bloqueo (segundos).
    Nota: simplificación — se 'retira' el ExitAgent del grid y del scheduler.
    """
    model = EvacuationModel(width=width, height=height, N=N, num_exits=num_exits, seed=seed)
    done_block = False

    steps = 0
    while model.running and steps < max_steps:
        t_now = steps * model.time_step
        if (not done_block) and (t_now >= t_bloqueo) and (len(model.exits) > exit_index):
            ex = model.exits[exit_index]
            # remover del grid y del scheduler
            try:
                model.grid.remove_agent(ex)
            except Exception:
                pass
            try:
                model.schedule.remove(ex)
            except Exception:
                pass
            # marcar como 'no disponible' (y quitar de la lista para evitar selección futura)
            model.exits.pop(exit_index)
            done_block = True

        model.step()
        steps += 1

    # al terminar, construir outputs usando el mismo cálculo que run_model:
    # para reutilizar, reconstruimos un mini-modelo solo para empaquetar resultados
    from .metrics import run_model as _run
    # truco: _run ya ejecuta; como ya ejecutamos, replicamos valores manualmente
    exit_times = getattr(model, "exit_times", [])
    import numpy as np, pandas as pd
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
        "evacuados": 0 if df.empty else int((~df["t_exit"].isna()).sum()),
        "t_bloqueo": t_bloqueo,
        "exit_index": exit_index,
        "num_exits_inicial": num_exits
    }
    return df, ts, perc, metrics


def anchos(N=300, width=25, height=25, lista_anchos=(1, 2, 3), seed=42, max_steps=5000):
    """
    Barrido de 'anchos' como PROXY simple usando número de salidas (=capacidad equivalente).
    Si luego implementas capacidad/puerta real, este wrapper queda igual.
    """
    resultados = []
    for a in lista_anchos:
        df, ts, perc, met = baseline(N=N, width=width, height=height, num_exits=int(a), seed=seed, max_steps=max_steps)
        met = {**met, "ancho_proxy": a, "num_exits": int(a)}
        resultados.append((a, df, ts, perc, met))
    return resultados