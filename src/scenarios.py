import numpy as np
import pandas as pd
from .model import EvacuationModel
from .space import bfs_distance_field, neighbors_moore  # ¡IMPORTANTE!
from .metrics import run_model

def baseline(N=300, width=25, height=25, num_exits=3, seed=42, max_steps=5000):
    model = EvacuationModel(width=width, height=height, N=N, num_exits=num_exits, seed=seed)
    return run_model(model, max_steps=max_steps)

def bloqueo(N=300, width=25, height=25, num_exits=3, seed=42, t_bloqueo=60.0, exit_index=0, max_steps=5000):
    """
    Bloquea una salida (exit_index) en t >= t_bloqueo (segundos).
    Implementación robusta: actualiza campo de distancias y libera agentes atrapados.
    """
    model = EvacuationModel(width=width, height=height, N=N, num_exits=num_exits, seed=seed)
    done_block = False
    blocked_exit_pos = None

    steps = 0
    while model.running and steps < max_steps:
        t_now = steps * model.time_step
        
        # Bloquear salida en el tiempo especificado
        if (not done_block) and (t_now >= t_bloqueo) and (0 <= exit_index < len(model.exits)):
            ex = model.exits[exit_index]
            blocked_exit_pos = ex.pos
            
            # 1. Remover salida del grid y scheduler
            try:
                model.grid.remove_agent(ex)
            except Exception:
                pass
            try:
                model.schedule.remove(ex)
            except Exception:
                pass
            
            # 2. Quitar de la lista de salidas
            model.exits.pop(exit_index)
            
            # 3. Actualizar campo de distancias (¡CRÍTICO!)
            model.exit_positions = [ex.pos for ex in model.exits]
            model.dist_field = bfs_distance_field(
                model.width, 
                model.height, 
                model.exit_positions, 
                model.obstacles
            )
            
            # 4. Liberar agentes atrapados en esa salida
            if blocked_exit_pos:
                x_ex, y_ex = blocked_exit_pos
                affected_agents = 0
                for nx, ny in neighbors_moore(x_ex, y_ex, model.width, model.height):
                    contents = model.grid.get_cell_list_contents([(nx, ny)])
                    for a in contents:
                        if hasattr(a, "state") and hasattr(a, "target_exit"):
                            if a.state == "WAITING" and a.target_exit == ex:
                                a.state = "MOVING"
                                a.target_exit = None  # forzar reelección
                                affected_agents += 1
            
            done_block = True
            print(f"✅ Salida {exit_index} bloqueada en t={t_now:.1f}s. "
                  f"Nuevas salidas: {len(model.exits)}, Agentes liberados: {affected_agents}")

        model.step()
        steps += 1

    # Recolectar resultados con la nueva estructura
    exit_events = getattr(model, "exit_events", [])
    df = pd.DataFrame(exit_events) if exit_events else pd.DataFrame(columns=["id", "t_exit"])
    if not df.empty and "t_exit" in df.columns:
        df = df.sort_values("t_exit").reset_index(drop=True)

    makespan = df["t_exit"].max() if not df.empty and "t_exit" in df.columns else np.nan
    p50 = df["t_exit"].quantile(0.5) if not df.empty and "t_exit" in df.columns else np.nan
    p90 = df["t_exit"].quantile(0.9) if not df.empty and "t_exit" in df.columns else np.nan

    ts = np.arange(0, steps + 1) * model.time_step
    evac_counts = []
    for s in range(steps + 1):
        t = s * model.time_step
        count = np.sum(df["t_exit"] <= t) if not df.empty and "t_exit" in df.columns else 0
        evac_counts.append(count)
    
    # Usar población inicial real
    initial_population = len(model.person_data) if hasattr(model, "person_data") else N
    perc = np.array(evac_counts) / max(initial_population, 1) * 100.0

    metrics = {
        "steps": steps,
        "time_step": model.time_step,
        "makespan": makespan,
        "p50": p50,
        "p90": p90,
        "evacuados": len(df) if not df.empty else 0,
        "t_bloqueo": t_bloqueo,
        "exit_index": exit_index,
        "num_exits_inicial": num_exits,
        "num_exits_final": len(model.exits),
        "initial_population": initial_population
    }
    return df, ts, perc, metrics

def anchos(N=300, width=25, height=25, lista_anchos=(1, 2, 3), seed=42, max_steps=5000):
    """
    Barrido de 'anchos' como PROXY simple usando número de salidas (=capacidad equivalente).
    """
    resultados = []
    for a in lista_anchos:
        df, ts, perc, met = baseline(N=N, width=width, height=height, num_exits=int(a), seed=seed, max_steps=max_steps)
        met = {**met, "ancho_proxy": a, "num_exits": int(a)}
        resultados.append((a, df, ts, perc, met))
    return resultados