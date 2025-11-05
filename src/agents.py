from mesa import Agent
import numpy as np
import random
from .space import neighbors_moore

class ExitAgent(Agent):
    """
    Salida con capacidad de servicio (personas/s).
    Implementa un 'service credit' acumulado por tick: credit += cap_ps * dt,
    y evacúa floor(credit) personas en cola; reduce credit en esa cantidad.
    La cola es implícita: personas adyacentes que se 'anclan' a esta salida.
    """
    def __init__(self, unique_id, model, pos, capacity_ps=1.3):
        super().__init__(unique_id, model)
        self.pos = pos
        self.capacity_ps = capacity_ps
        self.service_credit = 0.0

    def step(self):
        # acumular capacidad
        self.service_credit += self.capacity_ps * self.model.time_step

        # recolectar candidatos (personas esperando junto a la salida)
        x, y = self.pos
        candidates = []
        for nx, ny in neighbors_moore(x, y, self.model.width, self.model.height):
            contents = self.model.grid.get_cell_list_contents([(nx, ny)])
            for a in contents:
                if isinstance(a, PersonAgent) and (not a.evacuated) and a.target_exit is self and a.state == "WAITING":
                    candidates.append(a)

        # servir hasta 'credit' personas
        k = int(self.service_credit)
        served = 0
        # orden aleatorio para no sesgar
        random.shuffle(candidates)
        for a in candidates:
            if served >= k:
                break
            # evacuar
            a.evacuated = True
            t_now = self.model.schedule.steps * self.model.time_step
            a.t_exit = t_now
            self.model.exit_times.append(t_now)

            # remover del grid/schedule
            try:
                self.model.grid.remove_agent(a)
            except Exception:
                pass
            try:
                self.model.schedule.remove(a)
            except Exception:
                pass
            served += 1

        self.service_credit -= served


class PersonAgent(Agent):
    """
    Persona que se dirige a la salida con menor distancia BFS.
    Estados:
      - MOVING: se mueve hacia la salida objetivo
      - WAITING: anclado a una salida (en celda adyacente) esperando servicio
      - (al evacuar se elimina del modelo)
    """
    def __init__(self, unique_id, model, pos, v_cells_per_step=1):
        super().__init__(unique_id, model)
        self.pos = pos
        self.evacuated = False
        self.t_exit = None
        self.v_cells_per_step = v_cells_per_step
        self.state = "MOVING"
        self.target_exit = None  # ExitAgent elegido actualmente

    # ---------------------------
    def _choose_best_exit(self):
        """
        Elige la salida con menor distancia BFS desde la posición actual.
        Si hay empate, incorpora heurística ligera de cola (menor credit esperado).
        """
        x, y = self.pos
        best = []
        best_val = float("inf")
        for ex in self.model.exits:
            d = self.model.dist_field[y, x]  # distancia a la salida más cercana (global)
            # pequeña penalización por cola (inversa de credit disponible)
            # Nota: esto es un proxy muy barato; para colas reales, medir candidatos WAITING por salida.
            pen = 0.0
            best_val = min(best_val, d + pen)

        # recolectar empatadas (con misma mejor distancia global)
        for ex in self.model.exits:
            d = self.model.dist_field[y, x]
            if abs((d) - best_val) < 1e-9:
                best.append(ex)

        # aleatorizar entre empatadas
        return random.choice(best) if best else (self.model.exits[0] if self.model.exits else None)

    def _is_adjacent_to_exit(self, ex):
        """¿Está en una celda vecina (Moore) a la salida ex?"""
        if ex is None:
            return False
        ex_x, ex_y = ex.pos
        x, y = self.pos
        return max(abs(ex_x - x), abs(ex_y - y)) == 1 or (ex_x == x and ex_y == y)

    def _best_neighbor_step(self):
        """
        Elige vecino que reduzca la distancia BFS hacia CUALQUIER salida.
        Usa el campo global dist_field (mínimo a salidas).
        Permite quedarse si no hay mejora.
        """
        x, y = self.pos
        candidates = [(x, y)]
        best_val = self.model.dist_field[y, x]
        best_cells = [ (x, y) ]

        for nx, ny in neighbors_moore(x, y, self.model.width, self.model.height):
            if (nx, ny) in self.model.obstacles:
                continue
            val = self.model.dist_field[ny, nx]
            if val < best_val - 1e-9:
                best_val = val
                best_cells = [(nx, ny)]
            elif abs(val - best_val) < 1e-9:
                best_cells.append((nx, ny))

        return random.choice(best_cells)

    # ---------------------------
    def step(self):
        if self.evacuated:
            return

        if self.state == "WAITING":
            # no se mueve; el ExitAgent lo evacuará cuando tenga capacidad
            return

        # asegurar objetivo
        if (self.target_exit is None) or (self.target_exit not in self.model.exits):
            self.target_exit = self._choose_best_exit()

        # avanzar v_cells_per_step micro-pasos
        steps_left = self.v_cells_per_step
        while steps_left > 0 and (not self.evacuated) and self.state == "MOVING":
            new_pos = self._best_neighbor_step()
            self.model.grid.move_agent(self, new_pos)

            # ¿adyacente a la salida objetivo? -> anclarse a cola (WAITING)
            if self._is_adjacent_to_exit(self.target_exit):
                self.state = "WAITING"
                # quedarse en esta celda; el ExitAgent decidirá cuándo evacuar
                break

            steps_left -= 1