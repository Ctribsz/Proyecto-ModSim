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
        self.exit_count = 0  # Contador para throughput

    def step(self):
        # Acumular capacidad
        self.service_credit += self.capacity_ps * self.model.time_step

        # Recoger candidatos (personas esperando junto a la salida)
        x, y = self.pos
        candidates = []
        for nx, ny in neighbors_moore(x, y, self.model.width, self.model.height):
            contents = self.model.grid.get_cell_list_contents([(nx, ny)])
            for a in contents:
                if isinstance(a, PersonAgent) and (not a.evacuated) and a.target_exit is self and a.state == "WAITING":
                    candidates.append(a)

        # Servir hasta 'credit' personas
        k = int(self.service_credit)
        served = 0
        random.shuffle(candidates)  # Orden aleatorio para no sesgar
        for a in candidates:
            if served >= k:
                break
            # Evacuar
            a.evacuated = True
            t_now = self.model.schedule.steps * self.model.time_step
            a.t_exit = t_now
            self.model.exit_events.append({"id": a.unique_id, "t_exit": t_now})
            self.exit_count += 1  # ¡CORREGIDO: solo una vez!

            # Remover del grid/schedule
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
    Persona con atributos realistas: edad, movilidad, pánico, familiaridad, cumplimiento.
    Estados: MOVING → WAITING → (evacuado y removido)
    """
    def __init__(
        self,
        unique_id,
        model,
        pos,
        tipo="adulto",
        edad=30,
        v_cells_per_step=1,
        pánico=0.2,
        familiaridad=True,
        cumplimiento=1.0,
        movilidad_reducida=False,
    ):
        super().__init__(unique_id, model)
        self.pos = pos
        self.evacuated = False
        self.t_exit = None
        self.tipo = tipo
        self.edad = edad
        self.v_cells_per_step = v_cells_per_step
        self.pánico = pánico
        self.familiaridad = familiaridad
        self.cumplimiento = cumplimiento
        self.movilidad_reducida = movilidad_reducida
        self.state = "MOVING"
        self.target_exit = None
        self.reelecciones = 0
        self.preferred_exit_id = None  # para modelar familiaridad

    # --------------------------------------------------
    def _choose_best_exit(self):
        """Elige salida con utilidad: distancia, congestión, pánico, familiaridad."""
        if not self.model.exits:
            return None

        opciones = []
        for ex in self.model.exits:
            d = self.model.dist_field[self.pos[1], self.pos[0]]  # distancia global

            # Estimar congestión: agentes WAITING cerca de esta salida
            x_ex, y_ex = ex.pos
            cola = 0
            for nx, ny in neighbors_moore(x_ex, y_ex, self.model.width, self.model.height):
                for a in self.model.grid.get_cell_list_contents([(nx, ny)]):
                    if isinstance(a, PersonAgent) and a.state == "WAITING" and a.target_exit is ex:
                        cola += 1

            # Utilidad negativa (minimizar)
            utilidad = d + 0.5 * cola  # mayor distancia o cola → peor

            # Bonus por familiaridad (si ya usó esta salida antes o conoce el lugar)
            if self.familiaridad and self.preferred_exit_id is not None and ex.unique_id == self.preferred_exit_id:
                utilidad -= 0.3

            # Ruido por pánico: más indecisión si pánico alto
            if self.pánico > 0.6:
                utilidad += np.random.uniform(-0.4, 0.4)

            opciones.append((utilidad, ex))

        # Escoger con probabilidad softmax
        utils, exits = zip(*opciones)
        utils = np.array(utils)
        # Evitar overflow
        utils = utils - np.max(utils)
        probs = np.exp(-utils)  # menos utilidad → más probabilidad
        probs = probs / probs.sum()
        idx = np.random.choice(len(exits), p=probs)
        elegida = exits[idx]

        # Recordar salida elegida para modelar aprendizaje/familiaridad
        if self.familiaridad and self.preferred_exit_id is None:
            self.preferred_exit_id = elegida.unique_id

        return elegida

    # --------------------------------------------------
    def _is_adjacent_to_exit(self, ex):
        if ex is None:
            return False
        ex_x, ex_y = ex.pos
        x, y = self.pos
        return max(abs(ex_x - x), abs(ex_y - y)) <= 1

    # --------------------------------------------------
    def _best_neighbor_step(self):
        x, y = self.pos
        best_val = self.model.dist_field[y, x]
        best_cells = [(x, y)]

        for nx, ny in neighbors_moore(x, y, self.model.width, self.model.height):
            if (nx, ny) in self.model.obstacles:
                continue
            val = self.model.dist_field[ny, nx]
            if val < best_val - 1e-6:
                best_val = val
                best_cells = [(nx, ny)]
            elif abs(val - best_val) < 1e-6:
                best_cells.append((nx, ny))

        return random.choice(best_cells)

    # --------------------------------------------------
    def step(self):
        if self.evacuated:
            return

        if self.state == "WAITING":
            # Mientras espera, puede reconsiderar si el pánico es alto o el objetivo desaparece
            valid_exit = False
            if self.target_exit is not None:
                valid_exit = any(ex.unique_id == self.target_exit.unique_id for ex in self.model.exits)
            
            if not valid_exit or (self.pánico > 0.7 and self.random.random() < 0.05):
                self.target_exit = self._choose_best_exit()
                if self.target_exit:
                    self.state = "MOVING"  # volver a moverse hacia nueva salida
            return

        # Reevaluar salida cada 10 steps (simula indecisión realista)
        valid_exit = False
        if self.target_exit is not None:
            valid_exit = any(ex.unique_id == self.target_exit.unique_id for ex in self.model.exits)
        
        if self.model.schedule.steps % 10 == 0 or not valid_exit:
            old_exit = self.target_exit
            self.target_exit = self._choose_best_exit()
            if old_exit is not self.target_exit and self.target_exit is not None:
                self.reelecciones += 1

        # Si aún no tiene objetivo, asignar uno
        if self.target_exit is None:
            self.target_exit = self._choose_best_exit()

        # Avanzar v_cells_per_step micro-pasos
        steps_left = self.v_cells_per_step
        while steps_left > 0 and not self.evacuated and self.state == "MOVING":
            new_pos = self._best_neighbor_step()
            self.model.grid.move_agent(self, new_pos)

            # Si llega adyacente o encima de su salida objetivo → anclarse
            if self.target_exit is not None and self._is_adjacent_to_exit(self.target_exit):
                self.state = "WAITING"
                break

            steps_left -= 1