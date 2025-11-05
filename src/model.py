from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import random

from .agents import PersonAgent, ExitAgent
from .space import bfs_distance_field

class EvacuationModel(Model):
    """
    Modelo base de evacuación con:
    - Campo de distancias (BFS) hacia las salidas
    - Salidas con ancho -> capacidad (personas/seg)
    - Colas en puerta y servicio por capacidad
    Δt por defecto = 0.1 s
    """

    def __init__(
        self,
        width=25,
        height=25,
        N=300,
        num_exits=3,
        exit_widths=None,         # lista de anchos (m) por salida
        persons_speed_cells=1,    # celdas por tick (discreto)
        seed=None,
        time_step=0.1
    ):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.width = width
        self.height = height
        self.N = N
        self.num_exits = num_exits
        self.time_step = time_step
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True

        # Parámetros de puertas: si no envían anchos, 1.0 m por defecto
        if exit_widths is None:
            exit_widths = [1.0] * num_exits
        else:
            # ajustar tamaño si difiere
            if len(exit_widths) < num_exits:
                exit_widths = list(exit_widths) + [exit_widths[-1]] * (num_exits - len(exit_widths))
            elif len(exit_widths) > num_exits:
                exit_widths = exit_widths[:num_exits]
        self.exit_widths = exit_widths

        # === Crear salidas en el borde inferior, equidistantes ===
        self.exits = []
        self.exit_positions = self._generate_exit_positions()
        for i, pos in enumerate(self.exit_positions):
            width_m = self.exit_widths[i]
            # capacidad (personas/s) ~ 1.3 * ancho (regla simple)
            cap_ps = 1.3 * width_m
            exit_agent = ExitAgent(
                self.next_id(),
                self,
                pos=pos,
                capacity_ps=cap_ps
            )
            self.grid.place_agent(exit_agent, pos)
            self.schedule.add(exit_agent)
            self.exits.append(exit_agent)

        # === Campo de distancias (BFS) hacia salidas ===
        self.obstacles = set()  # si luego agregas paredes internas, añádelas aquí
        self.dist_field = bfs_distance_field(self.width, self.height, self.exit_positions, self.obstacles)

        # === Crear personas ===
        self.exit_times = []
        for _ in range(N):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            # evitar spawnear sobre una salida
            while (x, y) in self.exit_positions:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

            agent = PersonAgent(
                self.next_id(),
                self,
                pos=(x, y),
                v_cells_per_step=persons_speed_cells
            )
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        # === DataCollector ===
        self.datacollector = DataCollector(
            model_reporters={
                "Evacuados": lambda m: sum(a.evacuated for a in m.schedule.agents if isinstance(a, PersonAgent))
            }
        )

    # ------------------------------------------------
    def _generate_exit_positions(self):
        pos = []
        for i in range(self.num_exits):
            x = int((i + 1) * self.width / (self.num_exits + 1))
            y = 0  # borde inferior
            pos.append((x, y))
        return pos

    # ------------------------------------------------
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

        # detener si ya no quedan personas
        any_left = any(isinstance(a, PersonAgent) for a in self.schedule.agents)
        if not any_left:
            self.running = False
