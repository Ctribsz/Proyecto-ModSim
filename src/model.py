from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import random

from .agents import PersonAgent, ExitAgent


class EvacuationModel(Model):
    """
    Modelo base de evacuación de personas.
    Cada tick equivale a 0.1 segundos.
    """

    def __init__(self, width=20, height=20, N=50, num_exits=2, seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.width = width
        self.height = height
        self.num_exits = num_exits
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.running = True
        self.time_step = 0.1  # segundos

        # === Crear salidas ===
        self.exits = []
        exit_positions = self._generate_exit_positions()
        for pos in exit_positions:
            exit_agent = ExitAgent(self.next_id(), self, pos)
            self.grid.place_agent(exit_agent, pos)
            self.schedule.add(exit_agent)
            self.exits.append(exit_agent)

        # === Crear personas ===
        for _ in range(N):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            while any(isinstance(a, ExitAgent) for a in self.grid.get_cell_list_contents((x, y))):
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

            agent = PersonAgent(self.next_id(), self, (x, y))
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        # === Recolector de datos ===
        self.datacollector = DataCollector(
            model_reporters={
                "Evacuados": lambda m: sum(a.evacuated for a in m.schedule.agents if isinstance(a, PersonAgent))
            }
        )

    # ------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------
    def _generate_exit_positions(self):
        """Genera posiciones de salida en los bordes del mapa"""
        positions = []
        # Colocar salidas equidistantes en la parte inferior
        for i in range(self.num_exits):
            x = int((i + 1) * self.width / (self.num_exits + 1))
            y = 0  # borde inferior
            positions.append((x, y))
        return positions

    # ------------------------------------------------
    # Step principal
    # ------------------------------------------------
    def step(self):
        """Avanza la simulación un tick"""
        self.datacollector.collect(self)
        self.schedule.step()

        # Si todos evacuaron, detener
        all_evacuated = all(a.evacuated for a in self.schedule.agents if isinstance(a, PersonAgent))
        if all_evacuated:
            self.running = False
