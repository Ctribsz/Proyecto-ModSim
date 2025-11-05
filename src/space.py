from collections import deque
import numpy as np

def bfs_distance_field(width, height, exit_positions, obstacles=None):
    """
    Devuelve un array (height, width) con la distancia Manhattan mínima
    hacia la celda de cualquier salida. Celdas inalcanzables quedan con np.inf.
    obstacles: set((x,y)) con celdas bloqueadas (opcional).
    """
    if obstacles is None:
        obstacles = set()

    INF = np.inf
    dist = np.full((height, width), INF, dtype=float)

    q = deque()
    # inicializar con las celdas de salida
    for (x, y) in exit_positions:
        if 0 <= x < width and 0 <= y < height and (x, y) not in obstacles:
            dist[y, x] = 0.0
            q.append((x, y))

    # 4-neighbors BFS (suficiente para distancia base)
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        x, y = q.popleft()
        d0 = dist[y, x]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                if dist[ny, nx] > d0 + 1:
                    dist[ny, nx] = d0 + 1
                    q.append((nx, ny))

    return dist


def neighbors_moore(x, y, width, height):
    """Vecindad de Moore (8 vecinos) + validación de bordes."""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny
