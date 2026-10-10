import pygame
import math
import random
import sys
from collections import deque

# -------------------------------------------------------------------------------
# constants
# -------------------------------------------------------------------------------
WIDTH, HEIGHT = 900, 700
FPS = 60
CELL = 40
COLS = WIDTH  // CELL
ROWS = HEIGHT // CELL
RAY_STEP = 2  # Increased ray casting precision/resolution

WALL_COLOR  = (70, 90, 130)
WALL_DIM    = (30, 40, 65)
LAMP_RADIUS = 6
FINISH_RADIUS = 22

AUTO_SPEED = 2.5   # pixels/frame (auto mode speed)


# -------------------------------------------------------------------------------
# Wikipedia Line-Line Intersection
# -------------------------------------------------------------------------------
def segment_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, x1 + t*(x2-x1), y1 + t*(y2-y1)
    return None

# -------------------------------------------------------------------------------
# Maze generation (Recursive Backtracker)
# -------------------------------------------------------------------------------
def generate_maze(cols, rows):
    visited = [[False]*rows for _ in range(cols)]
    h_walls = [[True ]*rows for _ in range(cols)]    # right wall
    v_walls = [[True ]*cols for _ in range(rows)]    # bottom wall

    def neighbours(cx, cy):
        ns = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[nx][ny]:
                ns.append((nx, ny, dx, dy))
        return ns

    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        cx, cy = stack[-1]
        ns = neighbours(cx, cy)
        if ns:
            nx, ny, dx, dy = random.choice(ns)
            if dx ==  1: h_walls[cx][cy] = False
            if dx == -1: h_walls[nx][ny] = False
            if dy ==  1: v_walls[cy][cx] = False
            if dy == -1: v_walls[ny][nx] = False
            visited[nx][ny] = True
            stack.append((nx, ny))
        else:
            stack.pop()
    return h_walls, v_walls

def build_segments(cols, rows, cell, h_walls, v_walls):
    W, H = cols*cell, rows*cell
    segs = [
        (0, 0, W, 0), (W, 0, W, H),
        (W, H, 0, H), (0, H, 0, 0),
    ]
    for cx in range(cols-1):
        for cy in range(rows):
            if h_walls[cx][cy]:
                x = (cx+1)*cell
                segs.append((x, cy*cell, x, (cy+1)*cell))
    for cy in range(rows-1):
        for cx in range(cols):
            if v_walls[cy][cx]:
                y = (cy+1)*cell
                segs.append((cx*cell, y, (cx+1)*cell, y))
    return segs

# -------------------------------------------------------------------------------
# Cell graph: which cell can be moved to which
# -------------------------------------------------------------------------------
def build_graph(cols, rows, h_walls, v_walls):
    """Return the passable neighbors of each (cx,cy) cell.
    h_walls[cx][cy] = True → THERE IS a wall between cx and cx+1
    v_walls[cy][cx] = True → THERE IS a wall between cy and cy+1"""
    graph = {}
    for cx in range(cols):
        for cy in range(rows):
            nbrs = []
            # Right
            if cx+1 < cols and not h_walls[cx][cy]:
                nbrs.append((cx+1, cy))
            # Left
            if cx-1 >= 0 and not h_walls[cx-1][cy]:
                nbrs.append((cx-1, cy))
            # Down
            if cy+1 < rows and not v_walls[cy][cx]:
                nbrs.append((cx, cy+1))
            # Above
            if cy-1 >= 0 and not v_walls[cy-1][cx]:
                nbrs.append((cx, cy-1))
            graph[(cx,cy)] = nbrs
    return graph

# -------------------------------------------------------------------------------
# BFS: find path from pixel position to finish
# -------------------------------------------------------------------------------
def pixel_to_cell(px, py):
    return int(px // CELL), int(py // CELL)

def cell_center(cx, cy):
    return (cx * CELL + CELL // 2, cy * CELL + CELL // 2)

def bfs_path(graph, start_cell, finish_cell):
    """Return start→finish cell path with BFS."""
    queue = deque([(start_cell, [start_cell])])
    visited = {start_cell}
    while queue:
        node, path = queue.popleft()
        if node == finish_cell:
            return path
        for nb in graph.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return [start_cell]  # not found (should not be)

def build_waypoints(graph, lx, ly, finish_pos):
    """Create pixel waypoint list from current location to finish."""
    sc = pixel_to_cell(lx, ly)
    fc = pixel_to_cell(*finish_pos)
    path = bfs_path(graph, sc, fc)
    waypoints = [cell_center(cx, cy) for cx, cy in path]
    # The final point is the exact finishing position
    waypoints.append(finish_pos)
    return waypoints

# -------------------------------------------------------------------------------
# Wall collision check (manual mode)
# -------------------------------------------------------------------------------
def crosses_wall(x1, y1, x2, y2, segments):
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return False
    nx = -dy / length * LAMP_RADIUS
    ny =  dx / length * LAMP_RADIUS
    for ox, oy in [(0,0), (nx,ny), (-nx,-ny)]:
        ax, ay = x1+ox, y1+oy
        bx, by = x2+ox, y2+oy
        for seg in segments:
            if segment_intersect(ax, ay, bx, by, *seg):
                return True
    return False

# -------------------------------------------------------------------------------
# Ray casting
# -------------------------------------------------------------------------------
def cast_rays(lx, ly, segments):
    hits = []
    FAR  = 2000
    angle = 0.0
    while angle < 360.0:
        rad = math.radians(angle)
        rx2 = lx + math.cos(rad) * FAR
        ry2 = ly + math.sin(rad) * FAR
        best_t  = float('inf')
        best_pt = (rx2, ry2)
        for seg in segments:
            res = segment_intersect(lx, ly, rx2, ry2, *seg)
            if res and res[0] < best_t:
                best
