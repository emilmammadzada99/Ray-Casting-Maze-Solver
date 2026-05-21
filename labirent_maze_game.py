"""2D Maze Light Animation
===========================
Ray casting + Start/Finish + Automatic Solver (BFS) + Compass

Controls:
  Mouse → Manual mode: move the lamp
  A → Auto mode on/off (lamp finds finish via BFS)
  R → Generate new maze
  ESC → Exit"""

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
RAY_STEP = 8

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
# Maze production (Recursive Backtracker)
# -------------------------------------------------------------------------------
def generate_maze(cols, rows):
    visited = [[False]*rows for _ in range(cols)]
    h_walls = [[True ]*rows for _ in range(cols)]   # right wall
    v_walls = [[True ]*cols for _ in range(rows)]   # bottom wall

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
            # left
            if cx-1 >= 0 and not h_walls[cx-1][cy]:
                nbrs.append((cx-1, cy))
            # down
            if cy+1 < rows and not v_walls[cy][cx]:
                nbrs.append((cx, cy+1))
            # above
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
# ray casting
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
                best_t  = res[0]
                best_pt = (res[1], res[2])
        hits.append(best_pt)
        angle += RAY_STEP
    return hits

# -------------------------------------------------------------------------------
# compass drawing
# -------------------------------------------------------------------------------
def draw_compass(surface, lx, ly, finish_pos, auto_mode):
    fx, fy = finish_pos
    angle_to_finish = math.atan2(fy - ly, fx - lx)

    cx, cy = int(lx), int(ly)
    R = 28        # compass outer radius
    AR = 18       # arrow length

    # Compass background circle
    comp_surf = pygame.Surface((R*2+4, R*2+4), pygame.SRCALPHA)
    pygame.draw.circle(comp_surf, (20, 20, 40, 180), (R+2, R+2), R)
    pygame.draw.circle(comp_surf, (80, 100, 160, 200), (R+2, R+2), R, 2)
    surface.blit(comp_surf, (cx - R - 2, cy - R - 2))

    # Arrowhead (towards the finish)
    tip_x = cx + math.cos(angle_to_finish) * AR
    tip_y = cy + math.sin(angle_to_finish) * AR
    tail_x = cx - math.cos(angle_to_finish) * (AR * 0.5)
    tail_y = cy - math.sin(angle_to_finish) * (AR * 0.5)

    # Arrow color: blue in auto mode, green in manual mode
    arrow_color = (80, 180, 255) if auto_mode else (80, 255, 140)

    pygame.draw.line(surface, arrow_color,
                     (int(tail_x), int(tail_y)),
                     (int(tip_x),  int(tip_y)), 3)

    # Arrowhead (triangle)
    perp = angle_to_finish + math.pi / 2
    p1 = (int(tip_x), int(tip_y))
    p2 = (int(tip_x - math.cos(angle_to_finish)*8 + math.cos(perp)*5),
          int(tip_y - math.sin(angle_to_finish)*8 + math.sin(perp)*5))
    p3 = (int(tip_x - math.cos(angle_to_finish)*8 - math.cos(perp)*5),
          int(tip_y - math.sin(angle_to_finish)*8 - math.sin(perp)*5))
    pygame.draw.polygon(surface, arrow_color, [p1, p2, p3])

    # center point
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), 3)

# -------------------------------------------------------------------------------
# drawing aids
# -------------------------------------------------------------------------------
def draw_rays_fast(surface, lx, ly, hits):
    for hx, hy in hits:
        dist = math.hypot(hx-lx, hy-ly)
        brightness = max(20, int(160 - dist * 0.22))
        pygame.draw.line(surface, (brightness, brightness, brightness),
                         (int(lx), int(ly)), (int(hx), int(hy)), 1)

def draw_light_polygon(surface, lx, ly, hits):
    if len(hits) < 2:
        return
    poly_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pts = [(int(lx), int(ly))] + [(int(hx), int(hy)) for hx, hy in hits]
    pts.append(pts[1])
    pygame.draw.polygon(poly_surf, (255, 190, 60, 28), pts)
    surface.blit(poly_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

def draw_lamp(surface, lx, ly, tick):
    pulse = math.sin(tick * 0.06) * 2
    r = int(6 + pulse)
    pygame.draw.circle(surface, (180, 200, 255), (int(lx), int(ly)), r + 3)
    pygame.draw.circle(surface, (220, 230, 255), (int(lx), int(ly)), r)
    pygame.draw.circle(surface, (255, 255, 255), (int(lx), int(ly)), max(r-2, 2))

def draw_maze_walls(surface, segments):
    for seg in segments[4:]:
        x1, y1, x2, y2 = seg
        pygame.draw.line(surface, WALL_DIM,   (x1+2, y1+2), (x2+2, y2+2), 2)
        pygame.draw.line(surface, WALL_COLOR, (x1,   y1  ), (x2,   y2  ), 2)
    for seg in segments[:4]:
        pygame.draw.line(surface, WALL_COLOR, (seg[0], seg[1]), (seg[2], seg[3]), 3)

def draw_corner_label(surface, font, text, cx, cy, color, bg_color):
    txt_surf = font.render(text, True, color)
    tw, th = txt_surf.get_size()
    pad = 4
    rect = pygame.Rect(cx - tw//2 - pad, cy - th//2 - pad, tw + pad*2, th + pad*2)
    bg = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg.fill((*bg_color, 180))
    surface.blit(bg, rect.topleft)
    surface.blit(txt_surf, (cx - tw//2, cy - th//2))

def draw_finish(surface, font, fx, fy, tick):
    pulse = abs(math.sin(tick * 0.05))
    alpha = int(120 + 100 * pulse)
    r = FINISH_RADIUS
    circle_surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
    pygame.draw.circle(circle_surf, (0, 200, 80, alpha), (r, r), r)
    pygame.draw.circle(circle_surf, (0, 255, 100, min(alpha+60, 255)), (r, r), r, 2)
    surface.blit(circle_surf, (int(fx)-r, int(fy)-r))
    draw_corner_label(surface, font, "FINISH", int(fx), int(fy), (0, 255, 100), (0, 80, 30))

def draw_win_screen(surface, font, font_small, tick):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))
    pulse = abs(math.sin(tick * 0.07))
    big_font = pygame.font.SysFont("consolas", int(60 + 10*pulse), bold=True)
    win_surf = big_font.render("YOU WIN!", True, (0, 255, 100))
    wr, wh = win_surf.get_size()
    surface.blit(win_surf, (WIDTH//2 - wr//2, HEIGHT//2 - wh//2 - 30))
    sub = font_small.render("R → New Game  |  A → Automatic/Manual Mode", True, (180, 220, 180))
    sr, sh = sub.get_size()
    surface.blit(sub, (WIDTH//2 - sr//2, HEIGHT//2 + 50))

# -------------------------------------------------------------------------------
# Starting a game
# -------------------------------------------------------------------------------
def random_interior_pos():
    cx = random.randint(1, COLS - 2)
    cy = random.randint(1, ROWS - 2)
    return (cx * CELL + CELL // 2, cy * CELL + CELL // 2)

def new_game():
    h_walls, v_walls = generate_maze(COLS, ROWS)
    segments = build_segments(COLS, ROWS, CELL, h_walls, v_walls)
    graph    = build_graph(COLS, ROWS, h_walls, v_walls)

    start_pos = random_interior_pos()
    while True:
        finish_pos = random_interior_pos()
        sx, sy = int(start_pos[0] // CELL), int(start_pos[1] // CELL)
        fx, fy = int(finish_pos[0] // CELL), int(finish_pos[1] // CELL)
        if abs(sx - fx) + abs(sy - fy) >= 10:
            break

    return segments, graph, start_pos, finish_pos

# -------------------------------------------------------------------------------
# main loop
# -------------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        "Labirent Maze")
    clock = pygame.time.Clock()
    font       = pygame.font.SysFont("consolas", 13, bold=True)
    font_hud   = pygame.font.SysFont("consolas", 15)
    font_small = pygame.font.SysFont("consolas", 20)

    # Ground
    bg = pygame.Surface((WIDTH, HEIGHT))
    for cx in range(COLS):
        for cy in range(ROWS):
            shade = (10, 10, 18) if (cx+cy) % 2 == 0 else (8, 8, 15)
            pygame.draw.rect(bg, shade, (cx*CELL, cy*CELL, CELL, CELL))

    def reset():
        nonlocal segments, graph, start_pos, finish_pos, lx, ly, hits
        nonlocal won, auto_mode, waypoints, wp_idx
        segments, graph, start_pos, finish_pos = new_game()
        lx, ly = float(start_pos[0]), float(start_pos[1])
        hits = cast_rays(lx, ly, segments)
        won = False
        auto_mode  = False
        waypoints  = []
        wp_idx     = 0

    segments = graph = start_pos = finish_pos = None
    lx = ly = 0.0
    hits = []
    won = auto_mode = False
    waypoints = []
    wp_idx = 0

    reset()
    tick = 0

    while True:
        clock.tick(FPS)
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                if event.key == pygame.K_r:
                    reset()

                if event.key == pygame.K_a and not won:
                    auto_mode = not auto_mode
                    if auto_mode:
                        # BFS calculate path
                        waypoints = build_waypoints(graph, lx, ly, finish_pos)
                        wp_idx = 0

        # None
        # Movement
        # None
        if not won:
            if auto_mode:
                # Automatic: move to next waypoint
                if wp_idx < len(waypoints):
                    tx, ty = waypoints[wp_idx]
                    dx = tx - lx
                    dy = ty - ly
                    dist = math.hypot(dx, dy)
                    if dist < AUTO_SPEED:
                        lx, ly = float(tx), float(ty)
                        wp_idx += 1
                    else:
                        step = AUTO_SPEED / dist
                        lx += dx * step
                        ly += dy * step
                    hits = cast_rays(lx, ly, segments)
            else:
                # Manual: mouse control + wall collision
                mx, my = pygame.mouse.get_pos()
                mx = max(LAMP_RADIUS, min(WIDTH  - LAMP_RADIUS, mx))
                my = max(LAMP_RADIUS, min(HEIGHT - LAMP_RADIUS, my))
                if (mx, my) != (int(lx), int(ly)):
                    new_lx, new_ly = lx, ly
                    if not crosses_wall(lx, ly, float(mx), ly, segments):
                        new_lx = float(mx)
                    if not crosses_wall(new_lx, ly, new_lx, float(my), segments):
                        new_ly = float(my)
                    if (new_lx, new_ly) != (lx, ly):
                        lx, ly = new_lx, new_ly
                        hits = cast_rays(lx, ly, segments)

            # Finish control
            if math.hypot(lx - finish_pos[0], ly - finish_pos[1]) < FINISH_RADIUS:
                won = True
                auto_mode = False

        # None
        # Drawing
        # None
        screen.blit(bg, (0, 0))

        dark = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dark.fill((0, 0, 8, 210))
        screen.blit(dark, (0, 0))

        draw_light_polygon(screen, lx, ly, hits)
        draw_rays_fast(screen, lx, ly, hits)
        draw_maze_walls(screen, segments)
        pygame.draw.rect(screen, WALL_COLOR, (0, 0, WIDTH, HEIGHT), 3)

        draw_finish(screen, font, finish_pos[0], finish_pos[1], tick)
        draw_lamp(screen, lx, ly, tick)

        # Compass: show finish direction
        if not won:
            draw_compass(screen, lx, ly, finish_pos, auto_mode)

        draw_corner_label(screen, font, "START",
                          start_pos[0], start_pos[1],
                          (255, 80, 80), (80, 10, 10))

        if won:
            draw_win_screen(screen, font, font_small, tick)

        # HUD
        mode_txt = "[ AUTO ]" if auto_mode else "[MANUEL]"
        hud = font_hud.render(
            f"FPS: {int(clock.get_fps())}  {mode_txt}  |  A: Mode Control  |  R: New Game  |  ESC: Exit",
            True, (80, 110, 160)
        )
        screen.blit(hud, (8, 8))

        pygame.display.flip()

if __name__ == "__main__":
    main()