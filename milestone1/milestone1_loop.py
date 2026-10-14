# change to loop-based simulation for milestone 1 instead of multithreaded

import math
import sys
import time
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import pygame

GRID_SIZE = 10

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def delta(self):
        return {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
        }[self]

    def turn_left(self):
        return Direction((self.value - 1) % 4)

    def turn_right(self):
        return Direction((self.value + 1) % 4)


class Action(Enum):
    MOVE_FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    PICKUP = auto()


@dataclass
class Message:
    sender_id: int
    payload: str
    turn_sent: int


@dataclass
class Decision:
    robot_id: int
    action: Action
    position: tuple


class World:

    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.targets = set()
        self.robot_positions = {}
        self.robot_facings = {}

    def has_target(self, pos):
        return pos in self.targets

    def remove_target(self, pos):
        self.targets.discard(pos)

    def add_target(self, pos):
        self.targets.add(pos)

    def robots_at(self, pos):
        return [rid for rid, p in self.robot_positions.items() if p == pos]

    def set_robot_state(self, robot_id, pos, facing):
        self.robot_positions[robot_id] = pos
        self.robot_facings[robot_id] = facing

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def snapshot(self):
        return set(self.targets), dict(self.robot_positions), dict(self.robot_facings)

    def maybe_generate_random_targets(self, count=5):
        if len(self.targets) > 0:
            return
        occupied = set(self.robot_positions.values())
        reserved = {(3, 3), (7, 2), (2, 7), (7, 7)}
        for _ in range(count):
            for _attempt in range(50):
                pos = (random.randrange(self.size), random.randrange(self.size))
                if pos not in occupied and pos not in self.targets and pos not in reserved:
                    self.targets.add(pos)
                    break


class Robot:
    def __init__(self, simulation, robot_id, start_pos, start_facing, script=None):
        self.sim = simulation
        self.id = robot_id
        self.position = start_pos
        self.facing = start_facing
        self.script = script or {}
        self.inbox = deque()
        self.messages_this_turn = []
        self.turn = 0
        self._decided_action = Action.TURN_RIGHT
        self._sensed = {}

    def read_messages(self):
        self.messages_this_turn = []
        while self.inbox:
            self.messages_this_turn.append(self.inbox.popleft())

    def sense(self):
        dx, dy = self.facing.delta()
        front = (self.position[0] + dx, self.position[1] + dy)
        self._sensed = {
            "target_ahead": self.sim.world.has_target(front) if self.sim.world.in_bounds(front) else False,
            "robots_ahead": [rid for rid in self.sim.world.robots_at(front) if rid != self.id],
            "robots_here": [rid for rid in self.sim.world.robots_at(self.position) if rid != self.id],
            "wall_ahead": not self.sim.world.in_bounds(front),
        }

    def decide(self):
        if self.sim.random_mode:
            action = random.choice(list(Action))
        else:
            action = self.script.get(self.turn, Action.TURN_RIGHT)

        if action == Action.MOVE_FORWARD and self._sensed.get("wall_ahead"):
            action = random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])

        self._decided_action = action
        self.sim.decisions[self.id] = Decision(self.id, self._decided_action, self.position)

    def perform(self):
        action = self._decided_action
        if action == Action.MOVE_FORWARD:
            dx, dy = self.facing.delta()
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if self.sim.world.in_bounds(new_pos):
                self.position = new_pos
        elif action == Action.TURN_LEFT:
            self.facing = self.facing.turn_left()
        elif action == Action.TURN_RIGHT:
            self.facing = self.facing.turn_right()
        elif action == Action.PICKUP:
            if self.position in self.sim.valid_pickup_cells:
                self.sim.world.remove_target(self.position)
        self.sim.world.set_robot_state(self.id, self.position, self.facing)

    def send_messages(self):
        payload = f"pos={self.position}, face={self.facing.name}, plan={self._decided_action.name}"
        for other in self.sim.robots:
            if other.id != self.id:
                other.inbox.append(Message(self.id, payload, self.turn))


class Simulation:
    def __init__(self, robots_config, world: World, max_turns=2):
        self.world = world
        self.max_turns = max_turns

        self.random_mode = False
        self.decisions = {}
        self.valid_pickup_cells = set()
        self.illegal_log = []
        self.valid_log = []

        self.robots = []
        for cfg in robots_config:
            r = Robot(simulation=self, **cfg)
            world.set_robot_state(r.id, r.position, r.facing)
            self.robots.append(r)

    def referee_check(self, turn):
        self.valid_pickup_cells = set()
        pickups_by_cell = {}
        for d in self.decisions.values():
            if d.action == Action.PICKUP:
                pickups_by_cell.setdefault(d.position, []).append(d.robot_id)

        for cell, robot_ids in pickups_by_cell.items():
            has_target = self.world.has_target(cell)
            count = len(robot_ids)

            if not has_target:
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": "pickup attempted on empty cell (no target present)",
                })
            elif count == 1:
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": "only one robot attempted pickup (need exactly two)",
                })
            elif count == 2:
                self.valid_pickup_cells.add(cell)
                self.valid_log.append({"turn": turn, "cell": cell, "robots": robot_ids})
            else:
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": f"{count} robots attempted pickup (more than two)",
                })

    def run(self, renderer=None):
        for turn in range(self.max_turns):
            # if (no more targets) generate random targets
            self.world.maybe_generate_random_targets()
            self.decisions = {}

            # for all robots: read input messages
            for r in self.robots:
                r.turn = turn
                r.read_messages()
            if renderer:
                details = [f"R{r.id}: {len(r.messages_this_turn)} msgs rx'd" for r in self.robots]
                renderer.checkpoint("1 - Read Messages", turn, details=details)

            # for all robots: sense environment
            for r in self.robots:
                r.sense()
            if renderer:
                details = []
                for r in self.robots:
                    msg = f"R{r.id}: "
                    sees = []
                    if r._sensed.get("target_ahead"):
                        sees.append("See Target Ahead")
                    if r._sensed.get("wall_ahead"):
                        sees.append("See Wall")
                    r_ahead = r._sensed.get("robots_ahead")
                    if r_ahead:
                        sees.append(f"See RBs Ahead:{r_ahead}")
                    r_here = r._sensed.get("robots_here")
                    if r_here:
                        sees.append(f"See RBs Here:{r_here}")
                    details.append(msg + (" & ".join(sees) if sees else "Clear"))
                renderer.checkpoint("2 - Sense Environment", turn, details=details)

            # for all robots: decide on action to perform
            for r in self.robots:
                r.decide()
            if renderer:
                acts = [f"R{r.id} plans {r._decided_action.name}" for r in self.robots]
                renderer.checkpoint("3 - Decide Action", turn, details=acts)

            # Test for illegal behavior here
            self.referee_check(turn)
            if renderer:
                renderer.checkpoint("4 - Referee Check", turn, ingest_results=True,
                                    details=["Validating intentions..."])

            # for all robots: perform action
            for r in self.robots:
                r.perform()
            if renderer:
                renderer.checkpoint("5 - Perform Action", turn, details=["Executing valid actions..."])

            # for all robots: send output messages
            for r in self.robots:
                r.send_messages()
            if renderer:
                details = [f"R{r.id} TX: {r._decided_action.name} @ {r.position}" for r in self.robots]
                renderer.checkpoint("6 - Send Messages", turn, details=details)

    def print_summary(self):
        print("\n=== Milestone 1 summary ===")
        print(f"Valid pickups detected:   {len(self.valid_log)}")
        for v in self.valid_log:
            print(f"  turn {v['turn']}: cell {v['cell']} by robots {v['robots']}")
        print(f"Illegal pickups detected: {len(self.illegal_log)}")
        for i in self.illegal_log:
            print(f"  turn {i['turn']}: cell {i['cell']} -> {i['reason']} "
                  f"(robots {i['robots']})")


# =====================================================================
#  Scenario Configuration
# =====================================================================

def parse_path(path_str):
    mapping = {
        'F': Action.MOVE_FORWARD, 'L': Action.TURN_LEFT,
        'R': Action.TURN_RIGHT, 'P': Action.PICKUP,
    }
    return {turn: mapping[char] for turn, char in enumerate(path_str.replace(' ', ''))}


def patrol(loops=1, n=1):
    one_loop = ("F" * n + "R") * 4
    return one_loop * loops


def build_milestone1_scenario():
    world = World(GRID_SIZE)

    world.add_target((3, 3))   # valid A
    world.add_target((2, 7))   # illegal(single) -> valid ทีหลัง
    world.add_target((7, 7))   # illegal(too many) -> valid ทีหลัง
    # (7,2) ไม่มี target -> illegal(empty cell) ของ R3

    world.maybe_generate_random_targets()

    robots_config = [
        dict(robot_id=1, start_pos=(1, 3), start_facing=Direction.EAST,
             script=parse_path(patrol(1) + "F" + "F" + "P" + patrol(10))),
        dict(robot_id=2, start_pos=(3, 1), start_facing=Direction.SOUTH,
             script=parse_path(patrol(1) + "F" + "F" + "P" + patrol(10))),

        dict(robot_id=3, start_pos=(7, 4), start_facing=Direction.NORTH,
             script=parse_path(patrol(2) + "F" + "F" + "P" + patrol(10))),

        dict(robot_id=4, start_pos=(1, 7), start_facing=Direction.EAST,
             script=parse_path(patrol(3) + "F" + "P" + "P" + patrol(10))),
        dict(robot_id=5, start_pos=(2, 5), start_facing=Direction.SOUTH,
             script=parse_path(patrol(3) + "F" + "F" + "P" + patrol(10))),

        dict(robot_id=6, start_pos=(5, 7), start_facing=Direction.EAST,
             script=parse_path(patrol(4) + "F" + "F" + "P" + "P" + patrol(10))),
        dict(robot_id=7, start_pos=(7, 9), start_facing=Direction.NORTH,
             script=parse_path(patrol(4) + "F" + "F" + "P" + "P" + patrol(10))),
        dict(robot_id=8, start_pos=(9, 7), start_facing=Direction.WEST,
             script=parse_path(patrol(4) + "F" + "F" + "P" + "R" + patrol(10))),

        dict(robot_id=9, start_pos=(2, 2), start_facing=Direction.EAST,
             script=parse_path(patrol(10, n=2))),
        dict(robot_id=10, start_pos=(6, 2), start_facing=Direction.EAST,
             script=parse_path(patrol(10, n=2))),
    ]
    return Simulation(robots_config, world, max_turns=50)


# =====================================================================
#  Pygame renderer 
# =====================================================================

CELL = 50
MARGIN = 30
GRID_PX = GRID_SIZE * CELL
SIDEBAR_W = 560
WIN_W = MARGIN * 2 + GRID_PX + SIDEBAR_W
WIN_H = MARGIN * 2 + GRID_PX + 40

COLOR_BG = (14, 16, 21)
COLOR_GRID_BG = (18, 21, 29)
COLOR_GRID_LINE = (42, 47, 59)
COLOR_TEXT = (231, 233, 238)
COLOR_TEXT_DIM = (139, 146, 163)
COLOR_TARGET = (255, 255, 255)
COLOR_VALID = (79, 209, 165)
COLOR_ILLEGAL = (239, 100, 97)
COLOR_PANEL = (22, 25, 34)

ROBOT_COLORS = {
    1: (90, 169, 230),
    2: (239, 100, 97),
    3: (242, 193, 78),
    4: (177, 140, 255),
    5: (107, 207, 127),
    6: (255, 143, 171),
    7: (127, 209, 214),
    8: (255, 167, 38),
    9: (174, 213, 129),
    10: (77, 208, 225),
}

FLASH_DURATION = 0.9


class PygameRenderer:
    def __init__(self, world: World, robots):
        pygame.init()
        pygame.display.set_caption("CPR Milestone 1 - live simulation")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.font_small = pygame.font.SysFont(None, 16)
        self.font_mono = pygame.font.SysFont("consolas,menlo,monospace", 16)

        self.world = world
        self.robots = robots

        self.paused = True
        self.speed = 0.25
        self.phase_label = "ready - press SPACE to start"
        self.turn = 0
        self.flashes = []
        self.log_lines = []
        self.phase_details = []
        self.valid_count = 0
        self.illegal_count = 0
        self._advance_requested = False
        self._restart_requested = False
        self.running = True

    def _pump(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._quit()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    self._advance_requested = True
                elif event.key == pygame.K_UP:
                    self.speed = min(4.0, self.speed + 0.05)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.05, self.speed - 0.05)
                elif event.key == pygame.K_r:
                    self._restart_requested = True
                elif event.key == pygame.K_m:
                    self.sim.random_mode = not self.sim.random_mode

    def _quit(self):
        pygame.quit()
        sys.exit(0)

    def checkpoint(self, label, turn, ingest_results=False, details=None):
        self.phase_label = label
        self.turn = turn
        self.phase_details = details or []
        if ingest_results:
            self._ingest_referee_results(turn)

        self._advance_requested = False
        if self.paused:
            while self.paused and not self._advance_requested:
                self._pump()
                self._draw()
                self.clock.tick(60)
        else:
            phases_per_turn = 6
            target_dt = 1.0 / (self.speed * phases_per_turn)
            start = time.time()
            while time.time() - start < target_dt:
                self._pump()
                self._draw()
                self.clock.tick(60)

    def _ingest_referee_results(self, turn):
        sim_valid = [v for v in self.sim.valid_log if v["turn"] == turn]
        sim_illegal = [i for i in self.sim.illegal_log if i["turn"] == turn]
        now = time.time()
        for v in sim_valid:
            self.flashes.append((v["cell"], COLOR_VALID, now))
            self.log_lines.append((f"turn {turn}  VALID pickup at {v['cell']} "
                                   f"robots {v['robots']}", COLOR_VALID))
            self.valid_count += 1
        for i in sim_illegal:
            self.flashes.append((i["cell"], COLOR_ILLEGAL, now))
            self.log_lines.append((f"turn {turn}  ILLEGAL at {i['cell']}: {i['reason']} "
                                   f"(robots {i['robots']})", COLOR_ILLEGAL))
            self.illegal_count += 1
        self.log_lines = self.log_lines[-10:]

    def simulation_finished(self):
        self.phase_label = "simulation complete - press R to replay, ESC to quit"

    def _draw(self):
        self.screen.fill(COLOR_BG)
        self._draw_grid()
        self._draw_targets()
        self._draw_robots()
        self._draw_sidebar()
        pygame.display.flip()

    def _draw_grid(self):
        rect = pygame.Rect(MARGIN, MARGIN, GRID_PX, GRID_PX)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, rect)

        now = time.time()
        still_active = []
        for cell, color, start in self.flashes:
            age = now - start
            if age > FLASH_DURATION:
                continue
            still_active.append((cell, color, start))
            alpha = int(140 * (1 - age / FLASH_DURATION))
            x, y = cell
            flash_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            flash_surf.fill((*color, alpha))
            self.screen.blit(flash_surf, (MARGIN + x * CELL, MARGIN + y * CELL))
        self.flashes = still_active

        for i in range(GRID_SIZE + 1):
            x = MARGIN + i * CELL
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (x, MARGIN), (x, MARGIN + GRID_PX))
            y = MARGIN + i * CELL
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (MARGIN, y), (MARGIN + GRID_PX, y))

        for i in range(GRID_SIZE):
            label = self.font_small.render(str(i), True, COLOR_TEXT_DIM)
            self.screen.blit(label, (MARGIN + i * CELL + CELL // 2 - 4, MARGIN - 18))
            self.screen.blit(label, (MARGIN - 18, MARGIN + i * CELL + CELL // 2 - 6))

    def _cell_center(self, x, y):
        return (MARGIN + x * CELL + CELL // 2, MARGIN + y * CELL + CELL // 2)

    def _draw_targets(self):
        targets, _, _ = self.world.snapshot()
        for (x, y) in targets:
            cx, cy = self._cell_center(x, y)
            pygame.draw.circle(self.screen, COLOR_TARGET, (cx, cy), 12)

    def _draw_robots(self):
        _, positions, facings = self.world.snapshot()
        local_pts = [(0, -13), (10, 10), (0, 5), (-10, 10)]
        for rid, (x, y) in positions.items():
            cx, cy = self._cell_center(x, y)
            deg = facings[rid].value * 90
            rad = math.radians(deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            pts = []
            for px, py in local_pts:
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                pts.append((cx + rx, cy + ry))
            pygame.draw.polygon(self.screen, ROBOT_COLORS[rid], pts)
            pygame.draw.polygon(self.screen, COLOR_BG, pts, width=1)

            label = self.font_small.render(str(rid), True, ROBOT_COLORS[rid])
            self.screen.blit(label, (cx - label.get_width() // 2, cy + 16))

    def _draw_sidebar(self):
        panel_h = WIN_H - (MARGIN * 2)
        panel = pygame.Rect(MARGIN * 2 + GRID_PX, MARGIN, SIDEBAR_W - MARGIN, panel_h)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel, border_radius=8)
        x0 = panel.x + 16
        y = panel.y + 16

        def line(text, color=COLOR_TEXT, font=None, dy=22):
            nonlocal y
            surf = (font or self.font).render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += dy

        max_t = getattr(self.sim, "max_turns", "?") if hasattr(self, "sim") else "?"
        line(f"turn {self.turn} / {max_t}", COLOR_TEXT_DIM, self.font_mono, 26)
        line(self.phase_label, COLOR_TEXT, self.font_mono, 26)

        line(f"speed: {self.speed:.2f} turns/s   ({'paused' if self.paused else 'playing'})",
             COLOR_TEXT_DIM, self.font_mono, 26)

        mode_str = "RANDOM" if self.sim.random_mode else "SCRIPTED"
        line(f"mode:  {mode_str}", (255, 167, 38) if self.sim.random_mode else COLOR_VALID,
             self.font_mono, 26)

        line(f"valid: {self.valid_count}    illegal: {self.illegal_count}", COLOR_TEXT, self.font, 26)

        line("Current Phase Log:", COLOR_TEXT_DIM, self.font_small, 20)

        details = getattr(self, "phase_details", [])
        start_y_details = y

        for index, d in enumerate(details):
            text_color = (170, 180, 200)
            if d.startswith("R"):
                try:
                    parts = d.split()[0].replace("R", "").replace(":", "")
                    rid = int(parts)
                    if rid in ROBOT_COLORS:
                        text_color = ROBOT_COLORS[rid]
                except Exception:
                    pass
            if index < 5:
                line("  " + d, text_color, self.font_small, 18)
            else:
                surf = self.font_small.render("  " + d, True, text_color)
                self.screen.blit(surf, (x0 + 260, start_y_details + ((index - 5) * 18)))

        if len(details) > 0:
            y = start_y_details + (min(len(details), 5) * 18) + 15

        y = max(y + 10, panel.bottom - 240)
        line("Referee Log:", COLOR_TEXT_DIM, self.font_small, 18)
        for text, color in self.log_lines[-11:]:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += 18


def main():
    while True:
        sim = build_milestone1_scenario()
        renderer = PygameRenderer(sim.world, sim.robots)
        renderer.sim = sim
        renderer._restart_requested = False

        sim.run(renderer=renderer)
        sim.print_summary()
        renderer.simulation_finished()

        while not renderer._restart_requested:
            renderer._pump()
            renderer._draw()
            renderer.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    main()