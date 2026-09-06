"""
Cyber Physical Robotics - Milestone 1 (with a live pygame visualization)
=========================================================================
This is the SAME concurrent simulation as cpr_milestone1.py -- real
Python threads, one per robot, synchronized every phase by
threading.Barrier -- with a pygame window attached so you can watch it
run: robots drawn as colored arrows (pointing the way they're facing),
targets as amber circles, cells flashing green/red when the referee
validates or rejects a pickup attempt.

Install once:
    pip install pygame

Run:
    python cpr_milestone1_pygame.py

Controls (also printed to the console on startup):
    SPACE        play / pause
    RIGHT ARROW  step forward one phase (works while paused)
    UP / DOWN    speed up / slow down
    R            restart the scenario (once it has finished)
    ESC or Q     quit
"""

import math
import sys
import time
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
import pygame

GRID_SIZE = 10


# =====================================================================
#  Simulation core -- identical in spirit to cpr_milestone1.py.
#  Robots are real threads; every phase is an explicit barrier; the
#  referee runs single-threaded strictly between decide and perform.
# =====================================================================

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
    IDLE = auto()
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
    """Shared grid state. Mutated only during 'perform', which is
    barrier-separated from the next turn's 'sense' -- so sensing never
    races with the previous turn's writes. The lock is a defensive
    measure for the handful of shared-dict/set operations."""

    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.targets = set()
        self.robot_positions = {}
        self.robot_facings = {}
        self.lock = threading.Lock()

    def has_target(self, pos):
        with self.lock:
            return pos in self.targets

    def remove_target(self, pos):
        with self.lock:
            self.targets.discard(pos)

    def add_target(self, pos):
        with self.lock:
            self.targets.add(pos)

    def robots_at(self, pos):
        with self.lock:
            return [rid for rid, p in self.robot_positions.items() if p == pos]

    def set_robot_state(self, robot_id, pos, facing):
        with self.lock:
            self.robot_positions[robot_id] = pos
            self.robot_facings[robot_id] = facing

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def snapshot(self):
        """Cheap, consistent read for the renderer. Safe to call any
        time the renderer is invoked, since it only runs at points
        where every robot thread is blocked on a barrier or gate."""
        with self.lock:
            return set(self.targets), dict(self.robot_positions), dict(self.robot_facings)


class Simulation:
    def __init__(self, robots_config, world: World, max_turns=2):
        self.world = world
        self.max_turns = max_turns

        n = len(robots_config)
        # +1 party = this simulation/main thread, so every phase is an
        # explicit barrier across all robots -- genuinely parallel and
        # synchronized, not a sequential for-loop.
        self.barrier_read = threading.Barrier(n + 1)
        self.barrier_sense = threading.Barrier(n + 1)
        self.barrier_decide = threading.Barrier(n + 1)
        self.barrier_perform = threading.Barrier(n + 1)
        self.barrier_send = threading.Barrier(n + 1)
        self.perform_gate = threading.Event()

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
        for r in self.robots:
            r.start()

        for turn in range(self.max_turns):
            self.perform_gate.clear()

            self.barrier_read.wait()
            if renderer:
                renderer.checkpoint("1 - read messages", turn)

            self.barrier_sense.wait()
            if renderer:
                renderer.checkpoint("2 - sense environment", turn)

            self.barrier_decide.wait()
            if renderer:
                renderer.checkpoint("3 - decide action", turn)

            # --- referee runs here: single-threaded, before ANY robot acts ---
            self.referee_check(turn)
            self.perform_gate.set()
            if renderer:
                renderer.checkpoint("4 - referee check", turn, ingest_results=True)

            self.barrier_perform.wait()
            if renderer:
                renderer.checkpoint("5 - perform action", turn)

            self.barrier_send.wait()
            if renderer:
                renderer.checkpoint("6 - send messages", turn)

        for r in self.robots:
            r.join()

    def print_summary(self):
        print("\n=== Milestone 1 summary ===")
        print(f"Valid pickups detected:   {len(self.valid_log)}")
        for v in self.valid_log:
            print(f"  turn {v['turn']}: cell {v['cell']} by robots {v['robots']}")
        print(f"Illegal pickups detected: {len(self.illegal_log)}")
        for i in self.illegal_log:
            print(f"  turn {i['turn']}: cell {i['cell']} -> {i['reason']} "
                  f"(robots {i['robots']})")


class Robot(threading.Thread):
    def __init__(self, simulation: Simulation, robot_id, start_pos, start_facing,
                 script=None):
        super().__init__(daemon=True, name=f"Robot-{robot_id}")
        self.sim = simulation
        self.id = robot_id
        self.position = start_pos
        self.facing = start_facing
        self.script = script or {}
        self.inbox = queue.Queue()
        self.messages_this_turn = []
        self.turn = 0
        self._decided_action = Action.IDLE

    def read_messages(self):
        self.messages_this_turn = []
        while not self.inbox.empty():
            self.messages_this_turn.append(self.inbox.get_nowait())

    def sense(self):
        pass  # not needed for the scripted Milestone-1 demo

    def decide(self):
        self._decided_action = self.script.get(self.turn, Action.IDLE)
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
        payload = f"pos={self.position},facing={self.facing.name}"
        for other in self.sim.robots:
            if other.id != self.id:
                other.inbox.put(Message(self.id, payload, self.turn))

    def run(self):
        for turn in range(self.sim.max_turns):
            self.turn = turn

            self.read_messages()
            self.sim.barrier_read.wait()

            self.sense()
            self.sim.barrier_sense.wait()

            self.decide()
            self.sim.barrier_decide.wait()

            self.sim.perform_gate.wait()
            self.perform()
            self.sim.barrier_perform.wait()

            self.send_messages()
            self.sim.barrier_send.wait()


# def build_milestone1_scenario():
#     world = World(GRID_SIZE)
#     world.add_target((2, 2))
#     world.add_target((2, 7))
#     world.add_target((7, 7))
#     # (7, 2) deliberately left empty -> triggers the "empty cell" case.

#     robots_config = [
#         dict(robot_id=1, start_pos=(1, 2), start_facing=Direction.EAST,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#         dict(robot_id=2, start_pos=(2, 1), start_facing=Direction.SOUTH,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#         dict(robot_id=3, start_pos=(6, 2), start_facing=Direction.EAST,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#         dict(robot_id=4, start_pos=(1, 7), start_facing=Direction.EAST,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#         dict(robot_id=5, start_pos=(6, 7), start_facing=Direction.EAST,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#         dict(robot_id=6, start_pos=(7, 6), start_facing=Direction.SOUTH,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#         dict(robot_id=7, start_pos=(8, 7), start_facing=Direction.WEST,
#              script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
#     ]
#     return Simulation(robots_config, world, max_turns=2)

def build_milestone1_scenario():
    world = World(GRID_SIZE)
    # วาง Target ไว้ 3 จุดสำหรับแต่ละเคส
    world.add_target((2, 2))  # สำหรับ Case 1 (Valid)
    # (7, 2) จงใจไม่วาง Target เพื่อเทส Case 2 (Empty Cell)
    world.add_target((2, 7))  # สำหรับ Case 3 (1 Robot)
    world.add_target((7, 7))  # สำหรับ Case 4 (3 Robots)

    # จัดตารางเวลา (Script) ให้หุ่นยนต์แต่ละกลุ่มทำงานคนละ Turn กัน
    # หากยังไม่ถึง Turn ของตัวเอง หุ่นยนต์จะทำสถานะ Action.IDLE (ยืนรอ) โดยอัตโนมัติ
    robots_config = [
        # =========================================================
        # [Turn 0-1] Scenario 1: VALID pickup (หุ่น 2 ตัวหยิบพร้อมกัน)
        # =========================================================
        dict(robot_id=1, start_pos=(1, 2), start_facing=Direction.EAST,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
        dict(robot_id=2, start_pos=(2, 1), start_facing=Direction.SOUTH,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),

        # =========================================================
        # [Turn 4-5] Scenario 2: ILLEGAL (หยิบอากาศบนช่องที่ไม่มี Target)
        # =========================================================
        dict(robot_id=3, start_pos=(6, 2), start_facing=Direction.EAST,
             script={4: Action.MOVE_FORWARD, 5: Action.PICKUP}),

        # =========================================================
        # [Turn 8-9] Scenario 3: ILLEGAL (มีหุ่นแค่ 1 ตัว พยายามหยิบ Target)
        # =========================================================
        dict(robot_id=4, start_pos=(1, 7), start_facing=Direction.EAST,
             script={8: Action.MOVE_FORWARD, 9: Action.PICKUP}),

        # =========================================================
        # [Turn 12-13] Scenario 4: ILLEGAL (หุ่น 3 ตัวแย่งหยิบ Target พร้อมกัน)
        # =========================================================
        dict(robot_id=5, start_pos=(6, 7), start_facing=Direction.EAST,
             script={12: Action.MOVE_FORWARD, 13: Action.PICKUP}),
        dict(robot_id=6, start_pos=(7, 6), start_facing=Direction.SOUTH,
             script={12: Action.MOVE_FORWARD, 13: Action.PICKUP}),
        dict(robot_id=7, start_pos=(8, 7), start_facing=Direction.WEST,
             script={12: Action.MOVE_FORWARD, 13: Action.PICKUP}),
    ]
    
    # เพิ่ม max_turns เป็น 16 เพื่อให้ครอบคลุมเวลาตั้งแต่ Turn 0 ถึง 15
    return Simulation(robots_config, world, max_turns=16)


# =====================================================================
#  Pygame renderer
# =====================================================================

CELL = 50
MARGIN = 30
GRID_PX = GRID_SIZE * CELL
SIDEBAR_W = 340
WIN_W = MARGIN * 2 + GRID_PX + SIDEBAR_W
WIN_H = MARGIN * 2 + GRID_PX

COLOR_BG = (14, 16, 21)
COLOR_GRID_BG = (18, 21, 29)
COLOR_GRID_LINE = (42, 47, 59)
COLOR_TEXT = (231, 233, 238)
COLOR_TEXT_DIM = (139, 146, 163)
COLOR_TARGET = (242, 193, 78)
COLOR_VALID = (79, 209, 165)
COLOR_ILLEGAL = (239, 100, 97)
COLOR_PANEL = (22, 25, 34)

ROBOT_COLORS = {
    1: (90, 169, 230), 2: (239, 100, 97), 3: (242, 193, 78), 4: (177, 140, 255),
    5: (107, 207, 127), 6: (255, 143, 171), 7: (127, 209, 214),
}

FLASH_DURATION = 0.9  # seconds


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
        self.speed = 1.0          # turns per second
        self.phase_label = "ready - press SPACE to start"
        self.turn = 0
        self.flashes = []          # list of (cell, color, start_time)
        self.log_lines = []
        self.valid_count = 0
        self.illegal_count = 0
        self._advance_requested = False
        self._seen_illegal = 0
        self._seen_valid = 0
        self.running = True

    # ---- event handling -------------------------------------------
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
                    self.speed = min(4.0, self.speed + 0.25)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.25, self.speed - 0.25)
                elif event.key == pygame.K_r:
                    self._restart_requested = True

    def _quit(self):
        pygame.quit()
        sys.exit(0)

    # ---- called by Simulation.run() between phases -----------------
    def checkpoint(self, label, turn, ingest_results=False):
        self.phase_label = label
        self.turn = turn
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

    # ---- drawing -----------------------------------------------------
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
        panel = pygame.Rect(MARGIN * 2 + GRID_PX, MARGIN, SIDEBAR_W - MARGIN, GRID_PX)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel, border_radius=8)
        x0 = panel.x + 16
        y = panel.y + 16

        def line(text, color=COLOR_TEXT, font=None, dy=22):
            nonlocal y
            surf = (font or self.font).render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += dy

        max_t = getattr(self.sim, "max_turns", "?") if hasattr(self, "sim") else "?"
        line(f"turn {self.turn} / {max_t}", COLOR_TEXT_DIM, self.font_mono, 4)
        line(self.phase_label, COLOR_TEXT, self.font_mono, 26)

        line(f"speed: {self.speed:.2f} turns/s   ({'paused' if self.paused else 'playing'})",
             COLOR_TEXT_DIM, self.font_mono, 26)

        line(f"valid: {self.valid_count}    illegal: {self.illegal_count}", COLOR_TEXT, self.font, 28)

        line("referee log", COLOR_TEXT_DIM, self.font_small, 20)
        for text, color in self.log_lines[-9:]:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += 18

        y = panel.bottom - 90
        for text in ["SPACE  play / pause", "-> step forward (while paused)",
                     "UP/DOWN  speed", "R  restart   ESC/Q  quit"]:
            surf = self.font_small.render(text, True, COLOR_TEXT_DIM)
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

        # keep the window open showing the final state until the user
        # asks to restart or quit
        while not renderer._restart_requested:
            renderer._pump()
            renderer._draw()
            renderer.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    main()