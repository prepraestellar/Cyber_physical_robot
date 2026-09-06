"""
Cyber Physical Robotics - Milestone 1
======================================
Multi-robot 2D grid simulation that demonstrates illegal-pickup
detection under a concurrent, thread-per-robot, barrier-synchronized
architecture.

Milestone 1 requirements satisfied here:
  - No communication delay (messages are delivered with 0 delay; see
    Robot.send_messages for exactly where M3's exponential delay would
    be inserted later).
  - No "full"/intelligent robot behavior: each robot simply replays a
    pre-defined script of actions (a dict of turn -> Action). This is
    explicitly allowed by the assignment: "It is not necessary that
    robots move intelligently: merely that they move and act in
    pre-defined ways to illustrate the failure cases."
  - The simulation reliably DETECTS all illegal conditions:
      (a) pickup attempted on a cell with no target
      (b) exactly one robot attempts pickup on a cell with a target
      (c) more than two robots attempt pickup on a cell with a target
    ...and correctly does NOT flag the valid case (exactly two robots).

Concurrency model:
  Each robot runs in its own thread. Every phase of the turn structure
  from the assignment's pseudocode ("for all robots: ...") is realized
  as an explicit barrier so all robots execute that phase in parallel
  and wait for each other before the simulation advances:

      read messages -> sense -> decide -> [REFEREE CHECK] -> perform -> send

  The referee check runs once, single-threaded, strictly between the
  decide and perform barriers (robots are held on a gate) -- exactly
  where the assignment's pseudocode says "Test for illegal behavior
  here": after every robot has committed to an action, before any
  robot has actually moved or picked anything up.
"""

import threading
import queue
import random
from dataclasses import dataclass
from enum import Enum, auto


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
    """Snapshot of a robot's decided action, captured at decide-time
    (before perform). Pickup always applies to the robot's CURRENT
    position per the spec, so the referee must validate against this
    pre-move snapshot, not the post-move position."""
    robot_id: int
    action: Action
    position: tuple


class World:
    """Shared grid state: target positions and the robot registry.

    Mutations only happen during the 'perform' phase, which is
    barrier-separated from the 'sense' phase of the next turn, so
    concurrent reads during sensing never race with the writes from
    the previous turn. The lock below is a defensive measure for the
    few operations that touch the shared dicts/sets directly.
    """

    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.targets = set()          # set of (x, y) currently holding a target
        self.robot_positions = {}     # robot_id -> (x, y)
        self.robot_facings = {}       # robot_id -> Direction
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

    def maybe_generate_random_targets(self, count=1):
        """Per spec: 'if no more targets, generate random targets.'
        Only fires when the grid has zero targets left, so it never
        interferes with the Milestone-1 test targets placed at setup."""
        with self.lock:
            if self.targets:
                return
            occupied = set(self.robot_positions.values())
            for _ in range(count):
                for _attempt in range(50):
                    pos = (random.randrange(self.size), random.randrange(self.size))
                    if pos not in occupied and pos not in self.targets:
                        self.targets.add(pos)
                        break


class Simulation:
    def __init__(self, robots_config, world: World, max_turns=4):
        self.world = world
        self.max_turns = max_turns
        self.print_lock = threading.Lock()

        n = len(robots_config)
        # +1 party = this simulation/main thread, so EVERY phase is an
        # explicit barrier across all robots. This is what turns the
        # assignment's sequential-looking pseudocode ("for all robots")
        # into a genuinely parallel, synchronized turn structure.
        self.barrier_read = threading.Barrier(n + 1)
        self.barrier_sense = threading.Barrier(n + 1)
        self.barrier_decide = threading.Barrier(n + 1)
        self.barrier_perform = threading.Barrier(n + 1)
        self.barrier_send = threading.Barrier(n + 1)
        self.perform_gate = threading.Event()

        self.decisions = {}             # robot_id -> Decision, refreshed each turn
        self.valid_pickup_cells = set() # computed by the referee each turn
        self.illegal_log = []
        self.valid_log = []

        self.robots = []
        for cfg in robots_config:
            r = Robot(simulation=self, **cfg)
            world.set_robot_state(r.id, r.position, r.facing)
            self.robots.append(r)

    def log(self, msg):
        with self.print_lock:
            print(msg)

    # ---- referee: single-threaded, runs between decide and perform ----
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
                self.log(f"[turn {turn}] ILLEGAL at {cell}: robots {robot_ids} "
                         f"tried to pick up a target that isn't there")
            elif count == 1:
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": "only one robot attempted pickup (need exactly two)",
                })
                self.log(f"[turn {turn}] ILLEGAL at {cell}: only robot {robot_ids[0]} "
                         f"attempted pickup (needs exactly 2)")
            elif count == 2:
                self.valid_pickup_cells.add(cell)
                self.valid_log.append({"turn": turn, "cell": cell, "robots": robot_ids})
                self.log(f"[turn {turn}] VALID pickup at {cell}: robots {robot_ids}")
            else:  # count > 2
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": f"{count} robots attempted pickup (more than two)",
                })
                self.log(f"[turn {turn}] ILLEGAL at {cell}: {count} robots {robot_ids} "
                         f"all tried to pick up at once (max allowed is 2)")

    # ---- main loop ----------------------------------------------------
    def run(self):
        for r in self.robots:
            r.start()

        for turn in range(self.max_turns):
            self.world.maybe_generate_random_targets()

            # Close the gate BEFORE the barriers so no robot can slip
            # through on a stale "open" state left over from last turn.
            self.perform_gate.clear()

            self.barrier_read.wait()     # phase 1: read messages   (parallel)
            self.barrier_sense.wait()    # phase 2: sense environment (parallel)
            self.barrier_decide.wait()   # phase 3: decide action   (parallel)

            # --- referee runs here: single-threaded, before ANY robot acts ---
            self.referee_check(turn)
            self.perform_gate.set()

            self.barrier_perform.wait()  # phase 5: perform action (parallel)
            self.barrier_send.wait()     # phase 6: send messages  (parallel)

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
        # script: {turn_number: Action}. Missing turns default to IDLE.
        # This is the "pre-defined ways to illustrate the failure cases"
        # explicitly allowed for Milestone 1 -- no decision-making yet.
        self.script = script or {}
        self.inbox = queue.Queue()
        self.messages_this_turn = []
        self.turn = 0
        self._decided_action = Action.IDLE
        self._sensed = {}

    # --- one method per pseudocode phase -------------------------------
    def read_messages(self):
        self.messages_this_turn = []
        while not self.inbox.empty():
            self.messages_this_turn.append(self.inbox.get_nowait())

    def sense(self):
        dx, dy = self.facing.delta()
        front = (self.position[0] + dx, self.position[1] + dy)
        self._sensed = {
            "target_ahead": self.sim.world.has_target(front)
                             if self.sim.world.in_bounds(front) else False,
            "robots_ahead": [rid for rid in self.sim.world.robots_at(front)
                              if rid != self.id],
            "robots_here": [rid for rid in self.sim.world.robots_at(self.position)
                             if rid != self.id],
        }

    def decide(self):
        self._decided_action = self.script.get(self.turn, Action.IDLE)
        # Recorded with the CURRENT (pre-move) position, since pickup
        # always applies to "its current position" per the spec.
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
            # Only actually captures the target if the referee already
            # validated this cell as a legal (exactly-two-robot) pickup.
            if self.position in self.sim.valid_pickup_cells:
                self.sim.world.remove_target(self.position)
        # IDLE: nothing to do.
        self.sim.world.set_robot_state(self.id, self.position, self.facing)

    def send_messages(self):
        # Milestone 1: zero delay -- the message lands directly in every
        # other robot's inbox. It's still only consumed at the START of
        # next turn's read phase, since robots drain their inbox once
        # per turn. (Milestone 3 would replace the direct .put() below
        # with a per-receiver exponential-delay scheduler.)
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

            self.sim.perform_gate.wait()   # held here until referee finishes
            self.perform()
            self.sim.barrier_perform.wait()

            self.send_messages()
            self.sim.barrier_send.wait()


def build_milestone1_scenario():
    """Lays out 7 robots and 3 pre-placed targets on a 10x10 grid so
    that, on turn 1, all four required outcomes happen simultaneously:
      - (2,2): robots 1 & 2       -> VALID pickup
      - (7,2): robot 3, no target -> ILLEGAL (empty cell)
      - (2,7): robot 4 only       -> ILLEGAL (only one robot)
      - (7,7): robots 5, 6 & 7    -> ILLEGAL (more than two robots)
    """
    world = World(GRID_SIZE)

    world.add_target((2, 2))   # valid-pickup cell
    world.add_target((2, 7))   # single-robot illegal cell
    world.add_target((7, 7))   # too-many-robots illegal cell
    # (7, 2) deliberately left empty -> triggers the "empty cell" case.

    robots_config = [
        # --- Scenario A: VALID pickup, 2 robots converge on (2,2) ---
        dict(robot_id=1, start_pos=(1, 2), start_facing=Direction.EAST,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
        dict(robot_id=2, start_pos=(2, 1), start_facing=Direction.SOUTH,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),

        # --- Scenario B: ILLEGAL, pickup attempted on empty cell (7,2) ---
        dict(robot_id=3, start_pos=(6, 2), start_facing=Direction.EAST,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),

        # --- Scenario C: ILLEGAL, only 1 robot on target cell (2,7) ---
        dict(robot_id=4, start_pos=(1, 7), start_facing=Direction.EAST,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),

        # --- Scenario D: ILLEGAL, 3 robots converge on target cell (7,7) ---
        dict(robot_id=5, start_pos=(6, 7), start_facing=Direction.EAST,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
        dict(robot_id=6, start_pos=(7, 6), start_facing=Direction.SOUTH,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
        dict(robot_id=7, start_pos=(8, 7), start_facing=Direction.WEST,
             script={0: Action.MOVE_FORWARD, 1: Action.PICKUP}),
    ]

    return Simulation(robots_config, world, max_turns=4)


if __name__ == "__main__":
    sim = build_milestone1_scenario()
    sim.run()
    sim.print_summary()