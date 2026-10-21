import argparse
import colorsys
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

try:
    import pygame
except ImportError:          # headless runs do not need pygame
    pygame = None

GRID_SIZE = 10
STALE_AFTER = 20       # a cell looked at less than this many turns ago is "recent"
NEAR_OFFSETS = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 2]


# =====================================================================
#  Basic types
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
    MOVE_FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    PICKUP = auto()


class RobotState(Enum):
    SEARCH = auto()       # looking for targets
    COMMITTED = auto()    # committed to pick up the target at `target`


@dataclass
class Message:
    sender_id: int
    payload: dict
    turn_sent: int


@dataclass
class Decision:
    robot_id: int
    action: Action
    position: tuple


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# =====================================================================
#  World
# =====================================================================

class World:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.targets = set()
        self.robot_positions = {}
        self.robot_facings = {}
        self.generated_count = 0
        self.batch_size = 5

    def has_target(self, pos):
        return pos in self.targets

    def remove_target(self, pos):
        self.targets.discard(pos)

    def add_target(self, pos):
        if pos not in self.targets:
            self.targets.add(pos)
            self.generated_count += 1

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

    def apply_action(self, robot_id, action, valid_pickup_cells):
        """The PHYSICS: the robot only commands an action, the world decides what
        happens (it refuses to move a robot out of the grid, and it removes a
        target only if the referee found the pickup valid)."""
        pos = self.robot_positions[robot_id]
        facing = self.robot_facings[robot_id]
        if action == Action.MOVE_FORWARD:
            dx, dy = facing.delta()
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.in_bounds(new_pos):
                pos = new_pos
        elif action == Action.TURN_LEFT:
            facing = facing.turn_left()
        elif action == Action.TURN_RIGHT:
            facing = facing.turn_right()
        elif action == Action.PICKUP:
            if pos in valid_pickup_cells:
                self.remove_target(pos)
        self.set_robot_state(robot_id, pos, facing)

    def maybe_generate_random_targets(self, count=None):
        """If there are no more targets, generate `count` random ones
        (never on a cell occupied by a robot)."""
        if len(self.targets) > 0:
            return
        count = count or self.batch_size
        occupied = set(self.robot_positions.values())
        for _ in range(count):
            for _attempt in range(50):
                pos = (random.randrange(self.size), random.randrange(self.size))
                if pos not in occupied and pos not in self.targets:
                    self.add_target(pos)
                    break


# =====================================================================
#  Robot
# =====================================================================

class Robot:
    """A robot knows NOTHING about the simulation or the world. It only has:
         * its own variables (pose estimate, memory, inbox)
         * sensor readings that are handed to it:      sense(reading)
         * messages that are handed to it:             receive(msg)
       and it only produces:
         * an action it wants to perform:              decide() -> Action
         * a message it wants to broadcast:            get_outgoing_message()
       The pose (position, facing) is the robot's OWN estimate (dead reckoning:
       start pose + the actions it commanded). The true pose lives in World."""

    def __init__(self, robot_id, start_pos, start_facing, arena_size=GRID_SIZE,
                 search_mode="coverage"):
        self.id = robot_id
        self.position = start_pos          # own pose estimate (dead reckoning)
        self.facing = start_facing
        self.arena_size = arena_size       # prior knowledge: size of the arena
        self.search_mode = search_mode     # "coverage" (default) or "random" walk
        self.random_policy = False         # baseline controller (no coordination)

        self.inbox = deque()
        self.messages_this_turn = []
        self.turn = -1                     # own clock, +1 at every "read messages"

        # controller memory
        self.state = RobotState.SEARCH
        self.target = None                 # cell I am committed to
        self.captured = None               # cell I picked up last turn (broadcast once)
        self.known = set()                 # targets I know about (not yet captured)
        self.peers = {}                    # latest status received from each other robot
        self._peer_turn = {}
        self.last_seen = {}                # cell -> turn when ANY robot last looked at it
        self.looked = []                   # cells I looked at this turn (broadcast)
        self.goal = None                   # cell I am exploring towards (SEARCH only)
        self.goal_turn = 0

        self._decided_action = Action.TURN_RIGHT
        self._sensed = {}

    # ---------------- communication in ----------------
    def receive(self, msg):
        """Called by the network (the simulation) when a message arrives."""
        self.inbox.append(msg)

    # ---------------- phase: read messages ----------------
    def read_messages(self):
        self.turn += 1
        self.messages_this_turn = []
        while self.inbox:
            msg = self.inbox.popleft()
            self.messages_this_turn.append(msg)
            # keep the newest status of each sender
            if msg.turn_sent >= self._peer_turn.get(msg.sender_id, -1):
                self.peers[msg.sender_id] = msg.payload
                self._peer_turn[msg.sender_id] = msg.turn_sent

    # ---------------- phase: sense ----------------
    def sense(self, reading):
        """`reading` comes from the robot's sensors (computed by the simulation):
        {"target_ahead": bool, "robots_ahead": [ids], "robots_here": [ids]}
        - exactly the three sensors listed in the project description."""
        front = self._front_cell()
        self._sensed = reading
        # remember (and tell the others) which cells I have just looked at:
        # the cell in front (the only one I can sense) and the one I stand on
        self.looked = [self.position]
        if not self._wall_ahead():
            self.looked.append(front)
        for c in self.looked:
            self.last_seen[c] = self.turn

    # ---------------- helpers that use ONLY own pose ----------------
    def _front_cell(self):
        dx, dy = self.facing.delta()
        return (self.position[0] + dx, self.position[1] + dy)

    def _wall_ahead(self):
        x, y = self._front_cell()
        return not (0 <= x < self.arena_size and 0 <= y < self.arena_size)

    def _status(self):
        return {
            "pos": self.position,
            "facing": self.facing.name,
            "state": self.state,
            "target": self.target,
            "captured": self.captured,
            "looked": list(self.looked),
            "goal": self.goal if self.state == RobotState.SEARCH else None,
        }

    def _snapshot(self):
        """Everybody's status as of the end of the previous turn
        (other robots from messages, myself from my own memory)."""
        snap = {rid: dict(st) for rid, st in self.peers.items()}
        snap[self.id] = self._status()
        return snap

    # ---------------- phase: decide ----------------
    def decide(self):
        """Returns the action the robot wants to perform (it does not perform it)."""
        if self.random_policy:
            # baseline: no coordination at all (used to compare against the controller)
            self.state, self.target, self.captured = RobotState.SEARCH, None, None
            action = random.choice(list(Action))
        else:
            action = self._controller()

        if action == Action.MOVE_FORWARD and self._wall_ahead():
            action = random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])

        self._decided_action = action
        return action

    def _controller(self):
        snap = self._snapshot()
        start_state, start_target = self.state, self.target   # state at the START of this turn
        self.captured = None                                  # "captured" is broadcast for one turn only

        # --- 0a. shared "last seen" map: merge what the others looked at ---
        for rid, st in self.peers.items():
            for c in st.get("looked", []):
                if self.turn - 1 > self.last_seen.get(c, -1):
                    self.last_seen[c] = self.turn - 1

        # --- 0b. update the set of known targets ----------------------------
        for st in snap.values():
            if st["target"] is not None:
                self.known.add(st["target"])
        for st in snap.values():
            if st["captured"] is not None:
                self.known.discard(st["captured"])

        # who is committed to which cell (from the shared snapshot)
        committers = {}
        for rid, st in snap.items():
            if st["state"] == RobotState.COMMITTED and st["target"] is not None:
                committers.setdefault(st["target"], []).append(rid)
        for ids in committers.values():
            ids.sort()

        alone = self.turn > 0 and len(snap) == 1      # no other robot has ever been heard

        # --- 1. release my commitment if it is no longer valid ----------
        if start_state == RobotState.COMMITTED:
            T = start_target
            eligible = committers.get(T, [])[:2]        # only the 2 lowest IDs may pick up
            if self.id not in eligible or T not in self.known or alone:
                self.state, self.target = RobotState.SEARCH, None

        # --- 2. (re)assign helpers: same computation on every robot ------
        assignments = self._compute_assignments(snap, committers)
        if self.id in assignments:
            # either I am a free robot chosen as helper, or I am a lower-priority
            # half-committed owner that defects to a higher-priority target
            self.state = RobotState.COMMITTED
            self.target = assignments[self.id]

        # --- 3. found a new target right in front of me -------------------
        if self.state == RobotState.SEARCH and self._sensed.get("target_ahead") and not alone:
            T = self._front_cell()
            already = len(committers.get(T, [])) + sum(1 for t in assignments.values() if t == T)
            if already < 2:
                self.state, self.target = RobotState.COMMITTED, T
                self.known.add(T)

        # --- 4. choose the action ----------------------------------------
        if self.state == RobotState.COMMITTED:
            self.goal = None                  # no exploring while committed
            T = self.target
            if self.position != T:
                return self._navigate(T)

            # I am standing on my target: pick up only if I was already
            # committed to this cell at the start of the turn and my partner
            # (the other eligible robot) is in this cell with me.
            if start_state == RobotState.COMMITTED and start_target == T:
                eligible = committers.get(T, [])[:2]
                if len(eligible) == 2 and self.id in eligible:
                    partner = eligible[0] if eligible[1] == self.id else eligible[1]
                    if partner in self._sensed.get("robots_here", []):
                        self.captured = T
                        self.known.discard(T)
                        self.state, self.target = RobotState.SEARCH, None
                        return Action.PICKUP
            # wait in place (there is no "idle" action): look left, right, left, ...
            return Action.TURN_LEFT if self.turn % 2 == 0 else Action.TURN_RIGHT

        return self._search(snap)

    def _compute_assignments(self, snap, committers):
        """Deterministic helper assignment from the shared snapshot.
        Returns {robot_id: target_cell}. Every robot computes the same result.

        Slots that need a robot:
            - a target with ONE committed robot needs 1 more   (tier 0)
            - a known target nobody is committed to needs 2     (tier 1)
        Free (SEARCH) robots are matched to slots NEAREST-FIRST: repeatedly take
        the (robot, target) pair with the smallest Manhattan distance
        (half-committed targets first, then ties -> lower cell, lower ID).

        If a half-committed target is still missing its partner and there is no
        free robot left, the owner of a LOWER-priority half-committed target
        defects to it (this breaks the "everybody waits alone" deadlock).
        """
        pos = {rid: st["pos"] for rid, st in snap.items()}
        free = {rid for rid, st in snap.items() if st["state"] == RobotState.SEARCH}

        partial = sorted(T for T, ids in committers.items() if len(ids) == 1)
        empty = sorted(T for T in self.known if T not in committers)
        slots = {T: 1 for T in partial}
        slots.update({T: 2 for T in empty})
        tier = {T: 0 for T in partial}
        tier.update({T: 1 for T in empty})

        assignments = {}
        while free and any(k > 0 for k in slots.values()):
            _, _, T, r = min((tier[T], manhattan(pos[r], T), T, r)
                             for T, k in slots.items() if k > 0 for r in free)
            assignments[r] = T
            free.discard(r)
            slots[T] -= 1

        # deadlock breaking among half-committed targets
        unfilled = sorted((T for T in partial if slots[T] > 0),
                          key=lambda T: (committers[T][0], T))
        owner_target = {committers[T][0]: T for T in unfilled}
        abandoned = set()
        for i, T in enumerate(unfilled):
            if T in abandoned:
                continue
            owners = [committers[T2][0] for T2 in unfilled[i + 1:] if T2 not in abandoned]
            if owners:
                pick = min(owners, key=lambda r: (manhattan(pos[r], T), r))
                assignments[pick] = T
                abandoned.add(owner_target[pick])
        return assignments

    def _first_heading(self, T):
        """Direction of the first step of the (shortest, fewest-turns) path to T:
        along x first, then along y."""
        dx = T[0] - self.position[0]
        dy = T[1] - self.position[1]
        if dx != 0:
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _navigate(self, T):
        want = self._first_heading(T)
        if self.facing == want:
            return Action.MOVE_FORWARD
        diff = (want.value - self.facing.value) % 4
        return Action.TURN_LEFT if diff == 3 else Action.TURN_RIGHT

    def _search(self, snap):
        if self.search_mode == "random":
            return self._random_walk()
        return self._explore(snap)

    def _random_walk(self):
        """Random walk that prefers to keep going straight."""
        if self._wall_ahead():
            return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        if random.random() < 0.15:
            return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        return Action.MOVE_FORWARD

    def _explore(self, snap):
        """Coverage search with natural-looking movement.

        A robot can only sense the cell in front of it, so it cannot know where
        targets are before looking. It keeps a shared map of 'when was this cell
        last looked at' (everybody broadcasts the cells it looked at) and picks
        the CHEAPEST cell to go and look at next, where the cost is
            distance to the cell
          + quarter-turns needed to face it        (prefers to keep its heading)
          + penalty for cells looked at recently   (prefers cells nobody has seen lately)
          + penalty for cells near other robots    (spreads out, no crowding)
          + a little randomness                    (no two robots walk identical paths)
        Cells other robots are already heading to are skipped."""
        taken = {}
        occupied = {}
        for rid, st in snap.items():
            if rid == self.id:
                continue
            g = st.get("goal")
            if g is not None:
                taken[g] = min(rid, taken.get(g, rid))
            occupied[st["pos"]] = occupied.get(st["pos"], 0) + 1

        need_new = (self.goal is None
                    or self.position == self.goal
                    or self.last_seen.get(self.goal, -1) > self.goal_turn
                    or (self.goal in taken and taken[self.goal] < self.id))
        if need_new:
            best = None
            for x in range(self.arena_size):
                for y in range(self.arena_size):
                    c = (x, y)
                    if c == self.position or c in taken:
                        continue
                    age = self.turn - self.last_seen.get(c, -10 ** 6)       # never seen = very old
                    recency = max(0, STALE_AFTER - age) * 0.5
                    d = (self._first_heading(c).value - self.facing.value) % 4
                    quarter_turns = {0: 0, 1: 1, 2: 2, 3: 1}[d]
                    crowd = sum(occupied.get((x + dx, y + dy), 0) * (3 - abs(dx) - abs(dy))
                                for dx, dy in NEAR_OFFSETS)
                    cost = (manhattan(self.position, c) + quarter_turns + recency
                            + crowd + random.random() * 0.8)
                    if best is None or (cost, c) < best:
                        best = (cost, c)
            if best is None:
                self.goal = None
                return self._random_walk()
            self.goal, self.goal_turn = best[1], self.turn
        return self._navigate(self.goal)

    # ---------------- after the world executed my action ----------------
    def update_odometry(self, action):
        """Dead reckoning: update MY OWN pose estimate from the action I commanded.
        (The robot never commands a move into a wall, because it knows its pose and
        the arena size, so the estimate stays equal to the true pose.)"""
        if action == Action.MOVE_FORWARD and not self._wall_ahead():
            self.position = self._front_cell()
        elif action == Action.TURN_LEFT:
            self.facing = self.facing.turn_left()
        elif action == Action.TURN_RIGHT:
            self.facing = self.facing.turn_right()

    # ---------------- phase: send ----------------
    def get_outgoing_message(self):
        """What I broadcast this turn. I do not know who receives it (or how many
        robots exist): delivery is the job of the network (the simulation)."""
        return self._status()


# =====================================================================
#  Simulation
# =====================================================================

class Simulation:
    """Plays three roles that a robot must NOT be able to play itself:
         * the sensors   (sensor_reading)   - turns the true world into sensor data
         * the physics   (World.apply_action) - decides what really happens
         * the network   (route_messages)   - delivers broadcasts to the other robots
       plus the referee (referee_check)."""

    def __init__(self, robots_config, world: World, max_turns=500, search_mode="coverage"):
        self.world = world
        self.max_turns = max_turns

        self.decisions = {}
        self.valid_pickup_cells = set()
        self.illegal_log = []
        self.valid_log = []

        self.robots = []
        for cfg in robots_config:
            r = Robot(arena_size=world.size, search_mode=search_mode, **cfg)
            world.set_robot_state(r.id, r.position, r.facing)
            self.robots.append(r)

        self._random_mode = False
        self._search_mode = search_mode

    @property
    def random_mode(self):
        return self._random_mode

    @random_mode.setter
    def random_mode(self, flag):
        self._random_mode = flag
        for r in self.robots:
            r.random_policy = flag

    @property
    def search_mode(self):
        return self._search_mode

    @search_mode.setter
    def search_mode(self, mode):
        self._search_mode = mode
        for r in self.robots:
            r.search_mode = mode

    # ---- sensors: the only window a robot has on the world -------------
    def sensor_reading(self, robot_id):
        w = self.world
        pos, facing = w.robot_positions[robot_id], w.robot_facings[robot_id]
        dx, dy = facing.delta()
        front = (pos[0] + dx, pos[1] + dy)
        return {
            "target_ahead": w.has_target(front) if w.in_bounds(front) else False,
            "robots_ahead": [r for r in w.robots_at(front) if r != robot_id],
            "robots_here": [r for r in w.robots_at(pos) if r != robot_id],
        }

    # ---- network: broadcast to every other robot (ideal: no delay) -----
    def route_messages(self):
        for sender in self.robots:
            payload = sender.get_outgoing_message()
            for receiver in self.robots:
                if receiver is not sender:
                    receiver.receive(Message(sender.id, dict(payload), sender.turn))

    def referee_check(self, turn):
        """Tests the illegal conditions from the project description."""
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
                r.read_messages()
            if renderer:
                details = [f"R{r.id}: {len(r.messages_this_turn)} msgs rx'd" for r in self.robots]
                renderer.checkpoint("1 - Read Messages", turn, details=details)

            # for all robots: sense environment
            for r in self.robots:
                r.sense(self.sensor_reading(r.id))
            if renderer:
                details = []
                for r in self.robots:
                    sees = []
                    if r._sensed.get("target_ahead"):
                        sees.append("Target Ahead")
                    if r._sensed.get("robots_ahead"):
                        sees.append(f"RBs Ahead:{r._sensed['robots_ahead']}")
                    if r._sensed.get("robots_here"):
                        sees.append(f"RBs Here:{r._sensed['robots_here']}")
                    details.append(f"R{r.id}: " + (" & ".join(sees) if sees else "Clear"))
                renderer.checkpoint("2 - Sense Environment", turn, details=details)

            # for all robots: decide on action to perform
            for r in self.robots:
                action = r.decide()
                # the referee records the intention with the TRUE position of the robot
                self.decisions[r.id] = Decision(r.id, action, self.world.robot_positions[r.id])
            if renderer:
                details = []
                for r in self.robots:
                    tag = "SRCH" if r.state == RobotState.SEARCH else f"->{r.target}"
                    details.append(f"R{r.id} {tag} {r._decided_action.name}")
                renderer.checkpoint("3 - Decide Action", turn, details=details)

            # Test for illegal behavior here
            self.referee_check(turn)
            if renderer:
                renderer.checkpoint("4 - Referee Check", turn, ingest_results=True,
                                    details=["Validating intentions..."])

            # for all robots: perform action
            for r in self.robots:
                action = self.decisions[r.id].action
                self.world.apply_action(r.id, action, self.valid_pickup_cells)   # physics
                r.update_odometry(action)                                         # robot's own estimate
            if renderer:
                renderer.checkpoint("5 - Perform Action", turn, details=["Executing valid actions..."])

            # for all robots: send output messages
            self.route_messages()
            if renderer:
                details = [f"R{r.id} TX @ {r.position}" for r in self.robots]
                renderer.checkpoint("6 - Send Messages", turn, details=details)

    def print_summary(self):
        print("\n=== Milestone 2 summary ===")
        print(f"Robots:                    {len(self.robots)}")
        print(f"Turns simulated:           {self.max_turns}")
        print(f"Targets generated:         {self.world.generated_count}")
        print(f"Valid pickups (captured):  {len(self.valid_log)}")
        print(f"Targets still on the grid: {len(self.world.targets)}")
        print(f"Illegal pickups detected:  {len(self.illegal_log)}")
        for i in self.illegal_log:
            print(f"  turn {i['turn']}: cell {i['cell']} -> {i['reason']} (robots {i['robots']})")
        if self.valid_log:
            print(f"Average turns per capture: {self.max_turns / len(self.valid_log):.1f}")


# =====================================================================
#  Scenario
# =====================================================================

def build_scenario(num_robots=10, max_turns=500, search_mode="coverage", targets_per_batch=None):
    """Any number of robots (1..100, unique IDs), random distinct start cells and
    random initial facing. (With a single robot nothing can ever be captured,
    because a pickup needs exactly two robots - but it must stay legal.)"""
    if not 1 <= num_robots <= GRID_SIZE * GRID_SIZE:
        raise ValueError("number of robots must be between 1 and 100")
    world = World(GRID_SIZE)
    cells = random.sample([(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)], num_robots)
    robots_config = [
        dict(robot_id=i + 1, start_pos=cells[i], start_facing=random.choice(list(Direction)))
        for i in range(num_robots)
    ]
    sim = Simulation(robots_config, world, max_turns=max_turns, search_mode=search_mode)
    # how many targets appear each time the grid is empty (the description leaves it open)
    world.batch_size = targets_per_batch or max(5, (num_robots + 1) // 2)
    world.maybe_generate_random_targets()
    return sim


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
COLOR_COMMITTED = (255, 214, 102)

ROBOT_COLORS = {
    1: (90, 169, 230), 2: (239, 100, 97), 3: (242, 193, 78), 4: (177, 140, 255),
    5: (107, 207, 127), 6: (255, 143, 171), 7: (127, 209, 214), 8: (255, 167, 38),
    9: (174, 213, 129), 10: (77, 208, 225),
}


def robot_color(rid):
    if rid in ROBOT_COLORS:
        return ROBOT_COLORS[rid]
    h = (rid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


FLASH_DURATION = 0.9


class PygameRenderer:
    def __init__(self, world: World, robots):
        pygame.init()
        pygame.display.set_caption("CPR Milestone 2 - distributed control, no delay")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.font_small = pygame.font.SysFont(None, 16)
        self.font_mono = pygame.font.SysFont("consolas,menlo,monospace", 16)

        self.world = world
        self.robots = robots

        self.paused = True
        self.speed = 2.0
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
                    self.speed = min(30.0, self.speed + (0.5 if self.speed < 4 else 2.0))
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.05, self.speed - (0.5 if self.speed <= 4 else 2.0))
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
        committed = {r.id for r in self.robots if r.state == RobotState.COMMITTED}
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
            pygame.draw.polygon(self.screen, robot_color(rid), pts)
            pygame.draw.polygon(self.screen, COLOR_COMMITTED if rid in committed else COLOR_BG,
                                pts, width=2 if rid in committed else 1)

            label = self.font_small.render(str(rid), True, robot_color(rid))
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

        line(f"turn {self.turn} / {self.sim.max_turns}", COLOR_TEXT_DIM, self.font_mono, 26)
        line(self.phase_label, COLOR_TEXT, self.font_mono, 26)
        line(f"speed: {self.speed:.2f} turns/s   ({'paused' if self.paused else 'playing'})",
             COLOR_TEXT_DIM, self.font_mono, 26)

        mode_str = "RANDOM (baseline)" if self.sim.random_mode else "DISTRIBUTED CONTROL"
        line(f"mode:  {mode_str}", (255, 167, 38) if self.sim.random_mode else COLOR_VALID,
             self.font_mono, 26)
        line(f"valid: {self.valid_count}    illegal: {self.illegal_count}    "
             f"targets left: {len(self.world.targets)}", COLOR_TEXT, self.font, 26)

        line("Current Phase Log:", COLOR_TEXT_DIM, self.font_small, 20)
        details = self.phase_details[:10]
        start_y_details = y
        for index, d in enumerate(details):
            text_color = (170, 180, 200)
            if d.startswith("R"):
                try:
                    rid = int(d.split()[0].replace("R", "").replace(":", ""))
                    text_color = robot_color(rid)
                except Exception:
                    pass
            if index < 5:
                line("  " + d, text_color, self.font_small, 18)
            else:
                surf = self.font_small.render("  " + d, True, text_color)
                self.screen.blit(surf, (x0 + 260, start_y_details + ((index - 5) * 18)))
        if len(self.phase_details) > 10:
            more = self.font_small.render(f"  ... +{len(self.phase_details) - 10} more robots",
                                          True, COLOR_TEXT_DIM)
            self.screen.blit(more, (x0 + 260, start_y_details + 5 * 18))
        if details:
            y = start_y_details + (min(len(details), 5) * 18) + 15

        y = max(y + 10, panel.bottom - 240)
        line("Referee Log:", COLOR_TEXT_DIM, self.font_small, 18)
        for text, color in self.log_lines[-10:]:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += 18

        hint = self.font_small.render(
            "SPACE play/pause | RIGHT step | UP/DOWN speed | M mode | R restart",
            True, COLOR_TEXT_DIM)
        self.screen.blit(hint, (x0, panel.bottom - 22))


# =====================================================================
#  Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="CPR Milestone 2 simulation")
    parser.add_argument("--robots", type=int, default=10, help="number of robots (1..100)")
    parser.add_argument("--targets", type=int, default=None,
                        help="targets generated each time the grid is empty (default: max(5, robots/2))")
    parser.add_argument("--turns", type=int, default=500, help="max iterations")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--headless", action="store_true", help="no window, print summary only")
    parser.add_argument("--search", choices=["coverage", "random"], default="coverage",
                        help="how robots look for targets while free (default: coverage)")
    parser.add_argument("--random", action="store_true",
                        help="baseline: robots act randomly with no coordination")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.headless:
        sim = build_scenario(args.robots, args.turns, args.search, args.targets)
        sim.random_mode = args.random
        sim.run(renderer=None)
        sim.print_summary()
        return

    if pygame is None:
        sys.exit("pygame is not installed: pip install pygame (or use --headless)")

    while True:
        sim = build_scenario(args.robots, args.turns, args.search, args.targets)
        sim.random_mode = args.random
        renderer = PygameRenderer(sim.world, sim.robots)
        renderer.sim = sim

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