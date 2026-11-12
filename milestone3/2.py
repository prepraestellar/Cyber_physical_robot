"""CPR Milestone 3 - distributed control WITH random communication delay.

Continues from Milestone 2. What changed:

  * NETWORK (Simulation.route_messages + Robot.receive/read_messages):
    every broadcast is delivered to every other robot after its OWN random
    delay r ~ Exponential(mean), drawn per receiver (project description 2.3).
    The input queue counts r down once per "read messages" phase.

  * AGREEMENT - Paxos (single-decree, one instance per target):
    a robot that SEES a target proposes "robots W and E capture the target at
    cell c". Every robot is an acceptor and a learner. A value is chosen once a
    majority (quorum) of acceptors accepted it, so two robots can never believe
    different pairs own the same target, whatever the delays are.

  * ORDER - Lamport logical clocks:
    every message carries a Lamport timestamp; ballots and job priorities are
    (lamport, robot id) pairs, which gives one total order that every robot
    agrees on. A robot always works on its OLDEST job first (resource
    hierarchy -> no circular waiting -> no deadlock).

  * SIMULTANEOUS PICKUP - ACK / READY handshake + a physical "go" signal:
    messages alone can never make two robots act at exactly the same turn
    (unbounded delay = coordinated-attack problem), but SENSING is instantaneous.
        1. E learns the decision         -> broadcasts ACK
        2. W has the decision and ACK     -> broadcasts READY
        3. W waits ON the target cell; E waits NEXT to it, facing it
        4. E moves in only if it has READY and sees exactly W ahead
        5. next turn both see each other "here" -> both PICKUP
    E never steps on its own target cell in any other way, so W can trust that
    "E appeared next to me" means "E entered on purpose", and E can trust that
    W will pick up because READY proves W already holds E's ACK.
"""

import argparse
import colorsys
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

try:
    import pygame
except ImportError:          # headless runs do not need pygame
    pygame = None

GRID_SIZE = 10
STALE_AFTER = 20       # a cell looked at less than this many turns ago is "recent"
NEAR_OFFSETS = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 2]
SOFT_COST = 4          # walking over another pair's target cell counts as 4 extra steps
LOAD_COST = 6          # choosing a pair: each job a robot already has counts as 6 extra steps
                       # (a design choice; tests with 0, 3, 6 and 12 gave almost the same results)
NO_BALLOT = (-1, -1)


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


class Role(Enum):
    SEARCH = auto()      # no job: explore the grid looking for targets
    WAITER = auto()      # "W": go to the target cell and wait there
    ENTERER = auto()     # "E": wait next to the target cell, enter only on the handshake


@dataclass
class Message:
    sender_id: int
    payload: dict
    turn_sent: int
    lamport: int         # Lamport timestamp of the send event


@dataclass
class Decision:
    robot_id: int
    action: Action
    position: tuple


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step(pos, direction):
    dx, dy = direction.delta()
    return (pos[0] + dx, pos[1] + dy)


# =====================================================================
#  World  (unchanged from Milestone 2)
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
            new_pos = step(pos, facing)
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
    """The BRAIN of one robot.

    A robot cannot look at the World. It only knows:
        - what its sensors tell it        -> sense(reading)
        - what other robots tell it       -> receive(msg, r)
    and it can only:
        - choose an action                -> decide()
        - send one broadcast per turn     -> get_outgoing_message()
    Things it knows from the start: the arena size and how many robots are in
    the team (needed to know what "more than half" means for Paxos).

    Words used in this class:
        inst  = (cell, k)     "the k-th target that ever appeared on this cell"
                              e.g. ((3, 9), 0) is the first target seen on (3, 9)
        value = (W, E, key)   the pair chosen for that target by Paxos:
                              W = waiter id, E = enterer id,
                              key = (lamport, proposer id) = how old the job is
        job   = an inst whose value has been decided and that is not captured yet
    """

    def __init__(self, robot_id, start_pos, start_facing, arena_size=GRID_SIZE,
                 team_size=1, search_mode="coverage"):
        self.id = robot_id
        self.position = start_pos          # where I think I am (I count my own moves)
        self.facing = start_facing
        self.arena_size = arena_size
        self.team_size = team_size
        self.quorum = team_size // 2 + 1   # "more than half" of the team
        self.search_mode = search_mode
        self.random_policy = False         # baseline mode: random actions, no coordination

        # ---- radio ----
        self.input_queue = []              # my mailbox: [[turns still to wait, message], ...]
        self.messages_this_turn = []
        self.outbox = []                   # events I will broadcast at the end of this turn
        self.turn = -1                     # my turn counter (+1 every turn)
        self.lamport = 0                   # my Lamport counter (used for ballot numbers)
        self.delay_sum = 0                 # used to measure how slow messages are
        self.delay_n = 0

        # ---- what I know about the others (may be OLD because of the delay) ----
        self.peers = {}                    # robot id -> newest status message I got from it
        self._peer_turn = {}               # robot id -> turn that newest message was sent
        self.last_seen = {}                # cell -> last turn ANY robot looked at it
        self.looked = []                   # cells I looked at this turn
        self.goal = None                   # cell I am exploring towards (only when searching)
        self.goal_turn = 0

        # ---- facts that only ever get ADDED (never removed), so the order in which
        #      messages arrive does not matter ----
        self.sightings = {}                # cell -> last turn I saw a target there
        self.captures = {}                 # cell -> {k: turn it was picked up}
        self.decided = {}                  # inst -> value that Paxos chose
        self.pending = set()               # decided but not picked up yet
        self.ack_from = {}                 # (inst, robot) -> turn I learned that robot sent ACK
        self.ready_from = {}               # (inst, robot) -> turn I learned that robot sent READY
        self._ready_sent = set()           # insts I already sent READY for

        # ---- my Paxos notebook ----
        self.promised = {}                 # inst -> highest ballot I promised to listen to
        self.accepted = {}                 # inst -> (ballot, value) I accepted
        self.votes = {}                    # (inst, ballot) -> robots that said ACCEPTED
        self.proposals = {}                # inst -> my own proposal (if I am proposing)
        self.rival = {}                    # inst -> (highest ballot of ANOTHER proposer, turn seen)

        # ---- what I am doing ----
        self.role = Role.SEARCH            # SEARCH, WAITER or ENTERER
        self.job = None                    # the job I work on now (my OLDEST job)
        self.my_jobs = {}                  # all jobs I am part of
        self.soft = set()                  # target cells of OTHER pairs (I try not to walk there)
        self.prev_pos = None               # my position LAST turn
        self.prev_here = frozenset()       # robots on my cell LAST turn
        self.entered = None                # (inst, turn) when I (as E) stepped onto the target
        self.captured = None               # inst I picked up this turn (for display)
        self._protocol_move = False        # True when this turn's move is the handshake entry
        self.reason = ""                   # short "why" shown in the GUI decide log
        self._decided_action = Action.TURN_RIGHT
        self._sensed = {}
        self.stats = {"proposals": 0, "rounds": 0, "learned": 0, "conflicts": 0}

    # ================================================================
    #  Phase 1: read messages
    # ================================================================
    def receive(self, msg, r):
        """The network puts a message in my mailbox. I may read it after r more turns."""
        self.input_queue.append([r, msg])

    def read_messages(self):
        """Exactly the procedure of the project description (2.3):
               for every message in the mailbox:
                   if r == 0: read it      else: r = r - 1"""
        self.turn += 1
        self.messages_this_turn = []
        still_waiting = []
        for item in self.input_queue:
            if item[0] == 0:
                self._deliver(item[1])
            else:
                item[0] -= 1
                still_waiting.append(item)
        self.input_queue = still_waiting

    def _deliver(self, msg):
        """Read ONE message."""
        self.messages_this_turn.append(msg)

        # Lamport rule for every message I read: jump above the sender's number, then +1
        self.lamport = max(self.lamport, msg.lamport) + 1

        self.delay_sum += self.turn - msg.turn_sent - 1
        self.delay_n += 1
        p = msg.payload

        # Position / status: messages can arrive in the wrong order, so I keep only the
        # one that was SENT most recently (an older one arriving late is ignored)
        if msg.turn_sent > self._peer_turn.get(msg.sender_id, -1):
            self.peers[msg.sender_id] = p
            self._peer_turn[msg.sender_id] = msg.turn_sent
        for c in p["looked"]:
            if msg.turn_sent > self.last_seen.get(c, -1):
                self.last_seen[c] = msg.turn_sent

        # Events (Paxos, ACK, READY, CAPTURED): handle every one of them
        for ev in p["events"]:
            self._handle(ev, msg.sender_id)

    def _avg_delay(self):
        """How slow messages have been so far (used to decide how long to wait)."""
        return self.delay_sum / self.delay_n if self.delay_n else 1.0

    # ================================================================
    #  Events: everything I send besides my position
    # ================================================================
    def _emit(self, ev):
        """Send an event at the end of this turn. I also vote in Paxos myself, so I
        handle my own event right away (a robot does not need the radio to talk to itself)."""
        self.outbox.append(ev)
        self._handle(ev, self.id)

    def _handle(self, ev, sender):
        """React to one event received from `sender`."""
        kind = ev[0]

        # ---------- I am a VOTER (Paxos acceptor) ----------
        if kind == "PREPARE":
            # someone asks: "may I lead the vote with this ballot? did you accept a pair?"
            _, inst, b = ev
            if sender != self.id:
                self._note_rival(inst, b)
            if b >= self.promised.get(inst, NO_BALLOT):
                # ballot is not older than the one I promised: promise it and tell the
                # proposer which pair I already accepted (if any)
                self.promised[inst] = b
                ab, av = self.accepted.get(inst, (None, None))
                self._emit(("PROMISE", inst, b, sender, ab, av))
            else:
                # old ballot: refuse and say which (higher) ballot I promised
                self._emit(("NACK", inst, b, sender, self.promised[inst]))

        elif kind == "ACCEPT":
            # the proposer asks: "please accept this pair"
            _, inst, b, v = ev
            if b >= self.promised.get(inst, NO_BALLOT):
                self.promised[inst] = b
                self.accepted[inst] = (b, v)
                self._emit(("ACCEPTED", inst, b, v))           # tell EVERYONE I accepted
            else:
                self._emit(("NACK", inst, b, sender, self.promised[inst]))

        # ---------- I am a PROPOSER (only answers addressed to me) ----------
        elif kind == "PROMISE":
            _, inst, b, to, ab, av = ev
            if to == self.id:
                self._on_promise(inst, b, sender, ab, av)
        elif kind == "NACK":
            _, inst, b, to, higher = ev
            if to == self.id:
                self._on_nack(inst, b, higher)

        # ---------- I am a LEARNER: count the "I accepted" votes ----------
        elif kind == "ACCEPTED":
            _, inst, b, v = ev
            if inst in self.decided:
                return
            voters = self.votes.setdefault((inst, b), set())
            voters.add(sender)
            if len(voters) >= self.quorum:                     # more than half: decided
                self._learn(inst, v)

        # ---------- pickup handshake ----------
        elif kind == "ACK":                                    # E says "I know the job"
            self.ack_from.setdefault((ev[1], sender), self.turn)
            self._maybe_ready(ev[1])
        elif kind == "READY":                                  # W says "I know that E knows"
            self.ready_from.setdefault((ev[1], sender), self.turn)
        elif kind == "CAPTURED":                               # "this target was picked up"
            _, inst, t = ev
            self.captures.setdefault(inst[0], {})[inst[1]] = t
            self.pending.discard(inst)

    # ================================================================
    #  Paxos: proposer side
    # ================================================================
    def _patience(self):
        """How many turns to wait before trying again (longer when messages are slower)."""
        return int(6 * self._avg_delay() + 10)

    def _note_rival(self, inst, b):
        """Another robot is proposing for the same target. If its ballot is higher than
        mine, I step back and let it finish (otherwise we could outbid each other forever)."""
        old = self.rival.get(inst)
        if old is None or b > old[0]:
            self.rival[inst] = (b, self.turn)
        P = self.proposals.get(inst)
        if P is not None and P.get("ballot", NO_BALLOT) < b and P["phase"] != 0:
            P["phase"] = 0                                     # 0 = paused
            P["deadline"] = self.turn + self._patience() + random.randint(0, 5)

    def _rival_active(self, inst):
        """True if another robot proposed for this target recently."""
        r = self.rival.get(inst)
        return r is not None and self.turn - r[1] < self._patience()

    def _start_round(self, inst):
        """Send PREPARE with a NEW ballot = (my Lamport counter, my id).
        My Lamport counter is already bigger than every number I have seen,
        so this ballot is bigger than every ballot I have seen."""
        P = self.proposals[inst]
        P["attempts"] = P.get("attempts", 0) + 1
        self.lamport += 1
        P["ballot"] = (self.lamport, self.id)
        P["phase"] = 1                                         # 1 = waiting for promises
        P["promises"] = {}
        avg = self._avg_delay()
        backoff = min(2 ** (P["attempts"] - 1), 8)             # wait longer after each failure
        P["deadline"] = (self.turn + int((4 * avg + 6) * backoff)
                         + random.randint(0, int(2 * avg) + 2))
        self.stats["rounds"] += 1
        self._emit(("PREPARE", inst, P["ballot"]))

    def _on_promise(self, inst, b, sender, ab, av):
        """A voter promised my ballot. When more than half did, I send ACCEPT."""
        P = self.proposals.get(inst)
        if P is None or P["ballot"] != b or P["phase"] != 1:
            return
        P["promises"][sender] = (ab, av)
        if len(P["promises"]) >= self.quorum:
            # THE KEY RULE OF PAXOS: if any voter already accepted a pair, I must
            # propose that pair (the one with the highest ballot), not my own
            prior = [x for x in P["promises"].values() if x[0] is not None]
            value = max(prior, key=lambda x: x[0])[1] if prior else P["value"]
            P["phase"] = 2                                     # 2 = waiting for ACCEPTED
            self._emit(("ACCEPT", inst, b, value))

    def _on_nack(self, inst, b, higher):
        """A voter refused my ballot because it promised `higher`. I raise my counter
        above it and try again a bit later."""
        P = self.proposals.get(inst)
        self.lamport = max(self.lamport, higher[0])
        if P is not None and P["ballot"] == b and P["phase"] != 0:
            P["phase"] = 0
            p = self._patience()
            P["deadline"] = self.turn + random.randint(p // 2, p)

    def _proposer_tick(self, snap):
        """Every turn: retry proposals that took too long, start new ones."""
        for inst in list(self.proposals):
            P = self.proposals[inst]
            if inst in self.decided:
                del self.proposals[inst]                       # done, nothing more to do
            elif self.turn >= P["deadline"]:
                if self._rival_active(inst) and self.rival[inst][0] > P.get("ballot", NO_BALLOT):
                    P["deadline"] = self.turn + self._patience()   # someone else is ahead: keep waiting
                else:
                    self._start_round(inst)                        # time is up: try again
        for c, seen in list(self.sightings.items()):
            self._maybe_propose(c, seen, snap)

    def _maybe_propose(self, c, seen_turn, snap):
        """I saw a target on cell c. Start a vote for it, unless that is not needed."""
        if self.team_size < 2:
            return                                             # alone: nobody to pair with
        caps = self.captures.get(c)
        if caps:
            k = max(caps) + 1                                  # the next target on this cell
            if seen_turn <= caps[k - 1]:
                return              # I saw it BEFORE the last pickup there: that one is gone
        else:
            k = 0
        inst = (c, k)
        if inst in self.decided or inst in self.proposals or self._rival_active(inst):
            return                  # already decided, I am already voting, or someone else is
        pair = self._choose_pair(c, snap)
        if pair is None:
            return
        self.lamport += 1
        value = (pair[0], pair[1], (self.lamport, self.id))    # my own suggestion
        self.proposals[inst] = {"value": value}
        self.stats["proposals"] += 1
        self._start_round(inst)

    def _choose_pair(self, c, snap):
        """My suggestion for the pair, using what I know (positions may be old).

            score(robot) = distance to c  +  LOAD_COST x (jobs it already has)
            distance     = |x1 - x2| + |y1 - y2|   (number of steps on the grid)

        The two robots with the LOWEST score are chosen.
        The one that is closer to c becomes W (it gets there first and waits)."""
        load = {}
        for inst in self.pending:
            W, E, _ = self.decided[inst]
            load[W] = load.get(W, 0) + 1
            load[E] = load.get(E, 0) + 1
        cands = sorted((manhattan(st["pos"], c) + LOAD_COST * load.get(rid, 0),   # score
                        rid,                                                     # tie-break
                        manhattan(st["pos"], c))                                 # distance
                       for rid, st in snap.items())
        if len(cands) < 2:
            return None
        a, b = sorted(cands[:2], key=lambda x: (x[2], x[1]))   # closer one first
        return a[1], b[1]                                      # (W, E)

    # ================================================================
    #  Paxos: learner side
    # ================================================================
    def _learn(self, inst, v):
        """More than half accepted pair v: this is now the final answer for inst."""
        if inst in self.decided:
            if self.decided[inst] != v:
                self.stats["conflicts"] += 1                    # would be a Paxos bug: never happens
            return
        self.decided[inst] = v
        self.stats["learned"] += 1
        self.proposals.pop(inst, None)
        if not self._is_captured(inst):
            self.pending.add(inst)
        if v[1] == self.id:
            self._emit(("ACK", inst))                          # I am E: "I know the job"
        self._maybe_ready(inst)

    def _maybe_ready(self, inst):
        """I am W, I know the job and E's ACK has arrived: send READY (once)."""
        v = self.decided.get(inst)
        if (v is not None and v[0] == self.id and (inst, v[1]) in self.ack_from
                and inst not in self._ready_sent):
            self._ready_sent.add(inst)
            self._emit(("READY", inst))

    def _is_captured(self, inst):
        return inst[1] in self.captures.get(inst[0], {})

    # ================================================================
    #  Phase 2: sense
    # ================================================================
    def sense(self, reading):
        """My 3 sensors: {"target_ahead": bool, "robots_ahead": [ids], "robots_here": [ids]}"""
        front = self._front_cell()
        self._sensed = reading
        self.looked = [self.position]
        if not self._wall_ahead():
            self.looked.append(front)
            if reading.get("target_ahead"):
                self.sightings[front] = self.turn              # remember: target seen there now
        for c in self.looked:
            self.last_seen[c] = self.turn

    # ---------------- small helpers (they only use my own pose) ----------------
    def _front_cell(self):
        return step(self.position, self.facing)

    def _wall_ahead(self):
        x, y = self._front_cell()
        return not (0 <= x < self.arena_size and 0 <= y < self.arena_size)

    def _in_bounds(self, c):
        return 0 <= c[0] < self.arena_size and 0 <= c[1] < self.arena_size

    @property
    def target(self):
        """Cell of the job I work on (used by the GUI)."""
        return self.job[0] if self.job else None

    def _status(self):
        """What I tell the others about myself every turn."""
        return {
            "pos": self.position,
            "facing": self.facing.name,
            "role": self.role.name,
            "job": self.target,
            "looked": list(self.looked),
            "goal": self.goal if self.role == Role.SEARCH else None,
        }

    def _snapshot(self):
        """Everybody's status as far as I know (myself = exact, others = maybe old)."""
        snap = dict(self.peers)
        snap[self.id] = self._status()
        return snap

    def _enterer_cells(self):
        """Target cells where I am E. I may NEVER step on them, except with the
        handshake entry move."""
        return {inst[0] for inst in self.pending if self.decided[inst][1] == self.id}

    # ================================================================
    #  Phase 2 (continued): decide
    # ================================================================
    def decide(self):
        """Choose this turn's action. Then two safety checks."""
        self._protocol_move = False
        if self.random_policy:
            # baseline: no coordination at all (to compare against the controller)
            self.role, self.job, self.entered = Role.SEARCH, None, None
            self.reason = "random"
            action = random.choice(list(Action))
        else:
            action = self._controller()

        # check 1: never drive into the wall
        if action == Action.MOVE_FORWARD and self._wall_ahead():
            action = random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
            self.reason = "wall ahead"

        # check 2 (SAFETY MONITOR): as E, never step onto my target cell except with the
        # handshake entry. This is what lets W trust that "E appeared" = "E came to pick up".
        if (action == Action.MOVE_FORWARD and not self._protocol_move
                and not self.random_policy and self._front_cell() in self._enterer_cells()):
            action = Action.TURN_LEFT if self.turn % 2 == 0 else Action.TURN_RIGHT
            self.reason = "safety: not via handshake"

        if action != Action.MOVE_FORWARD and self.entered and self.entered[1] == self.turn:
            self.entered = None                # the entry move did not happen after all

        # remember where I am and who is with me: next turn this is "last turn"
        if self.random_policy:
            self.prev_pos = None
        else:
            self.prev_pos = self.position
            self.prev_here = frozenset(self._sensed.get("robots_here", []))
        self._decided_action = action
        return action

    def _controller(self):
        """The robot's plan, in order of importance."""
        snap = self._snapshot()
        self.captured = None
        self._refresh_jobs()                   # which jobs do I have? which one first?
        self._proposer_tick(snap)              # Paxos work for targets I saw

        # 1. Must I pick up NOW? (my partner is picking up this same turn)
        inst = self._pickup_obligation()
        if inst is not None:
            c, k = inst
            self.captures.setdefault(c, {})[k] = self.turn
            self.pending.discard(inst)
            self.captured = inst
            self._emit(("CAPTURED", inst, self.turn))
            self.reason = "pick up together"
            return Action.PICKUP

        # 2. I have a job: do my part as W or as E
        if self.job is not None:
            self.goal = None
            W, E, _ = self.my_jobs[self.job]
            c = self.job[0]
            if self.role == Role.WAITER:
                if self.position != c:
                    self.reason = "going to target"
                    return self._navigate(c)
                self.reason = "waits on cell"
                return self._wait()
            return self._enterer_action(self.job, c, W)

        # 3. No job: look for targets
        self.reason = "searching"
        return self._search(snap)

    def _refresh_jobs(self):
        """List my jobs and pick the OLDEST one (smallest key). Every robot sorts jobs
        the same way, so the oldest job in the team always has both robots coming
        (this is the 'resource hierarchy' idea: it prevents waiting in a circle)."""
        self.my_jobs, self.soft = {}, set()
        for inst in self.pending:
            v = self.decided[inst]
            if self.id in (v[0], v[1]):
                self.my_jobs[inst] = v
            else:
                self.soft.add(inst[0])
        if self.my_jobs:
            self.job = min(self.my_jobs, key=lambda i: (self.my_jobs[i][2], i))
            self.role = Role.WAITER if self.my_jobs[self.job][0] == self.id else Role.ENTERER
        else:
            self.job, self.role = None, Role.SEARCH

    def _pickup_obligation(self):
        """Do I have to pick up this turn? The two rules below are always true on the
        SAME turn for W and for E, so they pick up together."""
        here = frozenset(self._sensed.get("robots_here", []))

        # E rule: last turn I made the handshake entry, and now only W is here with me
        if self.entered is not None:
            inst, t_in = self.entered
            self.entered = None
            if (t_in == self.turn - 1 and inst in self.my_jobs
                    and self.position == inst[0] and here == {self.my_jobs[inst][0]}):
                return inst

        # W rule: last turn I was alone on the target and already had E's ACK,
        # and now E is here with me (E can only have come with the handshake entry)
        for inst, (W, E, _) in self.my_jobs.items():
            if W == self.id and inst[0] == self.position:
                t_ack = self.ack_from.get((inst, E))
                if (self.prev_pos == self.position and not self.prev_here
                        and t_ack is not None and t_ack <= self.turn - 1 and here == {E}):
                    return inst
        return None

    def _enterer_action(self, inst, c, W):
        """What E does: stand next to the target, face it, step in on the handshake."""
        spot = self._approach_cell(c)
        if spot is None:
            self.reason = "no free side cell"
            return self._wait()
        if self.position != spot:
            self.reason = "going beside target"
            return self._navigate(spot)
        want = self._heading_to(c)
        if self.facing != want:
            self.reason = "turns to face target"
            return self._turn_towards(want)
        ahead = frozenset(self._sensed.get("robots_ahead", []))
        if (inst, W) in self.ready_from and ahead == {W}:
            # I have READY and I see W alone on the target: step in (the "go" signal)
            self._protocol_move = True
            self.entered = (inst, self.turn)
            self.reason = "ENTERS (READY + W alone)"
            return Action.MOVE_FORWARD
        if (inst, W) not in self.ready_from:
            self.reason = "waits for READY"
        elif W not in ahead:
            self.reason = f"waits for R{W} to arrive"
        else:
            self.reason = f"waits, R{W} not alone"
        return Action.TURN_LEFT              # look away for one turn, turn back the next turn

    def _approach_cell(self, c):
        """The cell next to c where E waits. Never on a target cell if possible:
        standing on another pair's target would stop that W from ever being alone."""
        forbidden = self._enterer_cells()
        job_cells = self.soft | {i[0] for i in self.my_jobs}
        options = [n for n in (step(c, d) for d in Direction)
                   if self._in_bounds(n) and n not in forbidden]
        if not options:
            return None
        return min(options, key=lambda n: (n in job_cells, manhattan(self.position, n), n))

    def _wait(self):
        """There is no 'do nothing' action, so waiting = turn left, right, left, ..."""
        return Action.TURN_LEFT if self.turn % 2 == 0 else Action.TURN_RIGHT

    # ================================================================
    #  Moving around
    # ================================================================
    def _heading_to(self, c):
        """Direction of the first step towards c (x first, then y)."""
        dx, dy = c[0] - self.position[0], c[1] - self.position[1]
        if dx != 0:
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _turn_towards(self, want):
        """Move forward if I already face `want`, otherwise turn towards it."""
        if self.facing == want:
            return Action.MOVE_FORWARD
        diff = (want.value - self.facing.value) % 4
        return Action.TURN_LEFT if diff == 3 else Action.TURN_RIGHT

    def _distances_to(self, goal, forbidden):
        """For every cell: how many steps to reach `goal` (Dijkstra, started at the goal).
        Forbidden cells cannot be crossed. Other pairs' target cells cost extra."""
        dist = {goal: 0}
        pq = [(0, goal)]
        while pq:
            d, c = heapq.heappop(pq)
            if d > dist[c]:
                continue
            if c in forbidden and c != goal:
                continue
            enter = 1 + (SOFT_COST if (c in self.soft and c != goal) else 0)
            for dr in Direction:
                x = step(c, dr)
                if self._in_bounds(x) and d + enter < dist.get(x, math.inf):
                    dist[x] = d + enter
                    heapq.heappush(pq, (d + enter, x))
        return dist

    def _navigate(self, goal):
        """One action towards `goal`: look at my 4 neighbour cells, pick the one with
        the smallest remaining distance (if equal: the one that needs fewer turns),
        then turn towards it or move into it."""
        if self.position == goal:
            return self._wait()
        forbidden = self._enterer_cells()
        dist = self._distances_to(goal, forbidden)
        best = None
        for d in Direction:
            n = step(self.position, d)
            if not self._in_bounds(n) or n in forbidden or n not in dist:
                continue
            enter = 1 + (SOFT_COST if (n in self.soft and n != goal) else 0)
            turns = {0: 0, 1: 1, 2: 2, 3: 1}[(d.value - self.facing.value) % 4]
            key = (enter + dist[n], turns)
            if best is None or key < best[0]:
                best = (key, d)
        if best is None:
            return self._wait()
        return self._turn_towards(best[1])

    # ================================================================
    #  Searching for targets (same idea as Milestone 2)
    # ================================================================
    def _search(self, snap):
        if self.search_mode == "random":
            return self._random_walk()
        return self._explore(snap)

    def _random_walk(self):
        """Walk mostly straight, turn sometimes."""
        if self._wall_ahead():
            return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        if random.random() < 0.15:
            return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])
        return Action.MOVE_FORWARD

    def _explore(self, snap):
        """Pick the cheapest cell to go and look at:
              cost = distance + turns needed + 'looked at recently' + 'other robots near'
        (same as Milestone 2). Information about the others may be old now; that
        only makes searching a bit slower, it can never cause an illegal pickup."""
        taken, occupied = {}, {}
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
                    if c == self.position or c in taken or c in self.soft:
                        continue
                    age = self.turn - self.last_seen.get(c, -10 ** 6)
                    recency = max(0, STALE_AFTER - age) * 0.5
                    d = (self._heading_to(c).value - self.facing.value) % 4
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

    # ================================================================
    #  Phase 5: after the world performed my action
    # ================================================================
    def update_odometry(self, action):
        """Update where I think I am from the action I commanded."""
        if action == Action.MOVE_FORWARD and not self._wall_ahead():
            self.position = self._front_cell()
        elif action == Action.TURN_LEFT:
            self.facing = self.facing.turn_left()
        elif action == Action.TURN_RIGHT:
            self.facing = self.facing.turn_right()

    # ================================================================
    #  Phase 6: send
    # ================================================================
    def get_outgoing_message(self):
        """My broadcast: my status + every event of this turn.
        Lamport rule for sending: +1, and the number travels with the message."""
        payload = self._status()
        payload["events"] = self.outbox
        self.outbox = []
        self.lamport += 1
        return payload, self.lamport


# =====================================================================
#  Simulation
# =====================================================================

class Simulation:
    """Plays the roles a robot must NOT be able to play itself:
         * the sensors   (sensor_reading)
         * the physics   (World.apply_action)
         * the network   (route_messages)  - now with exponential per-receiver delay
       plus the referee (referee_check)."""

    def __init__(self, robots_config, world: World, max_turns=500, search_mode="coverage",
                 delay_mean=2.0):
        self.world = world
        self.max_turns = max_turns
        self.delay_mean = delay_mean

        self.decisions = {}
        self.valid_pickup_cells = set()
        self.illegal_log = []
        self.valid_log = []
        self.messages_sent = 0
        self.delay_total = 0
        self.log_events = False            # --log: print every coordination message sent

        n = len(robots_config)
        self.robots = []
        for cfg in robots_config:
            r = Robot(arena_size=world.size, team_size=n, search_mode=search_mode, **cfg)
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

    # ---- sensors ----
    def sensor_reading(self, robot_id):
        w = self.world
        pos, facing = w.robot_positions[robot_id], w.robot_facings[robot_id]
        front = step(pos, facing)
        return {
            "target_ahead": w.has_target(front) if w.in_bounds(front) else False,
            "robots_ahead": [r for r in w.robots_at(front) if r != robot_id],
            "robots_here": [r for r in w.robots_at(pos) if r != robot_id],
        }

    # ---- network: broadcast with random delay PER RECEIVER (spec 2.3) ----
    def sample_delay(self):
        """r ~ Exponential, rounded down to whole turns. The rate is chosen as
        ln(1 + 1/mean) so that the rounded value still has the requested mean."""
        if self.delay_mean <= 0:
            return 0
        return int(random.expovariate(math.log(1.0 + 1.0 / self.delay_mean)))

    def route_messages(self):
        for sender in self.robots:
            payload, lamport = sender.get_outgoing_message()
            msg = Message(sender.id, payload, sender.turn, lamport)
            if self.log_events:
                self._log_events(sender, payload["events"])
            for receiver in self.robots:
                if receiver is not sender:
                    r = self.sample_delay()
                    receiver.receive(msg, r)
                    self.messages_sent += 1
                    self.delay_total += r

    def in_flight(self):
        return sum(len(r.input_queue) for r in self.robots)

    # ---- optional event log (--log): the simulation "listens" to the radio ----
    def _log_events(self, sender, events):
        """Prints, as plain sentences, every coordination message a robot SENDS.
        (The other robots receive it later: each one after its own random delay.)"""
        def tgt(inst):
            return f"target {inst[0]} #{inst[1]}"

        def bal(b):
            return f"({b[0]},R{b[1]})"

        def pair(v):
            return f"W=R{v[0]}, E=R{v[1]}"

        me = f"R{sender.id}"
        for ev in events:
            kind = ev[0]
            if kind == "PREPARE":
                line = (f"{me} PREPARE  : asks everyone to vote on {tgt(ev[1])} with ballot "
                        f"{bal(ev[2])}, and asks if anyone already accepted a pair")
            elif kind == "PROMISE":
                said = (f"already accepted pair {pair(ev[5])}" if ev[5]
                        else "has not accepted any pair yet")
                who = "itself (it votes too)" if ev[3] == sender.id else f"R{ev[3]}"
                line = (f"{me} PROMISE  : tells {who} it will follow ballot {bal(ev[2])} "
                        f"for {tgt(ev[1])} ({said})")
            elif kind == "NACK":
                line = (f"{me} NACK     : tells R{ev[3]} it refuses ballot {bal(ev[2])} for "
                        f"{tgt(ev[1])}, because it already promised a newer ballot {bal(ev[4])}")
            elif kind == "ACCEPT":
                line = (f"{me} ACCEPT   : asks everyone to accept pair {pair(ev[3])} for "
                        f"{tgt(ev[1])} (ballot {bal(ev[2])})")
            elif kind == "ACCEPTED":
                line = (f"{me} ACCEPTED : tells everyone it accepted pair {pair(ev[3])} for "
                        f"{tgt(ev[1])}")
            elif kind == "ACK":
                line = f"{me} ACK      : (E) tells W it knows the job for {tgt(ev[1])}"
            elif kind == "READY":
                line = f"{me} READY    : (W) tells E it knows that E knows the job for {tgt(ev[1])}"
            elif kind == "CAPTURED":
                line = f"{me} CAPTURED : tells everyone {tgt(ev[1])} has been picked up"
            else:
                continue
            print(f"turn {sender.turn:4d}  {line}")

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
                    "reason": "pickup attempted on empty cell (no target present)"})
            elif count == 1:
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": "only one robot attempted pickup (need exactly two)"})
            elif count == 2:
                self.valid_pickup_cells.add(cell)
                self.valid_log.append({"turn": turn, "cell": cell, "robots": robot_ids})
            else:
                self.illegal_log.append({
                    "turn": turn, "cell": cell, "robots": robot_ids,
                    "reason": f"{count} robots attempted pickup (more than two)"})

    def run(self, renderer=None):
        for turn in range(self.max_turns):
            self.world.maybe_generate_random_targets()
            self.decisions = {}

            # for all robots: read input messages
            for r in self.robots:
                r.read_messages()
            if renderer:
                details = [f"R{r.id}: read {len(r.messages_this_turn)}, still waiting {len(r.input_queue)}"
                           for r in self.robots]
                renderer.checkpoint("Phase 1: read messages", turn, details=details)

            # for all robots: sense environment
            for r in self.robots:
                r.sense(self.sensor_reading(r.id))
            if renderer:
                details = []
                for r in self.robots:
                    sees = []
                    if r._sensed.get("target_ahead"):
                        sees.append("target ahead")
                    if r._sensed.get("robots_ahead"):
                        sees.append("robots ahead " + ",".join(f"R{i}" for i in r._sensed['robots_ahead']))
                    if r._sensed.get("robots_here"):
                        sees.append("robots here " + ",".join(f"R{i}" for i in r._sensed['robots_here']))
                    details.append(f"R{r.id}: " + (", ".join(sees) if sees else "nothing"))
                renderer.checkpoint("Phase 2: sense environment", turn, details=details)

            # for all robots: decide on action to perform
            for r in self.robots:
                action = r.decide()
                self.decisions[r.id] = Decision(r.id, action, self.world.robot_positions[r.id])
            if renderer:
                details = []
                short = {Action.MOVE_FORWARD: "FORWARD", Action.TURN_LEFT: "LEFT",
                         Action.TURN_RIGHT: "RIGHT", Action.PICKUP: "PICKUP"}
                for r in self.robots:
                    tag = "" if r.role == Role.SEARCH else (
                        f" {r.role.name[0]}({r.target[0]},{r.target[1]})")
                    details.append(f"R{r.id}{tag} {r.reason}: {short[r._decided_action]}")
                renderer.checkpoint("Phase 3: decide action", turn, details=details)

            # Test for illegal behavior here
            self.referee_check(turn)
            if renderer:
                renderer.checkpoint("Phase 4: referee check", turn, ingest_results=True,
                                    details=["The referee checks every PICKUP"])

            # for all robots: perform action
            for r in self.robots:
                action = self.decisions[r.id].action
                self.world.apply_action(r.id, action, self.valid_pickup_cells)
                r.update_odometry(action)
            if renderer:
                renderer.checkpoint("Phase 5: perform actions", turn, details=["Every robot does its action"])

            # for all robots: send output messages
            self.route_messages()
            if renderer:
                details = [f"R{r.id} sends from {r.position}" for r in self.robots]
                renderer.checkpoint("Phase 6: send messages", turn, details=details)

    # ---- results ----
    def paxos_agreement_violations(self):
        """Paxos safety check: no two robots may have learned different values."""
        values = {}
        for r in self.robots:
            for inst, v in r.decided.items():
                values.setdefault(inst, set()).add(v)
        return sum(1 for vs in values.values() if len(vs) > 1)

    def results(self):
        jobs = set()
        for r in self.robots:
            jobs.update(r.decided)
        return {
            "robots": len(self.robots),
            "turns": self.max_turns,
            "delay_mean": self.delay_mean,
            "generated": self.world.generated_count,
            "valid": len(self.valid_log),
            "illegal": len(self.illegal_log),
            "left": len(self.world.targets),
            "jobs_decided": len(jobs),
            "paxos_violations": self.paxos_agreement_violations(),
            "paxos_rounds": sum(r.stats["rounds"] for r in self.robots),
            "messages": self.messages_sent,
            "avg_delay": self.delay_total / self.messages_sent if self.messages_sent else 0.0,
        }

    def print_summary(self):
        res = self.results()
        print("\n=== Milestone 3 summary ===")
        print(f"Robots:                     {res['robots']}")
        print(f"Turns simulated:            {res['turns']}")
        print(f"Mean delay (Exp):           {res['delay_mean']}  "
              f"(measured {res['avg_delay']:.2f} turns over {res['messages']} deliveries)")
        print(f"Targets generated:          {res['generated']}")
        print(f"Valid pickups (captured):   {res['valid']}")
        print(f"Targets still on the grid:  {res['left']}")
        print(f"Jobs decided by Paxos:      {res['jobs_decided']}  ({res['paxos_rounds']} ballots)")
        print(f"Paxos agreement violations: {res['paxos_violations']}")
        print(f"Illegal pickups detected:   {res['illegal']}")
        for i in self.illegal_log:
            print(f"  turn {i['turn']}: cell {i['cell']} -> {i['reason']} (robots {i['robots']})")
        if self.valid_log:
            print(f"Average turns per capture:  {self.max_turns / len(self.valid_log):.1f}")


# =====================================================================
#  Scenario
# =====================================================================

def build_scenario(num_robots=10, max_turns=500, search_mode="coverage", targets_per_batch=None,
                   delay_mean=2.0):
    if not 1 <= num_robots <= GRID_SIZE * GRID_SIZE:
        raise ValueError("number of robots must be between 1 and 100")
    world = World(GRID_SIZE)
    cells = random.sample([(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)], num_robots)
    robots_config = [
        dict(robot_id=i + 1, start_pos=cells[i], start_facing=random.choice(list(Direction)))
        for i in range(num_robots)
    ]
    sim = Simulation(robots_config, world, max_turns=max_turns, search_mode=search_mode,
                     delay_mean=delay_mean)
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
COLOR_WAITER = (255, 214, 102)
COLOR_ENTERER = (255, 143, 60)
COLOR_JOB = (255, 214, 102)

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
        pygame.display.set_caption("CPR Milestone 3 - distributed control with communication delay")
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
                elif event.key == pygame.K_d:
                    self.sim.delay_mean = max(0.0, self.sim.delay_mean - 1.0)
                elif event.key == pygame.K_f:
                    self.sim.delay_mean = min(30.0, self.sim.delay_mean + 1.0)

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
            target_dt = 1.0 / (self.speed * 6)
            start = time.time()
            while time.time() - start < target_dt:
                self._pump()
                self._draw()
                self.clock.tick(60)

    def _ingest_referee_results(self, turn):
        now = time.time()
        for v in (v for v in self.sim.valid_log if v["turn"] == turn):
            self.flashes.append((v["cell"], COLOR_VALID, now))
            self.log_lines.append((f"turn {turn}  VALID pickup at {v['cell']} "
                                   f"robots {v['robots']}", COLOR_VALID))
            self.valid_count += 1
        for i in (i for i in self.sim.illegal_log if i["turn"] == turn):
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
        self._draw_jobs()
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

    def _draw_jobs(self):
        # a dashed-looking ring around every target that has a Paxos-decided pair
        cells = {r.target for r in self.robots if r.target is not None}
        for (x, y) in cells:
            pygame.draw.rect(self.screen, COLOR_JOB,
                             (MARGIN + x * CELL + 3, MARGIN + y * CELL + 3, CELL - 6, CELL - 6), 1)

    def _draw_targets(self):
        targets, _, _ = self.world.snapshot()
        for (x, y) in targets:
            cx, cy = self._cell_center(x, y)
            pygame.draw.circle(self.screen, COLOR_TARGET, (cx, cy), 12)

    def _draw_robots(self):
        _, positions, facings = self.world.snapshot()
        roles = {r.id: r.role for r in self.robots}
        local_pts = [(0, -13), (10, 10), (0, 5), (-10, 10)]
        for rid, (x, y) in positions.items():
            cx, cy = self._cell_center(x, y)
            rad = math.radians(facings[rid].value * 90)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            pts = [(cx + px * cos_a - py * sin_a, cy + px * sin_a + py * cos_a)
                   for px, py in local_pts]
            pygame.draw.polygon(self.screen, robot_color(rid), pts)
            role = roles[rid]
            if role == Role.WAITER:
                pygame.draw.polygon(self.screen, COLOR_WAITER, pts, width=2)
            elif role == Role.ENTERER:
                pygame.draw.polygon(self.screen, COLOR_ENTERER, pts, width=2)
            else:
                pygame.draw.polygon(self.screen, COLOR_BG, pts, width=1)
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

        line(f"turn {self.turn} / {self.sim.max_turns}", COLOR_TEXT_DIM, self.font_mono, 24)
        line(self.phase_label, COLOR_TEXT, self.font_mono, 24)
        line(f"speed: {self.speed:.2f} turns/s   ({'paused' if self.paused else 'playing'})",
             COLOR_TEXT_DIM, self.font_mono, 24)
        mode_str = "RANDOM (baseline)" if self.sim.random_mode else "PAXOS + HANDSHAKE"
        line(f"mode:  {mode_str}", (255, 167, 38) if self.sim.random_mode else COLOR_VALID,
             self.font_mono, 24)
        line(f"delay: Exp(mean={self.sim.delay_mean:.0f})   in flight: {self.sim.in_flight()}",
             COLOR_TEXT, self.font_mono, 24)
        line(f"valid: {self.valid_count}    illegal: {self.illegal_count}    "
             f"targets left: {len(self.world.targets)}", COLOR_TEXT, self.font, 24)

        line("This phase (W = waits on target, E = steps in):", COLOR_TEXT_DIM, self.font_small, 20)
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
        line("Referee log:", COLOR_TEXT_DIM, self.font_small, 18)
        for text, color in self.log_lines[-10:]:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += 18

        hint = self.font_small.render(
            "SPACE play/pause | RIGHT step | UP/DOWN speed | D/F delay -/+ | M mode | R restart",
            True, COLOR_TEXT_DIM)
        self.screen.blit(hint, (x0, panel.bottom - 22))


# =====================================================================
#  Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="CPR Milestone 3 simulation")
    parser.add_argument("--robots", type=int, default=10, help="number of robots (1..100)")
    parser.add_argument("--targets", type=int, default=None,
                        help="targets generated each time the grid is empty (default: max(5, robots/2))")
    parser.add_argument("--turns", type=int, default=500, help="max iterations")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="mean of the exponential message delay, in turns (0 = no delay)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--headless", action="store_true", help="no window, print summary only")
    parser.add_argument("--search", choices=["coverage", "random"], default="coverage",
                        help="how robots look for targets while free (default: coverage)")
    parser.add_argument("--random", action="store_true",
                        help="baseline: robots act randomly with no coordination")
    parser.add_argument("--log", action="store_true",
                        help="print every Paxos / handshake message a robot sends")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.headless:
        sim = build_scenario(args.robots, args.turns, args.search, args.targets, args.delay)
        sim.random_mode = args.random
        sim.log_events = args.log
        sim.run(renderer=None)
        sim.print_summary()
        return

    if pygame is None:
        sys.exit("pygame is not installed: pip install pygame (or use --headless)")

    while True:
        sim = build_scenario(args.robots, args.turns, args.search, args.targets, args.delay)
        sim.random_mode = args.random
        sim.log_events = args.log
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