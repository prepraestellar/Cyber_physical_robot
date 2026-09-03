"""
Cyber Physical Robotics - Milestone 1
--------------------------------------
Goal of Milestone 1 (per the project spec):
  - NO communication delay
  - NO full autonomous robot decision-making needed yet
  - Just PROVE the simulator correctly detects the three illegal conditions:
        1) A robot tries to pick up a target where there is NO target
        2) Exactly ONE robot tries to pick up a target where a target DOES exist
           (valid pickup requires exactly TWO robots at the same time)
        3) MORE THAN TWO robots try to pick up the same target at once

This script uses *scripted* robot actions (not real decision logic) to force
each of these scenarios to happen, so you can show your professor that the
detector works. Real autonomous behavior + broadcasting is Milestone 2.

Run:
    pip install matplotlib
    python milestone1_simulation.py
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon

GRID_SIZE = 10

# Directions as (dx, dy), listed in clockwise order for turning
DIR_VECTORS = {
    'N': (0, 1),
    'E': (1, 0),
    'S': (0, -1),
    'W': (-1, 0),
}
DIR_ORDER = ['N', 'E', 'S', 'W']

# Rotation angle (degrees) used to draw each direction as an arrow/triangle
DIR_ANGLE = {
    'N': 90,
    'E': 0,
    'S': 270,
    'W': 180,
}


class Robot:
    def __init__(self, robot_id, x, y, direction):
        self.id = robot_id
        self.x = x
        self.y = y
        self.direction = direction
        self.pending_action = None  # set each turn before performing

    def turn_left(self):
        idx = DIR_ORDER.index(self.direction)
        self.direction = DIR_ORDER[(idx - 1) % 4]

    def turn_right(self):
        idx = DIR_ORDER.index(self.direction)
        self.direction = DIR_ORDER[(idx + 1) % 4]

    def front_position(self):
        dx, dy = DIR_VECTORS[self.direction]
        return (self.x + dx, self.y + dy)


class Simulation:
    def __init__(self, num_robots=4, max_iterations=200, targets_per_spawn=3):
        self.grid_size = GRID_SIZE
        self.max_iterations = max_iterations
        self.iteration = 0
        self.targets = set()
        self.illegal_log = []
        self.targets_per_spawn = targets_per_spawn
        self.robots = []
        self._init_robots(num_robots)

    def _init_robots(self, num_robots):
        occupied = set()
        for i in range(num_robots):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    break
            direction = random.choice(DIR_ORDER)
            self.robots.append(Robot(i, x, y, direction))

    def spawn_targets(self):
        attempts, added = 0, 0
        while added < self.targets_per_spawn and attempts < 200:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in self.targets:
                self.targets.add((x, y))
                added += 1
            attempts += 1

    def in_bounds(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def step(self, scripted_actions=None):
        """
        scripted_actions: dict {robot_id: action_str}
        action_str in {'forward', 'left', 'right', 'pickup'}
        Any robot NOT listed just moves randomly (forward/left/right) -
        random movement never tries to pick up, so it can't interfere
        with the scripted illegal-condition demo.
        """
        self.iteration += 1

        if not self.targets:
            self.spawn_targets()

        # ---- 1) read input messages: skipped for Milestone 1 (no comms yet) ----

        # ---- 2) sense + decide action for every robot ----
        for robot in self.robots:
            if scripted_actions and robot.id in scripted_actions:
                robot.pending_action = scripted_actions[robot.id]
            else:
                robot.pending_action = random.choice(['forward', 'left', 'right'])

        # ---- 3) TEST FOR ILLEGAL BEHAVIOR (this is the Milestone 1 deliverable) ----
        self._check_illegal_conditions()

        # ---- 4) perform actions ----
        for robot in self.robots:
            self._perform_action(robot)

        # ---- 5) send output messages: skipped for Milestone 1 ----

    def _check_illegal_conditions(self):
        pickup_attempts = {}
        for robot in self.robots:
            if robot.pending_action == 'pickup':
                pos = (robot.x, robot.y)
                pickup_attempts.setdefault(pos, []).append(robot.id)

        for pos, ids in pickup_attempts.items():
            has_target = pos in self.targets
            if not has_target:
                msg = f"[ILLEGAL] iter {self.iteration}: robot(s) {ids} tried to pick up at {pos} but no target is there."
                self._log(msg)
            elif len(ids) == 1:
                msg = f"[ILLEGAL] iter {self.iteration}: exactly one robot {ids} tried to pick up target at {pos} (needs exactly 2)."
                self._log(msg)
            elif len(ids) > 2:
                msg = f"[ILLEGAL] iter {self.iteration}: {len(ids)} robots {ids} tried to pick up the same target at {pos} (max 2)."
                self._log(msg)
            elif len(ids) == 2:
                msg = f"[OK] iter {self.iteration}: robots {ids} validly captured target at {pos}."
                self._log(msg)

    def _log(self, msg):
        self.illegal_log.append(msg)
        print(msg)

    def _perform_action(self, robot):
        action = robot.pending_action
        if action == 'forward':
            fx, fy = robot.front_position()
            if self.in_bounds(fx, fy):
                robot.x, robot.y = fx, fy
        elif action == 'left':
            robot.turn_left()
        elif action == 'right':
            robot.turn_right()
        elif action == 'pickup':
            pos = (robot.x, robot.y)
            same_cell_pickups = [
                r for r in self.robots
                if r.pending_action == 'pickup' and (r.x, r.y) == pos
            ]
            if pos in self.targets and len(same_cell_pickups) == 2:
                self.targets.discard(pos)
        robot.pending_action = None


# ---------------------------------------------------------------------------
# Milestone-1 scripted scenarios: force each illegal condition to appear
# ---------------------------------------------------------------------------
def build_demo_scenarios(sim):
    """
    Returns a list of (description, scripted_actions_dict) to be replayed
    one per simulation step, forcing each illegal case in turn.
    We manually place robots and targets right before each scenario so the
    condition is guaranteed to trigger, regardless of random state.
    """
    scenarios = []

    # Scenario A: single robot tries to pick up where there is NO target
    def setup_a():
        sim.robots[0].x, sim.robots[0].y = 2, 2
        sim.targets.discard((2, 2))
    scenarios.append(("No target present", setup_a, {0: 'pickup'}))

    # Scenario B: exactly ONE robot tries to pick up a target that exists
    def setup_b():
        sim.robots[0].x, sim.robots[0].y = 5, 5
        sim.targets.add((5, 5))
    scenarios.append(("Only one robot present", setup_b, {0: 'pickup'}))

    # Scenario C: more than two robots (3) try to pick up the same target
    def setup_c():
        sim.robots[0].x, sim.robots[0].y = 7, 7
        if len(sim.robots) > 1:
            sim.robots[1].x, sim.robots[1].y = 7, 7
        if len(sim.robots) > 2:
            sim.robots[2].x, sim.robots[2].y = 7, 7
        sim.targets.add((7, 7))
    ids = [r.id for r in sim.robots[:3]]
    scenarios.append(("Three robots at once", setup_c, {i: 'pickup' for i in ids}))

    # Scenario D: the VALID case - exactly two robots pick up together (control case)
    def setup_d():
        sim.robots[0].x, sim.robots[0].y = 1, 8
        if len(sim.robots) > 1:
            sim.robots[1].x, sim.robots[1].y = 1, 8
        sim.targets.add((1, 8))
    valid_ids = [r.id for r in sim.robots[:2]]
    scenarios.append(("Valid 2-robot capture (control case)", setup_d, {i: 'pickup' for i in valid_ids}))

    return scenarios


# ---------------------------------------------------------------------------
# Real-time visualization
# ---------------------------------------------------------------------------
def run_visual_demo(num_robots=4, max_iterations=200):
    sim = Simulation(num_robots=num_robots, max_iterations=max_iterations)
    sim.spawn_targets()

    scenarios = build_demo_scenarios(sim)
    scenario_iter = iter(scenarios)
    current_label = ""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    # Gridlines go at cell BOUNDARIES (half-integers), so each unit square
    # is centered on an integer coordinate -> robots/targets sit inside
    # the cell instead of on the intersection.
    ax.set_xticks([i - 0.5 for i in range(GRID_SIZE + 1)])
    ax.set_yticks([i - 0.5 for i in range(GRID_SIZE + 1)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_aspect('equal')
    title = ax.set_title("Milestone 1: illegal-condition demo")

    def draw_frame(_):
        nonlocal current_label
        for patch in list(ax.patches):
            patch.remove()
        for txt in list(ax.texts):
            txt.remove()

        try:
            label, setup_fn, actions = next(scenario_iter)
            setup_fn()
            current_label = label
            sim.step(scripted_actions=actions)
        except StopIteration:
            # after scripted scenarios finish, just idle with random motion
            sim.step()

        # draw targets as yellow diamonds
        for (tx, ty) in sim.targets:
            diamond = RegularPolygon((tx, ty), numVertices=4, radius=0.3,
                                      orientation=0.785, color='gold')
            ax.add_patch(diamond)

        # draw robots as colored triangles pointing in their direction
        colors = ['royalblue', 'indianred', 'seagreen', 'darkorange', 'purple', 'teal']
        for robot in sim.robots:
            tri = RegularPolygon((robot.x, robot.y), numVertices=3, radius=0.35,
                                  orientation=_orientation_for(robot.direction),
                                  color=colors[robot.id % len(colors)])
            ax.add_patch(tri)
            ax.text(robot.x, robot.y - 0.45, f"R{robot.id}", ha='center', fontsize=7)

        title.set_text(f"Iter {sim.iteration} | scenario: {current_label}")
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(scenarios) + 20,
                                   interval=1200, repeat=False)
    plt.show()

    print("\n--- Illegal condition log ---")
    for line in sim.illegal_log:
        print(line)


def _orientation_for(direction):
    import math
    # RegularPolygon triangle default points up (N); rotate to match direction
    base = {'N': 0, 'E': -90, 'S': 180, 'W': 90}
    return math.radians(base[direction])


if __name__ == "__main__":
    run_visual_demo(num_robots=4, max_iterations=200)