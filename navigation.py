from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import tkinter as tk
from tkinter import font
import threading
import time
import os
import math
import heapq
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

# ─────────────────────────────────────────────
#  GLOBALS
# ─────────────────────────────────────────────
sim          = None
sim_lock     = threading.Lock()   # ZMQ remote API is not thread-safe — serialize all sim.* calls
leftMotor    = None
rightMotor   = None
proxSensor   = None
robotHandle  = None
frontCam     = None
topCam       = None
connected    = False
navigating   = False

robot_x      = 0.0
robot_y      = 0.0
robot_angle  = 0.0
path_history = []
goal_pos     = None
goal_pin     = None   # list of shape handles for the single in-scene X marker on the active goal
status_msg   = "Idle"

# Navigation constants
DRIVE_SPEED      = 1.2
TURN_SPEED       = 1.0
GOAL_THRESHOLD   = 0.3
OBSTACLE_RANGE   = 0.65   # metres — stop and scan for an opening if closer than this (safety-net only now
                           # that the planner routes around known obstacles; still guards against anything
                           # unmapped or moving, e.g. the human figure)
MAX_ACCEL        = 0.12   # max wheel-speed change per tick — smooths out jerky transitions

# Path-planning constants
GRID_CELL        = 0.08   # metres per grid cell used by the A* planner
ROBOT_RADIUS     = 0.22   # Pioneer P3-DX footprint radius (approx)
OBSTACLE_PAD     = 0.10   # extra safety margin added on top of ROBOT_RADIUS when inflating obstacles
WAYPOINT_THRESH  = 0.25   # metres — how close counts as "reached" for an intermediate waypoint

# Obstacle-avoidance behaviour constants
SENSOR_COOLDOWN       = 1.5   # seconds — after a sensor-only (no map corroboration) trigger and
                               # replan, ignore further sensor-only triggers for this long. Stops a
                               # flaky/self-detecting sensor from re-triggering every single tick.
REPLAN_RETRY_INTERVAL = 0.4   # seconds — while rotating to find an opening, retry the planner this
                               # often. As soon as a heading gives a clear route, it breaks out and drives.
ROTATE_TIMEOUT        = 8.0   # seconds — if still no route after rotating this long, flip direction
                               # instead of spinning the same way forever

grid_obstacles = []   # list of (min_x, max_x, min_y, max_y) axis-aligned rectangles — the
                       # real footprint of each obstacle found by scanning the scene. Using
                       # rectangles (not a single bounding circle) matters for long/thin props
                       # like shelving or racks: a circle sized by the diagonal of a 2m-long
                       # shelf balloons into a ~1m-radius blob covering floor space that's
                       # actually walkable. A rectangle tracks the real footprint instead.
grid_bounds    = None # (min_x, max_x, min_y, max_y) — planning/plot extent, from the Floor's bbox
floor_top_z    = 0.02 # world Z of the floor's actual top surface — markers sit just above this.
planned_path   = []   # list of (x, y) waypoints for the CURRENT navigation goal

INSPECTION_TARGETS = {
    "InspectionTarget1": None,
    "InspectionTarget2": None,
    "InspectionTarget3": None,
    "InspectionTarget4": None,
    "InspectionTarget5": None,
}

C_AMBER = "#f4a261"

# ─────────────────────────────────────────────
#  MOTOR CONTROL
# ─────────────────────────────────────────────
_motor_error_logged = False   # only warn once — avoids flooding the log every 50ms

def set_motors(left, right):
    global _motor_error_logged
    try:
        if sim and connected:
            with sim_lock:
                sim.setJointTargetVelocity(leftMotor,  left)
                sim.setJointTargetVelocity(rightMotor, right)
    except Exception as e:
        # This used to fail silently, which looked exactly like "the robot
        # just isn't moving" with no clue why. If this fires, the wheel
        # joints most likely aren't in Velocity control mode in CoppeliaSim
        # (Scene Object Properties → Joint → Motor enabled + Velocity), or
        # leftMotor/rightMotor resolved to a stale/wrong handle.
        if not _motor_error_logged:
            _motor_error_logged = True
            log(f"❌ Motor command failed — wheels won't move: {type(e).__name__}: {e}")
            log("   Check the wheel joints are set to Velocity control mode "
                "(Motor enabled) in CoppeliaSim.")

def stop_motors():
    set_motors(0, 0)

# ─────────────────────────────────────────────
#  GOAL PIN — single physical marker shown only on the active target
# ─────────────────────────────────────────────
def set_goal_pin(gx, gy):
    global goal_pin
    try:
        with sim_lock:
            if goal_pin is not None:
                try:
                    sim.removeObjects(goal_pin)
                except:
                    pass
                goal_pin = None

            bar_size = [0.5, 0.08, 0.02]
            bars = []
            marker_z = floor_top_z + 0.02
            for angle in (math.pi / 4, -math.pi / 4):
                b = sim.createPrimitiveShape(sim.primitiveshape_cuboid, bar_size, 0)
                sim.setObjectPosition(b, [gx, gy, marker_z], -1)
                sim.setObjectOrientation(b, [0, 0, angle], -1)
                sim.setShapeColor(b, None, sim.colorcomponent_ambient_diffuse,
                                   [1.0, 0.1, 0.1])
                sim.setObjectInt32Param(b, sim.shapeintparam_respondable, 0)
                sim.setObjectInt32Param(b, sim.shapeintparam_static, 1)
                sim.setObjectSpecialProperty(b, sim.objectspecialproperty_renderable)
                bars.append(b)

            goal_pin = bars
        log(f"🎯 Goal marker placed at ({gx:.2f}, {gy:.2f}, {marker_z:.2f})")
    except Exception as e:
        log(f"⚠ Could not place goal marker: {e}")

def clear_goal_pin():
    global goal_pin
    try:
        with sim_lock:
            if goal_pin is not None:
                sim.removeObjects(goal_pin)
                goal_pin = None
    except:
        goal_pin = None

# ─────────────────────────────────────────────
#  ROBOT POSE
# ─────────────────────────────────────────────
def get_robot_pose():
    global robot_x, robot_y, robot_angle
    try:
        with sim_lock:
            pos = sim.getObjectPosition(robotHandle, -1)
            ori = sim.getObjectOrientation(robotHandle, -1)
        robot_x     = pos[0]
        robot_y     = pos[1]
        robot_angle = ori[2]
        return robot_x, robot_y, robot_angle
    except:
        return robot_x, robot_y, robot_angle

# ─────────────────────────────────────────────
#  SENSOR
# ─────────────────────────────────────────────
def read_sensor():
    try:
        with sim_lock:
            result = sim.readProximitySensor(proxSensor)
        detected = result[0] == 1
        dist     = result[2][2] if detected else None
        return detected, dist
    except:
        return False, None

# ─────────────────────────────────────────────
#  ANGLE HELPERS
# ─────────────────────────────────────────────
def angle_to_goal(rx, ry, gx, gy):
    return math.atan2(gy - ry, gx - rx)

def angle_diff(target, current):
    diff = target - current
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < -math.pi: diff += 2 * math.pi
    return diff

def dist_to_goal(rx, ry, gx, gy):
    return math.sqrt((gx - rx)**2 + (gy - ry)**2)

# ─────────────────────────────────────────────
#  OBSTACLE MAP
# ─────────────────────────────────────────────
MAX_OBSTACLE_SHAPES = 200
MAX_OBSTACLE_RADIUS = 1.0   # metres — shapes bigger than this (walls, room shell, overhead
                            # gantries, etc.) are treated as scene structure, not obstacles

def build_obstacle_map():
    global grid_obstacles, grid_bounds
    grid_obstacles = []

    try:
        with sim_lock:
            robot_shapes = sim.getObjectsInTree(robotHandle, sim.object_shape_type, 0)
            all_shapes   = sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0)
            floor_handle = None
            try:
                floor_handle = sim.getObject('/Floor')
            except Exception:
                pass
    except Exception as e:
        log(f"⚠ Could not enumerate scene shapes: {e}")
        all_shapes, robot_shapes, floor_handle = [], [], None

    exclude = set(robot_shapes) | {robotHandle}
    if floor_handle is not None:
        exclude.add(floor_handle)

    candidates = [h for h in all_shapes if h not in exclude]
    if len(candidates) > MAX_OBSTACLE_SHAPES:
        log(f"⚠ {len(candidates)} shapes found — capping obstacle scan at "
            f"{MAX_OBSTACLE_SHAPES} to keep connect fast")
        candidates = candidates[:MAX_OBSTACLE_SHAPES]

    log(f"🔍 Scanning {len(candidates)} shape(s) for obstacles...")

    fail_count       = 0
    sample_error     = None
    tiny_count       = 0
    oversized_count  = 0

    for h in candidates:
        try:
            with sim_lock:
                name = sim.getObjectAlias(h, 0)
                pos  = sim.getObjectPosition(h, -1)
                ori  = sim.getObjectOrientation(h, -1)
                bb   = sim.getShapeBB(h)
            if 'floor' in name.lower():
                continue
            if len(bb) >= 2 and isinstance(bb[0], (list, tuple)):
                min_c, max_c = bb[0], bb[1]
                dx, dy = max_c[0] - min_c[0], max_c[1] - min_c[1]
            else:
                dx, dy = bb[0], bb[1]
            # Shape yaw close to 90°/270° means its local X/Y are swapped in
            # world space (e.g. a shelf rotated to run along Y instead of X)
            # — swap the footprint dims so the rectangle still lines up with
            # the object as it actually sits in the scene.
            yaw_deg = math.degrees(ori[2]) % 180
            if 45 < yaw_deg < 135:
                dx, dy = dy, dx
            half_x, half_y = dx / 2.0, dy / 2.0
            longest_dim = max(dx, dy)
            if longest_dim < 0.03:   # ignore tiny/negligible shapes (bolts, decals, etc.)
                tiny_count += 1
                continue
            if longest_dim > MAX_OBSTACLE_RADIUS * 2:
                # Oversized geometry (walls, room shell, ceiling, a conveyor
                # frame spanning the whole scene, etc.) — including these
                # blocks the planner everywhere and paints the plot solid
                # grey, hiding the actual obstacles underneath it.
                oversized_count += 1
                continue
            grid_obstacles.append((pos[0] - half_x, pos[0] + half_x,
                                    pos[1] - half_y, pos[1] + half_y))
        except Exception as e:
            fail_count += 1
            if sample_error is None:
                sample_error = f"{type(e).__name__}: {e}"
            continue

    if fail_count:
        log(f"⚠ {fail_count}/{len(candidates)} shape lookups failed during the "
            f"scan — sample error: {sample_error}")
    if tiny_count:
        log(f"ℹ {tiny_count} shape(s) skipped as too small (<0.03m radius)")
    if oversized_count:
        log(f"ℹ {oversized_count} shape(s) skipped as too large (>{MAX_OBSTACLE_RADIUS:.1f}m radius) "
            f"— treated as scene structure, not obstacles")

    global floor_top_z
    try:
        with sim_lock:
            floor  = sim.getObject('/Floor')
            fpos   = sim.getObjectPosition(floor, -1)
            fx_min = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_x)
            fx_max = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_x)
            fy_min = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_y)
            fy_max = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_y)
            fz_max = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_z)
        grid_bounds = (fpos[0] + fx_min - 0.3, fpos[0] + fx_max + 0.3,
                       fpos[1] + fy_min - 0.3, fpos[1] + fy_max + 0.3)
        floor_top_z = fpos[2] + fz_max
        log(f"ℹ Floor top surface at Z={floor_top_z:.3f}m — goal markers sit just above it")
    except Exception:
        grid_bounds = (-3.0, 3.0, -3.0, 3.0)

    log(f"🗺 Obstacle map built — {len(grid_obstacles)} obstacle(s) found "
        f"(grid X[{grid_bounds[0]:.1f},{grid_bounds[1]:.1f}] "
        f"Y[{grid_bounds[2]:.1f},{grid_bounds[3]:.1f}])")

# ─────────────────────────────────────────────
#  A* PATH PLANNING
# ─────────────────────────────────────────────
def _world_to_cell(x, y):
    min_x, _, min_y, _ = grid_bounds
    col = int((x - min_x) / GRID_CELL)
    row = int((y - min_y) / GRID_CELL)
    return row, col

def _cell_to_world(row, col):
    min_x, _, min_y, _ = grid_bounds
    x = min_x + (col + 0.5) * GRID_CELL
    y = min_y + (row + 0.5) * GRID_CELL
    return x, y

def _point_blocked(x, y):
    inflate = ROBOT_RADIUS + OBSTACLE_PAD
    for min_x, max_x, min_y, max_y in grid_obstacles:
        if (min_x - inflate <= x <= max_x + inflate and
                min_y - inflate <= y <= max_y + inflate):
            return True
    return False

def _point_blocked_check(x, y, min_x, max_x, min_y, max_y):
    inflate = ROBOT_RADIUS + OBSTACLE_PAD
    return (min_x - inflate <= x <= max_x + inflate and
            min_y - inflate <= y <= max_y + inflate)

def _line_clear(p1, p2):
    dist  = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(2, int(dist / (GRID_CELL * 0.5)))
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        if _point_blocked(x, y):
            return False
    return True

def _simplify_path(points):
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not _line_clear(points[i], points[j]):
            j -= 1
        simplified.append(points[j])
        i = j
    return simplified

def plan_path(start_x, start_y, goal_x, goal_y):
    if grid_bounds is None:
        return None

    min_x, max_x, min_y, max_y = grid_bounds
    n_rows = max(1, int((max_y - min_y) / GRID_CELL))
    n_cols = max(1, int((max_x - min_x) / GRID_CELL))

    start = _world_to_cell(start_x, start_y)
    goal  = _world_to_cell(goal_x, goal_y)

    def in_bounds(rc):
        r, c = rc
        return 0 <= r < n_rows and 0 <= c < n_cols

    def passable(rc):
        x, y = _cell_to_world(*rc)
        return not _point_blocked(x, y)

    if not in_bounds(goal):
        return None

    neighbours8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_heap = [(0.0, start)]
    came_from = {}
    g_score   = {start: 0.0}
    visited   = set()

    def h(rc):
        return math.hypot(rc[0] - goal[0], rc[1] - goal[1])

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        for dr, dc in neighbours8:
            nb = (current[0] + dr, current[1] + dc)
            if not in_bounds(nb) or nb in visited:
                continue
            if nb != goal and not passable(nb):
                continue
            step = math.hypot(dr, dc)
            tentative = g_score[current] + step
            if tentative < g_score.get(nb, float('inf')):
                g_score[nb]   = tentative
                came_from[nb] = current
                heapq.heappush(open_heap, (tentative + h(nb), nb))

    if goal != start and goal not in came_from:
        return None

    cells, cur = [goal], goal
    while cur != start:
        cur = came_from.get(cur)
        if cur is None:
            return None
        cells.append(cur)
    cells.reverse()

    waypoints = ([(start_x, start_y)]
                 + [_cell_to_world(r, c) for r, c in cells[1:-1]]
                 + [(goal_x, goal_y)])
    return _simplify_path(waypoints)

# ─────────────────────────────────────────────
#  NAVIGATION LOOP
# ─────────────────────────────────────────────
def navigate(start_x, start_y, goal_x, goal_y, waypoints=None):
    global navigating, status_msg, path_history, goal_pos, planned_path

    path_history = [(start_x, start_y)]
    gx, gy = goal_x, goal_y
    goal_pos = (gx, gy)

    remaining = list(waypoints) if waypoints else [(gx, gy)]
    if remaining and dist_to_goal(remaining[0][0], remaining[0][1], start_x, start_y) < 0.05:
        remaining = remaining[1:]
    if not remaining:
        remaining = [(gx, gy)]

    log(f"Starting navigation")
    log(f"Start: ({start_x:.2f}, {start_y:.2f})")
    log(f"Goal:  ({gx:.2f}, {gy:.2f})  via {len(remaining)} waypoint(s)")

    cur_left, cur_right = 0.0, 0.0
    aligning = False
    align_start_time = 0.0
    ALIGN_STALL_TIMEOUT = 4.0   # seconds — if still pivoting to align after this long,
                                 # something's off (turn direction not matching the
                                 # heading-error sign, drift, etc.) — force out of the
                                 # pivot-in-place and drive-with-correction instead of
                                 # spinning in place indefinitely.

    avoiding                = False
    rotate_dir               = 1
    rotate_start_time        = 0.0
    last_replan_attempt      = 0.0
    sensor_cooldown_until    = 0.0
    sensor_retrigger_count   = 0
    warned_sensor_issue      = False

    while navigating:
        rx, ry, ra = get_robot_pose()
        path_history.append((rx, ry))
        now = time.time()

        wx, wy   = remaining[0]
        is_final = len(remaining) == 1
        threshold = GOAL_THRESHOLD if is_final else WAYPOINT_THRESH

        d_wp = dist_to_goal(rx, ry, wx, wy)

        if d_wp < threshold:
            if is_final:
                stop_motors()
                status_msg = f"✅ Goal reached!"
                log(status_msg)
                root.after(0, lambda: status_label.config(
                    text=status_msg, fg="#00ff99"))
                root.after(0, lambda: nav_btn.config(
                    text="▶ Start Navigation", bg="#006633",
                    command=start_navigation))
                navigating = False
                return
            else:
                remaining = remaining[1:]
                continue

        map_obstacle = False
        for min_x, max_x, min_y, max_y in grid_obstacles:
            ocx, ocy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            if dist_to_goal(ocx, ocy, gx, gy) < 0.5:
                continue   # this obstacle IS essentially the goal — approaching it is expected
            if _point_blocked_check(rx, ry, min_x, max_x, min_y, max_y):
                map_obstacle = True
                break

        detected, sensor_dist = read_sensor()
        sensor_obstacle = detected and sensor_dist is not None and sensor_dist < OBSTACLE_RANGE
        if sensor_obstacle and now < sensor_cooldown_until:
            sensor_obstacle = False

        obstacle_ahead = map_obstacle or sensor_obstacle

        if obstacle_ahead:
            if not avoiding:
                avoiding = True
                rotate_start_time = now
                rotate_dir = 1 if (int(now * 10) % 2 == 0) else -1
                reason = "map — inside an obstacle shadow zone" if map_obstacle else \
                          f"sensor — {sensor_dist:.2f}m ahead"
                m = f"🚧 Obstacle detected ({reason}) — searching for a clear path"
                log(m)
                status_msg = m
                root.after(0, lambda m=m: status_label.config(text=m, fg="#ff4444"))
                last_replan_attempt = 0.0

                if sensor_obstacle and not map_obstacle:
                    sensor_retrigger_count += 1
                    if sensor_retrigger_count == 4 and not warned_sensor_issue:
                        warned_sensor_issue = True
                        log("⚠ The proximity sensor keeps reporting an obstacle with "
                            "nothing corresponding in the scanned map — this usually "
                            "means it's detecting the robot's OWN body (mount clipped "
                            "into the chassis, or the robot not excluded from its "
                            "'Detectable entities'). Worth checking in CoppeliaSim. "
                            "Relying on the obstacle map + turning to search for now.")
                else:
                    sensor_retrigger_count = 0

            if now - last_replan_attempt >= REPLAN_RETRY_INTERVAL:
                last_replan_attempt = now
                new_wp = plan_path(rx, ry, gx, gy)
                if new_wp and len(new_wp) > 1:
                    remaining = new_wp
                    planned_path = new_wp
                    avoiding = False
                    m2 = "✓ Clear path found — resuming"
                    log(m2)
                    root.after(0, lambda m2=m2: status_label.config(text=m2, fg="#00ff99"))

                    if sensor_obstacle and not map_obstacle:
                        sensor_cooldown_until = now + SENSOR_COOLDOWN

                    wx, wy = remaining[0]
                else:
                    if now - rotate_start_time > ROTATE_TIMEOUT:
                        rotate_start_time = now
                        rotate_dir *= -1
                        log("↻ Still no clear route — reversing turn direction")

            if avoiding:
                target_left  = -rotate_dir * TURN_SPEED
                target_right =  rotate_dir * TURN_SPEED
                cur_left  += max(-MAX_ACCEL, min(MAX_ACCEL, target_left  - cur_left))
                cur_right += max(-MAX_ACCEL, min(MAX_ACCEL, target_right - cur_right))
                set_motors(cur_left, cur_right)
                time.sleep(0.05)
                continue

        elif avoiding:
            avoiding = False
            m = "✓ Clear — resuming toward goal"
            log(m)
            status_msg = m
            root.after(0, lambda m=m: status_label.config(text=m, fg="#00ff99"))

        target_angle = angle_to_goal(rx, ry, wx, wy)
        diff         = angle_diff(target_angle, ra)
        ALIGN_ENTER  = 0.35
        ALIGN_EXIT   = 0.15

        if aligning:
            if abs(diff) < ALIGN_EXIT:
                aligning = False
            elif now - align_start_time > ALIGN_STALL_TIMEOUT:
                # Been pivoting to align for too long without reaching the
                # exit threshold — stop trying to nail the heading exactly
                # and fall through to drive-with-correction below, which
                # still makes forward progress toward the goal even with an
                # imperfect heading, instead of spinning in place forever.
                aligning = False
                log("↻ Alignment taking too long — driving with correction instead of pivoting")
        else:
            if abs(diff) > ALIGN_ENTER:
                aligning = True
                align_start_time = now

        if aligning:
            turn_dir  = 1 if diff > 0 else -1
            turn_mag  = min(1.0, abs(diff) / (math.pi / 2)) * TURN_SPEED
            turn_mag  = max(0.3, turn_mag)
            target_left  = -turn_dir * turn_mag
            target_right =  turn_dir * turn_mag
        else:
            correction   = max(-1.0, min(1.0, diff / (math.pi / 6))) * 0.4 * TURN_SPEED
            target_left  = DRIVE_SPEED - correction
            target_right = DRIVE_SPEED + correction

        d_goal = dist_to_goal(rx, ry, gx, gy)
        leg = "" if is_final else f" (leg {len(waypoints) - len(remaining) + 1}/{len(waypoints)})"
        m = f"Heading to goal — {d_goal:.2f}m remaining{leg}"
        status_msg = m
        root.after(0, lambda m=m: status_label.config(text=m, fg="#00ccff"))

        cur_left  += max(-MAX_ACCEL, min(MAX_ACCEL, target_left  - cur_left))
        cur_right += max(-MAX_ACCEL, min(MAX_ACCEL, target_right - cur_right))

        set_motors(cur_left, cur_right)
        time.sleep(0.05)

    stop_motors()

# ─────────────────────────────────────────────
#  START / STOP NAVIGATION
# ─────────────────────────────────────────────
def start_navigation():
    global navigating, planned_path

    if not connected:
        log("❌ Not connected!")
        return
    if navigating:
        log("⚠ Already navigating!")
        return

    target_name = target_var.get()
    target_pos  = INSPECTION_TARGETS.get(target_name)
    if target_pos is None:
        log(f"❌ Could not find {target_name}")
        return

    gx, gy = target_pos
    rx, ry, _ = get_robot_pose()

    set_goal_pin(gx, gy)

    log("🧭 Planning path around known obstacles...")
    wp = plan_path(rx, ry, gx, gy)
    if wp:
        planned_path = wp
        log(f"✓ Path planned — {len(wp)} waypoint(s)")
    else:
        planned_path = [(rx, ry), (gx, gy)]
        log("⚠ No clear planned route found — starting anyway; it will turn "
            "and search for an opening as it goes")

    log(f"🚀 Navigating to {target_name}")
    navigating = True
    nav_btn.config(text="⏹ Stop", bg="#cc0000", command=stop_navigation)

    threading.Thread(
        target=navigate, args=(rx, ry, gx, gy, planned_path), daemon=True).start()

def stop_navigation():
    global navigating, goal_pos, planned_path
    navigating = False
    goal_pos = None
    planned_path = []
    clear_goal_pin()
    stop_motors()
    log("⏹ Navigation stopped")
    nav_btn.config(text="▶ Start Navigation", bg="#006633",
                   command=start_navigation)
    root.after(0, lambda: status_label.config(
        text="Navigation stopped.", fg="#aaaaaa"))
    root.after(0, on_target_selected)

# ─────────────────────────────────────────────
#  TARGET PREVIEW
# ─────────────────────────────────────────────
def on_target_selected(*_args):
    global goal_pos
    if not connected or navigating:
        return
    target_name = target_var.get()
    target_pos  = INSPECTION_TARGETS.get(target_name)
    if target_pos is None:
        log(f"⚠ {target_name} has no known position (wasn't found in the scene at connect)")
        return
    gx, gy = target_pos
    goal_pos = (gx, gy)
    set_goal_pin(gx, gy)
    log(f"📍 Marker moved to {target_name} at ({gx:.2f}, {gy:.2f}) — press Start Navigation to go there")
    root.after(0, update_path_plot)

# ─────────────────────────────────────────────
#  MANUAL CONTROL
# ─────────────────────────────────────────────
MANUAL_DRIVE_SPEED = 1.2
MANUAL_TURN_SPEED  = 1.0

def manual_drive(left, right):
    global navigating
    if navigating:
        stop_navigation()
    if not connected:
        log("❌ Not connected!")
        return
    set_motors(left, right)
    root.after(0, lambda: status_label.config(
        text="🕹 Manual control", fg="#ffcc00"))

def manual_release(event=None):
    stop_motors()

# ─────────────────────────────────────────────
#  EMERGENCY STOP
# ─────────────────────────────────────────────
def emergency_stop():
    global navigating, goal_pos, planned_path
    navigating = False
    goal_pos = None
    planned_path = []
    clear_goal_pin()
    stop_motors()
    log("🛑 EMERGENCY STOP — all motion halted")
    status_label.config(text="🛑 EMERGENCY STOP", fg="#ff0000")
    nav_btn.config(text="▶ Start Navigation", bg="#006633",
                   command=start_navigation)
    root.after(0, on_target_selected)

# ─────────────────────────────────────────────
#  PATH PLOT — matches the reference UI: dark navy canvas, grey obstacle
#  blobs, yellow dashed Planned route, cyan Robot trail, green Start dot,
#  cyan Robot dot, red star Goal with a yellow T# label, legend top-right.
# ─────────────────────────────────────────────
def update_path_plot():
    try:
        fig, ax = plt.subplots(figsize=(2.8, 2.8), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        for (min_x, max_x, min_y, max_y) in grid_obstacles:
            body = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                  color="#666666", alpha=0.75, zorder=1, linewidth=0)
            ax.add_patch(body)

        if len(planned_path) > 1:
            pxs = [p[0] for p in planned_path]
            pys = [p[1] for p in planned_path]
            ax.plot(pxs, pys, color="#ffcc00", linewidth=1.2, linestyle='--',
                     alpha=0.85, zorder=2, label="Planned")

        if len(path_history) > 1:
            xs = [p[0] for p in path_history]
            ys = [p[1] for p in path_history]
            ax.plot(xs, ys, color="#00ccff", linewidth=1.5, alpha=0.8, zorder=3)
            ax.plot(xs[0],  ys[0],  'go', markersize=8,  label="Start", zorder=4)
            ax.plot(xs[-1], ys[-1], 'co', markersize=6,  label="Robot", zorder=4)

        if goal_pos:
            active_label = None
            for name, pos in INSPECTION_TARGETS.items():
                if pos and abs(pos[0]-goal_pos[0]) < 1e-6 and abs(pos[1]-goal_pos[1]) < 1e-6:
                    active_label = f"T{name[-1]}"
                    break

            ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=14,
                     markeredgecolor="white", markeredgewidth=0.6,
                     label="Goal", zorder=5)
            if active_label:
                ax.annotate(active_label, goal_pos, xytext=(0, 8),
                            textcoords='offset points',
                            color="#ffff00", fontsize=8, fontweight='bold',
                            ha='center', va='bottom', zorder=6)

        if grid_bounds:
            min_x, max_x, min_y, max_y = grid_bounds
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title("Path", color="white", fontsize=8, pad=3)
        ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=7)
        ax.set_ylabel("Y (m)", color="#aaaaaa", fontsize=7)
        ax.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="#444",
                  labelcolor="white", loc="upper right")

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                    facecolor="#1a1a2e")
        buf.seek(0)
        img   = Image.open(buf).resize((230, 230))
        photo = ImageTk.PhotoImage(img)
        path_label.config(image=photo)
        path_label.image = photo
        plt.close(fig)
        buf.close()
    except Exception as e:
        log(f"⚠ Path plot failed to render: {e}")

# ─────────────────────────────────────────────
#  CLOSE SIMULATION
# ─────────────────────────────────────────────
def close_simulation():
    global connected, navigating
    if not connected:
        log("⚠ Not connected — nothing to stop")
        return
    try:
        if navigating:
            stop_navigation()
        stop_motors()
        clear_goal_pin()
        with sim_lock:
            sim.stopSimulation()
        connected = False
        log("⏹ Simulation stopped")
        root.after(0, lambda: connect_btn.config(
            text="▶  CONNECT & START SIMULATION", state='normal', bg="#006633",
            command=lambda: threading.Thread(target=connect, daemon=True).start()))
        root.after(0, lambda: close_btn.config(state='disabled'))
        root.after(0, lambda: status_label.config(
            text="Simulation stopped.", fg="#aaaaaa"))
        root.after(0, lambda: front_cam_label.config(image='', text="Waiting..."))
        root.after(0, lambda: top_cam_label.config(image='', text="Waiting..."))
    except Exception as e:
        log(f"⚠ Could not stop simulation: {e}")

# ─────────────────────────────────────────────
#  INSPECTION PHOTO — snapshot the front camera to disk
# ─────────────────────────────────────────────
PHOTO_DIR = os.path.join(os.getcwd(), "inspection_photos")

def take_inspection_photo():
    if not connected:
        log("❌ Not connected — can't take a photo")
        return
    try:
        with sim_lock:
            img, res = sim.getVisionSensorImg(frontCam)
        arr = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        arr = np.flipud(arr)
        pil = Image.fromarray(arr)

        os.makedirs(PHOTO_DIR, exist_ok=True)
        fname = f"inspection_{time.strftime('%Y%m%d_%H%M%S')}.png"
        fpath = os.path.join(PHOTO_DIR, fname)
        pil.save(fpath)

        log(f"📸 Inspection photo saved: {fpath}")
        root.after(0, lambda: status_label.config(
            text=f"📸 Photo saved: {fname}", fg="#ffcc00"))
    except Exception as e:
        log(f"⚠ Could not save inspection photo: {e}")

# ─────────────────────────────────────────────
#  CONNECT
# ─────────────────────────────────────────────
def connect():
    global sim, leftMotor, rightMotor, proxSensor
    global robotHandle, frontCam, topCam, connected

    root.after(0, lambda: connect_btn.config(
        text="Connecting...", state='disabled', bg="#555555"))
    log("Connecting to CoppeliaSim...")

    try:
        client = RemoteAPIClient()
        with sim_lock:
            sim         = client.getObject('sim')
            leftMotor   = sim.getObject('/PioneerP3DX/leftMotor')
            rightMotor  = sim.getObject('/PioneerP3DX/rightMotor')
            proxSensor  = sim.getObject('/PioneerP3DX/proximitySensor')
            robotHandle = sim.getObject('/PioneerP3DX')
            frontCam    = sim.getObject('/PioneerP3DX/front_camera')
            topCam      = sim.getObject('/top_camera')

            for name in INSPECTION_TARGETS:
                try:
                    handle = sim.getObject(f'/{name}')
                    pos    = sim.getObjectPosition(handle, -1)
                    INSPECTION_TARGETS[name] = (pos[0], pos[1])
                    log(f"✓ {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
                except:
                    log(f"⚠ {name} not found in scene")

            sim.startSimulation()
        time.sleep(0.5)
        connected = True

        build_obstacle_map()

        root.after(0, lambda: connect_btn.config(
            text="✓ Connected", bg="#006600", state='disabled'))
        root.after(0, lambda: close_btn.config(state='normal'))
        root.after(0, lambda: status_label.config(
            text="Connected. Select a target and press Start Navigation.",
            fg="#00ff99"))
        log("✓ Connected! Simulation started.")

        root.after(0, on_target_selected)

        threading.Thread(target=camera_loop, daemon=True).start()
        threading.Thread(target=pose_loop,   daemon=True).start()

    except Exception as e:
        root.after(0, lambda: connect_btn.config(
            text="⚠ Retry", state='normal', bg="#cc6600"))
        log(f"❌ Failed: {e}")

# ─────────────────────────────────────────────
#  BACKGROUND LOOPS
# ─────────────────────────────────────────────
def pose_loop():
    while True:
        if connected:
            get_robot_pose()
            root.after(0, update_path_plot)
            root.after(0, lambda: pose_label.config(
                text=f"Position: ({robot_x:.2f}, {robot_y:.2f})  "
                     f"Angle: {math.degrees(robot_angle):.1f}°"))
        time.sleep(0.2)

def camera_loop():
    while True:
        if connected:
            for cam, lbl in [(frontCam, front_cam_label),
                             (topCam,   top_cam_label)]:
                try:
                    with sim_lock:
                        img, res = sim.getVisionSensorImg(cam)
                    arr = np.frombuffer(img, dtype=np.uint8).reshape(
                        res[1], res[0], 3)
                    arr = np.flipud(arr)
                    pil = Image.fromarray(arr).resize((200, 200))
                    fi  = ImageTk.PhotoImage(pil)
                    root.after(0, lambda i=fi, l=lbl: (
                        l.config(image=i), setattr(l, 'image', i)))
                except:
                    pass
        time.sleep(0.12)

# ─────────────────────────────────────────────
#  LOG
# ─────────────────────────────────────────────
def log(text):
    try:
        activity_log.config(state='normal')
        activity_log.insert('end', f"{text}\n")
        activity_log.see('end')
        activity_log.config(state='disabled')
    except:
        pass

# ─────────────────────────────────────────────
#  BUILD UI  — matches reference screenshot: dark navy theme, title bar,
#  connect bar, 3-column content (cameras | path+controls | status+log)
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("CW2 — Autonomous Navigation System")
root.configure(bg="#1a1a2e")
root.geometry("1050x720")
root.resizable(False, False)

DARK  = "#1a1a2e"
PANEL = "#16213e"
ACCENT= "#0f3460"
GREEN = "#00ff99"
WHITE = "#ffffff"
GREY  = "#aaaaaa"

title_font = font.Font(family="Helvetica", size=13, weight="bold")
label_font = font.Font(family="Helvetica", size=10)
btn_font   = font.Font(family="Helvetica", size=11, weight="bold")
small_font = font.Font(family="Helvetica", size=9)

# Title
tf = tk.Frame(root, bg=ACCENT, pady=8)
tf.pack(fill='x')
tk.Label(tf,
    text="🤖  CW2 — AUTONOMOUS NAVIGATION SYSTEM  |  A* Path Planning + Reactive Avoidance  |  Industrial Inspection",
    bg=ACCENT, fg=WHITE, font=title_font).pack()

# Connect bar
cf = tk.Frame(root, bg="#0a0a1a", pady=5)
cf.pack(fill='x', padx=10)
tk.Label(cf, text="CoppeliaSim:", bg="#0a0a1a", fg=GREY,
         font=label_font).pack(side='left', padx=(0,6))
connect_btn = tk.Button(cf, text="▶  CONNECT & START SIMULATION",
    bg="#006633", fg=WHITE, font=btn_font, relief='flat',
    cursor="hand2", padx=10,
    command=lambda: threading.Thread(target=connect, daemon=True).start())
connect_btn.pack(side='left')
close_btn = tk.Button(cf, text="⏹  CLOSE SIMULATION",
    bg="#661a1a", fg=WHITE, font=btn_font, relief='flat',
    cursor="hand2", padx=10, state='disabled',
    command=lambda: threading.Thread(target=close_simulation, daemon=True).start())
close_btn.pack(side='left', padx=(8,0))
pose_label = tk.Label(cf, text="Position: —", bg="#0a0a1a",
                      fg=GREY, font=small_font)
pose_label.pack(side='right', padx=10)

# Content
content = tk.Frame(root, bg=DARK)
content.pack(fill='both', expand=True, padx=8, pady=6)

# Left — cameras
lc = tk.Frame(content, bg=DARK)
lc.pack(side='left', fill='y', padx=(0,6))
tk.Label(lc, text="📷 Front Camera", bg=DARK, fg=GREEN,
         font=label_font).pack(anchor='w')
front_cam_label = tk.Label(lc, bg="black", width=200, height=200,
                            text="Waiting...", fg=GREY)
front_cam_label.pack()
tk.Label(lc, text="", bg=DARK, height=1).pack()
tk.Label(lc, text="🛰 Top Camera", bg=DARK, fg=GREEN,
         font=label_font).pack(anchor='w')
top_cam_label = tk.Label(lc, bg="black", width=200, height=200,
                          text="Waiting...", fg=GREY)
top_cam_label.pack()

# Middle — path + controls
mc = tk.Frame(content, bg=DARK)
mc.pack(side='left', fill='y', padx=(0,6))
tk.Label(mc, text="🗺 Path Visualisation", bg=DARK, fg=GREEN,
         font=label_font).pack(anchor='w')
path_label = tk.Label(mc, bg="#16213e", width=230, height=230,
                       text="No path yet", fg=GREY)
path_label.pack()

nf = tk.Frame(mc, bg=PANEL, padx=8, pady=8)
nf.pack(fill='x', pady=(8,0))
tk.Label(nf, text="Select Inspection Target:", bg=PANEL, fg=WHITE,
         font=label_font).pack(anchor='w')
target_var = tk.StringVar(value="InspectionTarget1")
target_var.trace_add('write', on_target_selected)
for t in ["InspectionTarget1", "InspectionTarget2", "InspectionTarget3",
          "InspectionTarget4", "InspectionTarget5"]:
    tk.Radiobutton(nf, text=t, variable=target_var, value=t,
                   bg=PANEL, fg=WHITE, selectcolor=ACCENT,
                   activebackground=PANEL, font=small_font).pack(anchor='w')
tk.Label(nf, text="", bg=PANEL).pack()
nav_btn = tk.Button(nf, text="▶ Start Navigation",
    bg="#006633", fg=WHITE, font=btn_font, width=22, height=2,
    relief='flat', cursor="hand2", command=start_navigation)
nav_btn.pack()
photo_btn = tk.Button(nf, text="📸 Take Inspection Photo",
    bg=ACCENT, fg=WHITE, font=btn_font, width=22, height=1,
    relief='flat', cursor="hand2",
    command=lambda: threading.Thread(target=take_inspection_photo, daemon=True).start())
photo_btn.pack(pady=(6,0))

af = tk.Frame(mc, bg=PANEL, padx=8, pady=6)
af.pack(fill='x', pady=(6,0))
tk.Label(af, text="Algorithm: A* Path Planning + Reactive Backup", bg=PANEL,
         fg=C_AMBER, font=label_font).pack(anchor='w')
tk.Label(af, text="Drive speed: 1.2 m/s", bg=PANEL, fg=GREY,
         font=small_font).pack(anchor='w')
tk.Label(af, text="Goal threshold: 0.30m", bg=PANEL, fg=GREY,
         font=small_font).pack(anchor='w')
tk.Label(af, text="Obstacle range (safety net): 0.65m", bg=PANEL, fg=GREY,
         font=small_font).pack(anchor='w')
tk.Label(af, text="Grid cell: 0.08m  |  Inflation: 0.32m", bg=PANEL, fg=GREY,
         font=small_font).pack(anchor='w')

# Right — status + log
rc = tk.Frame(content, bg=DARK)
rc.pack(side='left', fill='both', expand=True)

estop_btn = tk.Button(rc, text="🛑  EMERGENCY STOP", bg="#ff0000", fg=WHITE,
    font=btn_font, height=2, relief='flat', cursor="hand2",
    activebackground="#aa0000", command=emergency_stop)
estop_btn.pack(fill='x', pady=(0,6))

# Manual control D-pad
mpf = tk.Frame(rc, bg=PANEL, padx=8, pady=6)
mpf.pack(fill='x', pady=(0,6))
tk.Label(mpf, text="🕹 Manual Control (press & hold)", bg=PANEL, fg=WHITE,
         font=label_font).pack(anchor='w', pady=(0,4))

pad = tk.Frame(mpf, bg=PANEL)
pad.pack()

def _mk_manual_btn(parent, text, left, right, row, col):
    b = tk.Button(parent, text=text, bg=ACCENT, fg=WHITE, font=btn_font,
                   width=3, height=1, relief='flat', cursor="hand2")
    b.bind("<ButtonPress-1>",   lambda e: manual_drive(left, right))
    b.bind("<ButtonRelease-1>", manual_release)
    b.grid(row=row, column=col, padx=3, pady=3)
    return b

_mk_manual_btn(pad, "▲", MANUAL_DRIVE_SPEED,  MANUAL_DRIVE_SPEED,  0, 1)
_mk_manual_btn(pad, "◀", -MANUAL_TURN_SPEED,  MANUAL_TURN_SPEED,   1, 0)
stop_pad_btn = tk.Button(pad, text="■", bg="#555555", fg=WHITE, font=btn_font,
    width=3, height=1, relief='flat', cursor="hand2", command=manual_release)
stop_pad_btn.grid(row=1, column=1, padx=3, pady=3)
_mk_manual_btn(pad, "▶", MANUAL_TURN_SPEED,   -MANUAL_TURN_SPEED,  1, 2)
_mk_manual_btn(pad, "▼", -MANUAL_DRIVE_SPEED, -MANUAL_DRIVE_SPEED, 2, 1)

sf = tk.Frame(rc, bg=PANEL, padx=10, pady=8)
sf.pack(fill='x', pady=(0,6))
tk.Label(sf, text="Navigation Status", bg=PANEL, fg=C_AMBER,
         font=btn_font).pack(anchor='w')
status_label = tk.Label(sf, text="Not connected", bg=PANEL, fg=GREY,
                         font=label_font, wraplength=320)
status_label.pack(anchor='w')

lf = tk.LabelFrame(rc, text=" 📋  Activity Log ", bg=DARK,
                   fg=GREEN, font=label_font)
lf.pack(fill='both', expand=False)
activity_log = tk.Text(lf, height=12, bg="#0d0d1a", fg=GREEN,
    font=("Courier", 9), state='disabled', relief='flat', wrap='word')
activity_log.pack(fill='both', expand=True, padx=4, pady=4)

tk.Label(root,
    text="CW2 — Autonomous Navigation  |  A* Path Planning + Reactive Avoidance  |  Pioneer P3-DX  |  CoppeliaSim EDU",
    bg=ACCENT, fg=GREY, font=small_font, anchor='w', padx=10
).pack(fill='x', side='bottom')

log("System ready. Press CONNECT to start.")
log("Algorithm: scans the scene for obstacles, plans an A* route around them, "
    "then turns and re-plans on the fly if it ever hits a blocked path.")

root.mainloop()
stop_motors()