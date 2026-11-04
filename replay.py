#!/usr/bin/env python3
"""Replay a MediaPipe hand-tracking CSV on the G1/Inspire hand in MuJoCo.

The CSV supplies finger shape and camera-relative palm orientation.
MediaPipe world landmarks are used for hand shape; image landmarks (x, y and
relative z) are used for the camera-relative palm frame. The controller maps
the five digits to the Inspire hand's six actuators and uses wrist IK to follow
the recorded palm rotation while keeping the arm, waist and wrist translation
fixed.

Usage:
    python replay.py --csv output_csv/rfingers.csv
    python replay.py --csv output_csv/rfingers.csv --slow 1.5
    python replay.py --csv output_csv/rfingers.csv --scene path/to/scene_hand_only.xml

The waist, shoulders and elbow are held at their initial poses. Wrist
orientation is retargeted from the CSV; wrist translation stays fixed. The
same anatomical axis convention is used for left and right hands; no arbitrary
left-hand reflection is applied.
Only --csv is required; --scene and --slow are optional.
"""

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


FINGERS = ('index', 'middle', 'ring', 'pinky', 'thumb')
BASES = np.array([5, 9, 13, 17, 1])
MIDS = BASES + 1
TIPS = np.array([8, 12, 16, 20, 4])

# Fixed, deliberately small set of internal filtering constants.
PALM_CONDITION_MIN = 0.08
SHAPE_FRAME_OUTLIER_DEG = 38.0
SHAPE_FRAME_RATE_LIMIT_DEG_S = 180.0
ORIENTATION_FRAME_OUTLIER_DEG = 30.0
ORIENTATION_FRAME_RATE_LIMIT_DEG_S = 120.0
FRAME_MEDIAN_WINDOW = 11
LANDMARK_MEDIAN_WINDOW = 5
ORIENTATION_CUTOFF_HZ = 3.0
ORIENTATION_RATE_DEG_S = 120.0
MAX_TRACKING_GAP_S = 1.0


# -----------------------------------------------------------------------------
# Geometry and filtering
# -----------------------------------------------------------------------------
def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError('Degenerate hand landmark geometry.')
    return np.asarray(v, dtype=float) / n


def palm_frame(wrist, index, middle, pinky, ring=None,
               min_condition=PALM_CONDITION_MIN):
    """Orthonormal 3-D palm frame from anatomical landmarks.

    Columns are across-palm, wrist-to-knuckles, and palm-normal axes.
    The conditioning ratio rejects frames where the two defining directions
    become nearly parallel, avoiding unstable normal-vector flips.
    """
    x = unit(np.asarray(index, float) - np.asarray(pinky, float))
    if ring is None:
        mcp_centre = np.asarray(middle, float)
    else:
        mcp_centre = (np.asarray(index, float) + np.asarray(middle, float)
                      + np.asarray(ring, float) + np.asarray(pinky, float)) / 4.0
    y0 = mcp_centre - np.asarray(wrist, float)
    y_perp = y0 - x * float(np.dot(x, y0))
    condition = float(np.linalg.norm(y_perp) / max(np.linalg.norm(y0), 1e-12))
    if condition < min_condition:
        raise ValueError('Poorly conditioned palm frame.')
    y = unit(y_perp)
    z = unit(np.cross(x, y))
    y = unit(np.cross(z, x))
    H = np.column_stack((x, y, z))
    if np.linalg.det(H) < 0.0:
        H[:, 2] *= -1.0
    return H, condition


def rotation_vector(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    if q[0] < 0.0:
        q *= -1.0
    s = float(np.linalg.norm(q[1:]))
    return np.zeros(3) if s < 1e-10 else q[1:] / s * (2.0 * np.arctan2(s, q[0]))


def rotation_matrix(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.eye(3)
    x, y, z = np.asarray(v, float) / angle
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotation_z(angle):
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_distance_deg(A, B):
    return float(np.degrees(np.linalg.norm(rotation_vector(A.T @ B))))


def blend_rotation(a, b, u):
    return a @ rotation_matrix(float(u) * rotation_vector(a.T @ b))


def minjerk(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u**3 * (10.0 - 15.0*u + 6.0*u*u)


def rolling_median(values, window):
    x = np.asarray(values, float)
    if len(x) < 3 or window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(x, (radius, radius), mode='edge')
    return np.array([np.median(padded[i:i+window]) for i in range(len(x))])


def rolling_median_vectors(values, window):
    a = np.asarray(values, float)
    if len(a) < 3 or window <= 1:
        return a.copy()
    flat = a.reshape(len(a), -1)
    filtered = np.column_stack([rolling_median(flat[:, j], window)
                                for j in range(flat.shape[1])])
    return filtered.reshape(a.shape)


def stabilise_frames(times, frames, window=FRAME_MEDIAN_WINDOW,
                      outlier_deg=ORIENTATION_FRAME_OUTLIER_DEG,
                      max_rate_deg_s=ORIENTATION_FRAME_RATE_LIMIT_DEG_S):
    """Reject short palm-frame flips and limit physically implausible rotation rates.

    The local SO(3) medoid rejects isolated or short clusters of landmark
    failures. The rate limit is applied using source-video timestamps, so it
    remains independent of the MuJoCo playback speed.
    """
    ts = np.asarray(times, float)
    src = np.asarray(frames, float)
    if len(src) < 2:
        return src.copy(), 0

    if window % 2 == 0:
        window += 1
    radius = window // 2
    stable = src.copy()
    replaced = np.zeros(len(src), dtype=bool)

    for i in range(len(src)):
        lo, hi = max(0, i-radius), min(len(src), i+radius+1)
        # Do not let unrelated samples across a tracking gap vote together.
        indices = np.arange(lo, hi)
        indices = indices[np.abs(ts[indices] - ts[i]) <= 0.20]
        neighbours = src[indices]
        if len(neighbours) < 3:
            continue

        costs = np.array([
            sum(rotation_distance_deg(candidate, other) for other in neighbours)
            for candidate in neighbours
        ])
        medoid = neighbours[int(np.argmin(costs))]
        if rotation_distance_deg(src[i], medoid) > outlier_deg:
            stable[i] = medoid
            replaced[i] = True

    # Apply continuity after outlier rejection. This does not create a motion
    # sequence; it only prevents a noisy estimate from commanding an instant
    # rotation that the recorded hand could not physically make.
    for i in range(1, len(stable)):
        dt = max(float(ts[i] - ts[i-1]), 1e-6)
        delta = rotation_vector(stable[i-1].T @ stable[i])
        limit = math.radians(max_rate_deg_s * dt)
        norm = float(np.linalg.norm(delta))
        if norm > limit > 0.0:
            stable[i] = stable[i-1] @ rotation_matrix(delta * (limit / norm))

    return stable, int(replaced.sum())


# -----------------------------------------------------------------------------
# Human CSV -> hand pose
# -----------------------------------------------------------------------------
class Demonstration:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        if 'time_s' not in df:
            raise ValueError('CSV is missing time_s.')
        self.times = df.time_s.to_numpy(float)
        if (len(df) < 3 or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0.0)):
            raise ValueError('CSV needs at least 3 rows with strictly increasing timestamps.')

        world_cols = [f'world_lm_{i}_{a}' for i in range(21) for a in 'xyz']
        image_cols = [f'lm_{i}_{a}' for i in range(21) for a in 'xyz']
        missing = sorted((set(world_cols) | set(image_cols)) - set(df.columns))
        if missing:
            raise ValueError(f'CSV is missing required MediaPipe landmark columns: {missing[:8]}')

        detect_col = 'hand_detected' if 'hand_detected' in df else 'detected'
        if detect_col not in df:
            raise ValueError('CSV needs hand_detected or detected.')

        if 'handedness' in df:
            labels = df.handedness.fillna('').astype(str).str.lower().to_numpy()
            self.hand_side = 'left' if np.sum(labels == 'left') > np.sum(labels == 'right') else 'right'
            detected = (labels == self.hand_side) | (labels == '')
        else:
            self.hand_side = 'right'
            detected = np.ones(len(df), dtype=bool)

        points = df[world_cols].to_numpy(float).reshape(-1, 21, 3)
        image = df[image_cols].to_numpy(float).reshape(-1, 21, 3)
        valid = ((df[detect_col].to_numpy() == 1) & detected
                 & np.isfinite(points).all(axis=(1, 2))
                 & np.isfinite(image).all(axis=(1, 2)))
        if 'world_detected' in df:
            valid &= df.world_detected.to_numpy() == 1

        mirrored = (df.input_mirrored.to_numpy() == 1 if 'input_mirrored' in df
                    else np.zeros(len(df), dtype=bool))
        points[mirrored, :, 0] *= -1.0
        image_width = df.image_width.to_numpy(float)
        image_height = df.image_height.to_numpy(float)
        if not np.all(np.isfinite(image_width) & np.isfinite(image_height)
                      & (image_width > 0.0) & (image_height > 0.0)):
            raise ValueError('CSV has invalid image dimensions.')
        image[mirrored, :, 0] = 1.0 - image[mirrored, :, 0]
        # Image-landmark x/y are normalized image coordinates; z is a
        # wrist-relative depth estimate on approximately the same scale as x.
        # Convert y to the x-axis scale so (x, y, z) is an isotropic 2.5-D
        # camera-relative coordinate system.
        image[:, :, 1] *= (image_height / image_width)[:, None]

        # Estimate a reliable 3-D palm frame for finger-shape coordinates.
        raw_frames = np.tile(np.eye(3), (len(df), 1, 1))
        frame_good = np.zeros(len(df), dtype=bool)
        lengths = np.zeros((len(df), 5), dtype=float)
        proportions_good = np.zeros(len(df), dtype=bool)
        for i in np.flatnonzero(valid):
            p = points[i]
            finger_lengths = np.array([
                sum(np.linalg.norm(p[j+1] - p[j]) for j in range(base, base+3))
                for base in BASES
            ])
            palm_width = float(np.linalg.norm(p[5] - p[17]))
            if (palm_width < 1e-7 or not np.isfinite(finger_lengths).all()
                    or np.any(finger_lengths / palm_width < 0.2)
                    or np.any(finger_lengths / palm_width > 5.0)):
                continue
            proportions_good[i] = True
            lengths[i] = finger_lengths
            try:
                raw_frames[i], _ = palm_frame(p[0], p[5], p[9], p[17], ring=p[13])
                frame_good[i] = True
            except ValueError:
                pass

        frame_ids = np.flatnonzero(frame_good)
        if len(frame_ids) < 3:
            raise ValueError('Could not obtain enough stable 3-D palm frames from this CSV.')
        stable_frames, shape_corrected_count = stabilise_frames(
            self.times[frame_ids], raw_frames[frame_ids],
            window=7, outlier_deg=SHAPE_FRAME_OUTLIER_DEG,
            max_rate_deg_s=SHAPE_FRAME_RATE_LIMIT_DEG_S)
        raw_frames[frame_ids] = stable_frames

        # Fill occasional bad palm-frame samples by interpolating nearby valid frames.
        tracking_ids = np.flatnonzero(valid & proportions_good)
        if len(tracking_ids) < 3:
            raise ValueError('Too few valid hand-tracking frames.')
        shape_frames = np.empty((len(df), 3, 3), dtype=float)
        for i in tracking_ids:
            k = int(np.searchsorted(frame_ids, i, side='left'))
            if k <= 0:
                shape_frames[i] = raw_frames[frame_ids[0]]
            elif k >= len(frame_ids):
                shape_frames[i] = raw_frames[frame_ids[-1]]
            else:
                a, b = int(frame_ids[k-1]), int(frame_ids[k])
                u = (self.times[i] - self.times[a]) / max(self.times[b] - self.times[a], 1e-9)
                shape_frames[i] = blend_rotation(raw_frames[a], raw_frames[b], u)

        self.tip_vectors = np.zeros((len(df), 5, 3), dtype=float)
        self.mid_directions = np.zeros((len(df), 5, 3), dtype=float)
        for i in tracking_ids:
            p, H = points[i], shape_frames[i]
            self.tip_vectors[i] = ((p[TIPS] - p[BASES]) @ H) / np.maximum(lengths[i, :, None], 1e-8)
            mid = (p[MIDS] - p[BASES]) @ H
            self.mid_directions[i] = mid / np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-8)

        # Smooth landmark noise without changing the movement sequence.
        self.tip_vectors[tracking_ids] = rolling_median_vectors(self.tip_vectors[tracking_ids],
                                                                LANDMARK_MEDIAN_WINDOW)
        self.mid_directions[tracking_ids] = rolling_median_vectors(self.mid_directions[tracking_ids],
                                                                   LANDMARK_MEDIAN_WINDOW)
        mid_norms = np.linalg.norm(self.mid_directions[tracking_ids], axis=2, keepdims=True)
        self.mid_directions[tracking_ids] /= np.maximum(mid_norms, 1e-8)

        # Finger shape and global palm orientation must use different signals:
        # - world_lm_* gives a useful metric hand-centred shape for finger
        #   retargeting;
        # - lm_x/lm_y/lm_z gives image/camera-relative position and depth for
        #   estimating how the palm turns relative to the camera.
        #
        # Do NOT derive camera orientation directly from world_lm_*: those
        # landmarks are centred on the hand and are not a room-fixed camera
        # pose. Do NOT use only atan2(index_mcp - pinky_mcp): that only tracks
        # in-image roll and cannot represent palm tilt toward/away from camera.
        camera_raw = np.tile(np.eye(3), (len(df), 1, 1))
        camera_good = np.zeros(len(df), dtype=bool)
        camera_width = np.zeros(len(df), dtype=float)
        for i in tracking_ids:
            p = image[i]
            camera_width[i] = np.linalg.norm(p[5] - p[17])
            try:
                camera_raw[i], _ = palm_frame(
                    p[0], p[5], p[9], p[17], ring=p[13],
                    min_condition=PALM_CONDITION_MIN)
                camera_good[i] = camera_width[i] > 0.003
            except ValueError:
                pass

        orientation_ids = tracking_ids[camera_good[tracking_ids]]
        if len(orientation_ids) < 3:
            raise ValueError('Too few valid camera-space palm frames in the CSV.')

        # Correct isolated and short clustered tracking flips in the camera-
        # relative palm frame. Then interpolate rejected/invalid frames between
        # neighbouring accepted samples in SO(3).
        camera_stable, orientation_corrected_count = stabilise_frames(
            self.times[orientation_ids], camera_raw[orientation_ids],
            window=FRAME_MEDIAN_WINDOW,
            outlier_deg=ORIENTATION_FRAME_OUTLIER_DEG,
            max_rate_deg_s=ORIENTATION_FRAME_RATE_LIMIT_DEG_S)

        self.orientation_frames = np.empty((len(df), 3, 3), dtype=float)
        for i in tracking_ids:
            k = int(np.searchsorted(orientation_ids, i, side='left'))
            if k <= 0:
                self.orientation_frames[i] = camera_stable[0]
            elif k >= len(orientation_ids):
                self.orientation_frames[i] = camera_stable[-1]
            elif orientation_ids[k] == i:
                self.orientation_frames[i] = camera_stable[k]
            else:
                a, b = int(orientation_ids[k-1]), int(orientation_ids[k])
                gap = float(self.times[b] - self.times[a])
                if gap <= 0.20:
                    u = (self.times[i] - self.times[a]) / max(gap, 1e-9)
                    self.orientation_frames[i] = blend_rotation(
                        camera_stable[k-1], camera_stable[k], u)
                else:
                    # A long camera-tracking gap: hold the nearest trusted pose
                    # rather than inventing a rotation across missing data.
                    self.orientation_frames[i] = (
                        camera_stable[k-1] if self.times[i]-self.times[a] <=
                        self.times[b]-self.times[i] else camera_stable[k])

        self.ids = tracking_ids
        self.start = float(self.times[self.ids[0]])
        self.end = float(self.times[self.ids[-1]])
        neutral_idx = int(self.ids[0])
        self.neutral_orientation = self.orientation_frames[neutral_idx].copy()
        self.valid_count = len(self.ids)
        self.rejected_count = len(df) - len(self.ids)
        self.shape_corrected_count = shape_corrected_count
        self.orientation_corrected_count = orientation_corrected_count
        self.frame_corrected_count = shape_corrected_count + orientation_corrected_count
        print(f'CSV: {len(df)} rows; {self.valid_count} valid {self.hand_side}-hand frames; '
              f'{self.rejected_count} rejected; corrected {shape_corrected_count} shape-frame and '
              f'{orientation_corrected_count} camera-orientation outliers.')

    def sample(self, video_time):
        valid_times = self.times[self.ids]
        k = int(np.searchsorted(valid_times, video_time, side='right') - 1)
        k = max(0, min(k, len(self.ids)-1))
        a = int(self.ids[k])
        age = float(video_time - self.times[a])
        if age > MAX_TRACKING_GAP_S:
            raise ValueError(f'Hand tracking missing for {age:.2f}s at video time {video_time:.2f}s.')
        if k + 1 < len(self.ids):
            b = int(self.ids[k+1])
            gap = float(self.times[b] - self.times[a])
            if gap <= 0.20 + 1e-9:
                u = float(np.clip(age / gap, 0.0, 1.0))
                H = blend_rotation(self.orientation_frames[a], self.orientation_frames[b], u)
                tips = (1-u)*self.tip_vectors[a] + u*self.tip_vectors[b]
                mids = (1-u)*self.mid_directions[a] + u*self.mid_directions[b]
                mids /= np.maximum(np.linalg.norm(mids, axis=1, keepdims=True), 1e-8)
                return H, tips, mids
        return (self.orientation_frames[a].copy(), self.tip_vectors[a].copy(),
                self.mid_directions[a].copy())


# -----------------------------------------------------------------------------
# MuJoCo robot model and IK
# -----------------------------------------------------------------------------
class Robot:
    def __init__(self, scene, hand_side):
        self.hand_side = hand_side
        cap = 'L' if hand_side == 'left' else 'R'
        arm_prefix = 'left_' if hand_side == 'left' else 'right_'
        self.actuator_names = (
            f'{cap}_index_proximal', f'{cap}_middle_proximal', f'{cap}_ring_proximal',
            f'{cap}_pinky_proximal', f'{cap}_thumb_proximal_pitch', f'{cap}_thumb_proximal_yaw'
        )
        # This replay changes hand orientation at a fixed wrist location.
        # Keep shoulder, elbow and waist at their initial servo targets; use
        # the three wrist joints for the orientation task.
        self.arm_names = (
            f'{arm_prefix}wrist_roll',
            f'{arm_prefix}wrist_pitch',
            f'{arm_prefix}wrist_yaw'
        )

        try:
            self.m = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        except Exception as exc:
            raise ValueError(f'Cannot load scene or referenced robot meshes: {exc}') from exc
        m = self.m
        self.d, self.arm_data, self.hand_data = (mujoco.MjData(m) for _ in range(3))

        def lookup(kind, name):
            idx = mujoco.mj_name2id(m, kind, name)
            if idx < 0:
                raise ValueError(f'Model is missing required object: {name}')
            return idx

        self.body = lambda name: lookup(mujoco.mjtObj.mjOBJ_BODY, name)
        self.joint = lambda name: lookup(mujoco.mjtObj.mjOBJ_JOINT, name)
        self.actuator = lambda name: lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.hand = self.body(f'{cap}_hand_base_link')
        pelvis = self.body('pelvis')

        # Apply gravity compensation only to robot links, never the world/cube.
        self.robot_bodies = set()
        for b in range(1, m.nbody):
            ancestor = b
            while ancestor and ancestor != pelvis:
                ancestor = m.body_parentid[ancestor]
            if ancestor == pelvis:
                self.robot_bodies.add(b)
                m.body_gravcomp[b] = 1.0
        for j in range(m.njnt):
            if (m.jnt_bodyid[j] in self.robot_bodies
                    and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError('This replay expects the fixed-pelvis G1 model.')

        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Convert body torque motors to in-memory PD position servos. Keep their
        # original limits as torque/force limits and keep the waist at its start pose.
        for a in range(m.nu):
            j = m.actuator_trnid[a, 0]
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or j < 0:
                raise ValueError('Expected direct joint actuators in the supplied model.')
            if m.jnt_bodyid[j] not in self.robot_bodies:
                continue
            if (m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE
                    or not np.allclose(m.actuator_gear[a], [1, 0, 0, 0, 0, 0])):
                raise ValueError('Expected unit-gear hinge actuators.')
            name = m.actuator(a).name
            if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE:
                if m.actuator_ctrllimited[a]:
                    m.actuator_forcelimited[a] = True
                    m.actuator_forcerange[a] = m.actuator_ctrlrange[a].copy()
                kp, kd = ((60.0, 3.0) if 'wrist' in name else
                          ((300.0, 15.0) if ('shoulder' in name or 'elbow' in name)
                           else (400.0, 20.0)))
                m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
                m.actuator_gainprm[a, 0] = kp
                m.actuator_biasprm[a, :3] = [0.0, -kp, -kd]
                m.actuator_ctrllimited[a] = True
                m.actuator_ctrlrange[a] = m.jnt_range[j]
            self.d.ctrl[a] = self.d.qpos[m.jnt_qposadr[j]]

        self.fa = np.array([self.actuator(n) for n in self.actuator_names], dtype=int)
        self.fj = m.actuator_trnid[self.fa, 0]
        self.fq = m.jnt_qposadr[self.fj]
        self.finger_low = np.maximum(m.actuator_ctrlrange[self.fa, 0], m.jnt_range[self.fj, 0])
        self.finger_high = np.minimum(m.actuator_ctrlrange[self.fa, 1], m.jnt_range[self.fj, 1])
        for a in self.fa:
            m.actuator_gainprm[a, 0] = 3.0
            m.actuator_biasprm[a, :3] = [0.0, -3.0, -0.15]

        self.aj = np.array([self.joint(n + '_joint') for n in self.arm_names], dtype=int)
        self.aq = m.jnt_qposadr[self.aj]
        self.av = m.jnt_dofadr[self.aj]
        self.aa = np.array([self.actuator(n) for n in self.arm_names], dtype=int)
        self.arm_low, self.arm_high = m.jnt_range[self.aj].T
        self.arm_target = self.d.qpos[self.aq].copy()
        self.finger_target = self.d.qpos[self.fq].copy()

        # Read the Inspire hand's equality constraints so FK respects its coupled links.
        self.couplings = []
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]:
                j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
                if m.joint(j1).name.startswith(f'{cap}_'):
                    if j2 < 0:
                        raise ValueError('Unexpected single-joint hand equality.')
                    self.couplings.append((int(m.jnt_qposadr[j1]), int(m.jnt_qposadr[j2]),
                                           m.eq_data[e, :5].copy()))

        self.base_bodies = ([self.body(f'{cap}_{f}_proximal') for f in FINGERS[:4]]
                            + [self.body(f'{cap}_thumb_proximal_base')])
        self.mid_bodies = ([self.body(f'{cap}_{f}_intermediate') for f in FINGERS[:4]]
                           + [self.body(f'{cap}_thumb_proximal')])
        end_bodies = ([self.body(f'{cap}_{f}_intermediate') for f in FINGERS[:4]]
                      + [self.body(f'{cap}_thumb_distal')])
        self.tip_geoms = []
        for b in end_bodies:
            found = [g for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
                     if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[g] == 0]
            if len(found) != 1:
                raise ValueError('Expected one non-colliding fingertip marker per digit in the XML.')
            self.tip_geoms.append(found[0])

        mujoco.mj_forward(m, self.d)
        R = self.d.xmat[self.hand].reshape(3, 3)
        origin = self.d.xpos[self.hand].copy()
        base_local = (self.d.xpos[self.base_bodies] - origin) @ R
        self.C, _ = palm_frame(np.zeros(3), base_local[0], base_local[1], base_local[3],
                               ring=base_local[2], min_condition=0.02)
        self.Rneutral = R.copy()
        self.Pneutral = self.Rneutral @ self.C
        self.wrist_position = origin.copy()  # Keep wrist translation fixed during replay.
        self.anchors = base_local @ self.C
        mid_local = (self.d.xpos[self.mid_bodies] - origin) @ R @ self.C
        tip_local = (self.d.geom_xpos[self.tip_geoms] - origin) @ R @ self.C
        self.mid_lengths = np.linalg.norm(mid_local - self.anchors, axis=1)
        self.lengths = self.mid_lengths + np.linalg.norm(tip_local - mid_local, axis=1)

        thumb_chain = [self.d.xpos[self.base_bodies[4]], self.d.xpos[self.mid_bodies[4]],
                       self.d.body(f'{cap}_thumb_intermediate').xpos,
                       self.d.body(f'{cap}_thumb_distal').xpos,
                       tip_local[4] @ self.C.T @ R.T + origin]
        self.lengths[4] = sum(np.linalg.norm(thumb_chain[i+1] - thumb_chain[i])
                              for i in range(len(thumb_chain)-1))

        self.dt = max(1, round(0.02 / m.opt.timestep)) * m.opt.timestep
        self.substeps = max(1, round(0.02 / m.opt.timestep))
        self.viewer = None
        self.wall = time.monotonic()
        self.desired_R = self.Rneutral.copy()
        self.target_tips = tip_local.copy()

    def positions(self, data):
        R = data.xmat[self.hand].reshape(3, 3)
        origin = data.xpos[self.hand]
        tips = (data.geom_xpos[self.tip_geoms] - origin) @ R @ self.C
        mids = (data.xpos[self.mid_bodies] - origin) @ R @ self.C
        return tips, mids

    def hand_fk(self, q):
        k, m = self.hand_data, self.m
        k.qpos[:] = self.d.qpos
        k.qpos[self.fq] = q
        for dependent, driver, coef in self.couplings:
            x = k.qpos[driver] - m.qpos0[driver]
            k.qpos[dependent] = m.qpos0[dependent] + np.polynomial.polynomial.polyval(x, coef)
        mujoco.mj_kinematics(m, k)
        return self.positions(k)

    def hand_ik(self, target_tips, target_mids):
        previous = self.finger_target.copy()

        def residual(q):
            tips, mids = self.hand_fk(q)
            return np.r_[((tips-target_tips)/0.08).ravel(),
                         (0.3*(mids-target_mids)/0.08).ravel(),
                         (0.2*((tips[4]-tips[:2])-(target_tips[4]-target_tips[:2]))/0.08).ravel(),
                         0.025*(q-previous)]

        q = previous.copy()
        for _ in range(8):
            error = residual(q)
            J = np.column_stack([(residual(q + np.eye(6)[i]*1e-4) - error)/1e-4
                                 for i in range(6)])
            dq = np.linalg.solve(J.T @ J + 1e-4*np.eye(6), -J.T @ error)
            dq = np.clip(dq, -0.25, 0.25)
            accepted = False
            for step in (1.0, 0.5, 0.25):
                candidate = np.clip(q + step*dq, self.finger_low, self.finger_high)
                if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                    q, accepted = candidate, True
                    break
            if not accepted or np.linalg.norm(dq) < 1e-5:
                break

        max_step = 2.0 * self.dt
        self.finger_target = np.clip(previous + np.clip(q-previous, -max_step, max_step),
                                     self.finger_low, self.finger_high)
        self.d.ctrl[self.fa] = self.finger_target
        tips, _ = self.hand_fk(q)
        return float(np.sqrt(np.mean(np.sum((tips-target_tips)**2, axis=1))))

    def arm_ik(self, desired_R):
        """Track CSV palm orientation using only the three wrist joints.

        There is no position/orientation over-constrained 6-D solve here:
        the wrist translation is not commanded, and shoulder/elbow/waist
        remain at their initial targets. The wrist Jacobian solves only the
        3-D orientation error.
        """
        m, k = self.m, self.arm_data
        k.qpos[:] = self.d.qpos
        k.qpos[self.aq] = self.arm_target
        _, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(15):
            mujoco.mj_kinematics(m, k)
            mujoco.mj_comPos(m, k)
            actual_R = k.xmat[self.hand].reshape(3, 3)
            error = rotation_vector(desired_R @ actual_R.T)
            if np.linalg.norm(error) < 1e-5:
                break
            jp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, k, jp, jr, self.hand)
            Jw = jr[:, self.av]
            delta = Jw.T @ np.linalg.solve(
                Jw @ Jw.T + 2e-4 * np.eye(3), error)
            k.qpos[self.aq] = np.clip(
                k.qpos[self.aq] + np.clip(delta, -0.04, 0.04),
                self.arm_low, self.arm_high)

        # Human motion is rate-limited in the recorded-time domain before
        # reaching this IK. This additional joint-target slew limit avoids
        # sharp servo target jumps.
        self.arm_target += np.clip(
            k.qpos[self.aq] - self.arm_target, -1.5*self.dt, 1.5*self.dt)
        self.d.ctrl[self.aa] = self.arm_target

    def start_viewer(self):
        from mujoco import viewer
        self.viewer = viewer.launch_passive(self.m, self.d)
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = [0.28, -0.12, 1.05]
            self.viewer.cam.distance = 0.9
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -15
        self.wall = time.monotonic()

    def step(self):
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)
        mujoco.mj_forward(self.m, self.d)
        if not np.isfinite(self.d.qpos).all() or np.max(np.abs(self.d.qvel)) > 100.0:
            raise RuntimeError('Unstable simulation; replay stopped.')
        if self.viewer:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            # No custom user-scene geoms: render only the robot and scene.
            # (The green fingertip spheres in the previous version were
            # visualization markers, not physical objects or actuators.)
            self.viewer.sync()
            self.wall += self.dt
            time.sleep(max(0.0, self.wall - time.monotonic()))
            if self.wall < time.monotonic() - 1.0:
                self.wall = time.monotonic()

    def hold_final_frame(self):
        if not self.viewer:
            return
        print('Replay finished. Close the MuJoCo window to exit.')
        # The last rendered frame is already visible. Do not keep calling
        # sync() while idle: that is unnecessary and can complicate shutdown.
        while self.viewer.is_running():
            time.sleep(0.05)


# -----------------------------------------------------------------------------
# Replay
# -----------------------------------------------------------------------------
def run(csv_path, scene_path, slow):
    if not np.isfinite(slow) or slow <= 0.0:
        raise ValueError('--slow must be a positive number.')

    demo = Demonstration(csv_path)
    robot = Robot(scene_path, demo.hand_side)
    alpha = 1.0 - math.exp(-2.0 * math.pi * ORIENTATION_CUTOFF_HZ * robot.dt)
    max_rotation_step = math.radians(ORIENTATION_RATE_DEG_S * robot.dt)

    try:
        robot.start_viewer()

        # Neutral pose: initialise the arm and hand at the first valid sample.
        H, vectors, directions = demo.sample(demo.start)
        target_tips = robot.anchors + vectors * robot.lengths[:, None]
        target_mids = robot.anchors + directions * robot.mid_lengths[:, None]
        relative = demo.neutral_orientation.T @ H
        raw = rotation_vector(relative)
        desired_R = robot.Pneutral @ rotation_matrix(raw) @ robot.C.T
        filtered_R = robot.Rneutral.copy()
        filtered_tips = robot.positions(robot.d)[0].copy()
        filtered_mids = robot.positions(robot.d)[1].copy()
        robot.target_tips = filtered_tips.copy()
        robot.desired_R = filtered_R.copy()

        # Smoothly initialise the targets rather than snapping to the first frame.
        for i in range(max(1, round(1.0 / robot.dt))):
            u = (i + 1) / max(1, round(1.0 / robot.dt))
            filtered_R = blend_rotation(robot.Rneutral, desired_R, minjerk(u))
            filtered_tips += minjerk(u) * (target_tips - filtered_tips)
            filtered_mids += minjerk(u) * (target_mids - filtered_mids)
            robot.arm_ik(filtered_R)
            robot.hand_ik(filtered_tips, filtered_mids)
            robot.desired_R = filtered_R.copy()
            robot.target_tips = filtered_tips.copy()
            robot.step()

        duration = (demo.end - demo.start) * slow
        steps = int(math.ceil(duration / robot.dt)) + 1
        for i in range(steps):
            video_time = min(demo.end, demo.start + i * robot.dt / slow)
            H, vectors, directions = demo.sample(video_time)

            # Retarget the camera-relative palm rotation measured from the
            # CSV's x/y/depth landmarks. Both hands use the same anatomical
            # axes (index-minus-pinky, wrist-to-MCP, cross product); no
            # left/right sign flip is applied. Only noisy/impossible-rate
            # estimates are filtered; the movement sequence remains data-driven.
            relative = demo.neutral_orientation.T @ H
            raw = rotation_vector(relative)
            requested_R = robot.Pneutral @ rotation_matrix(raw) @ robot.C.T
            candidate_R = blend_rotation(filtered_R, requested_R, alpha)
            delta = rotation_vector(filtered_R.T @ candidate_R)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_rotation_step > 0.0:
                candidate_R = filtered_R @ rotation_matrix(delta * (max_rotation_step / delta_norm))
            filtered_R = candidate_R

            requested_tips = robot.anchors + vectors * robot.lengths[:, None]
            requested_mids = robot.anchors + directions * robot.mid_lengths[:, None]
            filtered_tips += alpha * (requested_tips - filtered_tips)
            filtered_mids += alpha * (requested_mids - filtered_mids)
            mid_norms = np.linalg.norm(filtered_mids, axis=1, keepdims=True)
            filtered_mids /= np.maximum(mid_norms, 1e-8)

            robot.arm_ik(filtered_R)
            robot.hand_ik(filtered_tips, filtered_mids)
            robot.desired_R = filtered_R.copy()
            robot.target_tips = filtered_tips.copy()
            robot.step()

        print(f'Replay complete: {steps} frames; {demo.valid_count} valid CSV frames; '
              f'{demo.shape_corrected_count} shape-frame and '
              f'{demo.orientation_corrected_count} camera-orientation outliers corrected; '
              f'speed={slow:g}x.')
        robot.hold_final_frame()
    finally:
        if robot.viewer:
            robot.viewer.close()


def main():
    parser = argparse.ArgumentParser(description='Replay a MediaPipe hand CSV on the G1/Inspire hand.')
    parser.add_argument('--csv', required=True, help='MediaPipe landmark CSV.')
    parser.add_argument('--scene', default='unitree_mujoco/unitree_robots/g1/scene_hand_only.xml',
                        help='MuJoCo scene XML (default: scene_hand_only.xml).')
    parser.add_argument('--slow', type=float, default=1.0,
                        help='Playback speed relative to the source recording; e.g. 1.5 is 1.5x slower.')
    args = parser.parse_args()
    try:
        run(args.csv, args.scene, args.slow)
    except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
