# Compare the maxnorm sweep, row-paired (same held-out rows and CEM seeds in every config).
#   python -I sweep_analyze.py <dir with ppe_maxnorm_*.json and *_rows.npz>
import glob
import json
import os
import sys

import numpy as np

d = sys.argv[1]
MAX_STEP = 0.10 * 8 / 30  # the arm server's executed-step cap (0.10 m/s x one model step)
runs = []
for js in sorted(glob.glob(os.path.join(d, "ppe_maxnorm_*.json")), key=lambda p: -float(p.split("_")[-1][:-5])):
    rows = js.replace(".json", "_rows.npz")
    if not os.path.exists(rows):
        continue
    res, z = json.load(open(js)), np.load(rows)
    runs.append((res["planner"]["maxnorm"], res, {k: z[k] for k in z.files}))
if not runs:
    sys.exit("no finished configs yet")
ref_rows = runs[0][2]["row"]
for mn, _, r in runs:
    assert np.array_equal(r["row"], ref_rows), "configs must share rows"

true = runs[0][2]["true"]
t3 = true[:, :3] * 1e3
tmag = np.linalg.norm(t3, axis=1)
mov, still = tmag >= 5, tmag < 2
print(f"{len(ref_rows)} held-out rows; expert step |xyz|: median {np.median(tmag):.1f} mm, p90 "
      f"{np.percentile(tmag, 90):.1f}, max {tmag.max():.1f}; moving (>=5 mm) {mov.sum()}, still (<2 mm) {still.sum()}")
hdr = (f"{'maxnorm':>8} | {'yz dir ok':>9} {'yz <45deg':>9} {'yz err/|t|':>10} | {'3D cos>0':>8} {'3D cos med':>10} | "
       f"{'|plan| med':>10} {'ratio med':>9} {'at bound':>8} | {'exec err':>8} {'still |p|':>9} | {'grip acc':>8} "
       f"{'grip@.75':>8}")
print(hdr + "\n" + "-" * len(hdr))
for mn, res, r in runs:
    o = res["overall"]
    p3 = r["plan"][:, :3] * 1e3
    pmag = np.linalg.norm(p3, axis=1)
    cos3 = np.sum(p3[mov] * t3[mov], 1) / (pmag[mov] * tmag[mov] + 1e-9)
    at_bound = np.mean(np.any(np.abs(r["plan"][:, :3]) >= 0.99 * mn, axis=1))
    scale = np.minimum(1.0, MAX_STEP * 1e3 / np.maximum(pmag, 1e-9))  # what the server would execute
    exec_err = np.linalg.norm(p3 * scale[:, None] - t3, axis=1)
    grip_sat = np.mean(np.abs(r["plan"][:, 6]) >= 0.74)
    print(f"{mn * 1e3:6.0f}mm | {o['direction_correct']:9.2f} {o['direction_within_45deg']:9.2f} "
          f"{o['yz_err_over_zero_median']:10.2f} | {np.mean(cos3 > 0):8.2f} {np.median(cos3):10.2f} | "
          f"{np.median(pmag):8.1f}mm {np.median(pmag[mov] / tmag[mov]):9.2f} {at_bound:8.2f} | "
          f"{np.median(exec_err):6.1f}mm {np.median(pmag[still]):7.1f}mm | {o['gripper_sign_accuracy']:8.2f} "
          f"{grip_sat:8.2f}")
print("\nby goal (3D cos>0 on moving rows / median exec err mm):")
for g in (1, 2, 3):
    m = runs[0][2]["goal"] == g
    cells = []
    for mn, _, r in runs:
        p3 = r["plan"][:, :3] * 1e3
        pmag = np.linalg.norm(p3, axis=1)
        mm = m & mov
        cos3 = np.sum(p3[mm] * t3[mm], 1) / (pmag[mm] * tmag[mm] + 1e-9)
        scale = np.minimum(1.0, MAX_STEP * 1e3 / np.maximum(pmag, 1e-9))
        err = np.linalg.norm(p3 * scale[:, None] - t3, axis=1)[m]
        cells.append(f"{mn * 1e3:.0f}mm {np.mean(cos3 > 0):.2f}/{np.median(err):.1f}")
    print(f"  goal {g} ({m.sum()} rows, {(m & mov).sum()} moving): " + " | ".join(cells))

print("\nmedian error of the executed step (mm) if each planned step were capped at c mm (rows: maxnorm):")
caps = [26.7, 16.7, 10.0, 8.0, 5.0]  # server caps: --max_speed 0.10 / 0.0625 / 0.03
print(f"{'':>8}" + "".join(f"{f'cap {c:g}':>10}" for c in caps) + f"{'expert size':>13}{'zero step':>11}")
for mn, _, r in runs:
    p3, t3_ = r["plan"][:, :3] * 1e3, r["true"][:, :3] * 1e3
    pm, tm = np.linalg.norm(p3, axis=1), np.linalg.norm(t3_, axis=1)
    cells = [np.median(np.linalg.norm(p3 * np.minimum(1, c / np.maximum(pm, 1e-9))[:, None] - t3_, axis=1)) for c in caps]
    oracle = np.linalg.norm(p3 / np.maximum(pm, 1e-9)[:, None] * tm[:, None] - t3_, axis=1)  # planned direction, expert size
    print(f"{mn * 1e3:6.0f}mm" + "".join(f"{c:10.1f}" for c in cells) + f"{np.median(oracle):13.1f}{np.median(tm):11.1f}")
