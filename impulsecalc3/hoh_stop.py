"""Credit-stop for HOH mesher. Import this; do not re-derive.

One live remesh + one spec repair, then FAIL.json. No Grok, no zipper loop.
"""

from __future__ import annotations

MAX_LIVE_REMESH = 1
MAX_REPAIR = 1
MESH_CLOCK_S = 180.0
N_CELLS_CAP = 70_000
N_CELLS_MIN = 20_000
MAX_NONORTHO_DEG = 50.0
RECTANGLE_YSTDEV_OVER_PITCH = 0.05
BAN_GROK_BUILD = True
BAN_EXECUTOR_UNTIL_PASS = True


class HohStop(RuntimeError):
    """Second remesh / second repair / rectangle outer — stop the turn."""


def assert_not_rectangle(ystdev_over_pitch: float) -> None:
    if float(ystdev_over_pitch) <= RECTANGLE_YSTDEV_OVER_PITCH:
        raise HohStop(
            f"rectangle outer: stdev(P_bot.y)/pitch={ystdev_over_pitch:.4g} "
            f"<= {RECTANGLE_YSTDEV_OVER_PITCH}. Do not TFI. Do not checkMesh."
        )


def assert_budget(n_cells: int, mesh_seconds: float) -> None:
    if int(n_cells) > N_CELLS_CAP:
        raise HohStop(f"n_cells={n_cells} > {N_CELLS_CAP}")
    if float(mesh_seconds) > MESH_CLOCK_S:
        raise HohStop(f"mesh clock {mesh_seconds:.1f}s > {MESH_CLOCK_S}s")


def assert_repair_budget(repair_count: int) -> None:
    if int(repair_count) > MAX_REPAIR:
        raise HohStop("repair budget spent. Write FAIL.json. Do not remesh again.")
