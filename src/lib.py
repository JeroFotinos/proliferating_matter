from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from core import SquareLattice, SimulationResult, CultureState

def plot_population_curves(results: list[SimulationResult], labels: list[str]) -> None:
    """
    Plot cell count versus time for several simulations.
    """
    plt.figure(figsize=(8, 5))

    for result, label in zip(results, labels):
        plt.step(result.times, result.cell_counts, where="post", label=label)

    plt.xlabel("Time")
    plt.ylabel("Number of occupied sites")
    plt.title("Cell culture growth")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_snapshots(
    lattice: SquareLattice,
    result: SimulationResult,
    times_to_show: list[float],
    title_prefix: str = "",
) -> None:
    """
    Plot occupancy snapshots with an explicit binary colormap.

    Convention:
    - 0 = empty   -> white
    - 1 = occupied -> black
    """
    n = len(times_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, ts in zip(axes[0], times_to_show):
        arr = lattice.reshape_state(result.snapshots[float(ts)]).astype(int)

        ax.imshow(
            arr,
            origin="lower",
            interpolation="nearest",
            cmap="binary",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{title_prefix}t = {ts:g}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.tight_layout()
    plt.show()


def plot_birth_time(
    lattice: SquareLattice,
    state: CultureState,
    title: str = "Birth time of occupied sites",
) -> None:
    """
    Plot the birth-time field. Empty sites are masked.
    """
    birth = state.birth_time.copy()
    img = lattice.reshape_state(birth)

    masked = np.ma.masked_invalid(img)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(masked, origin="lower", interpolation="nearest")
    plt.colorbar(im, label="Birth time")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()


def plot_cell_id(
    lattice: SquareLattice,
    state: CultureState,
    title: str = "Cell id map",
) -> None:
    """
    Plot cell ids for occupied sites. Empty sites are masked.
    """
    values = state.cell_id.astype(float)
    values[values < 0] = np.nan
    img = lattice.reshape_state(values)

    masked = np.ma.masked_invalid(img)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(masked, origin="lower", interpolation="nearest")
    plt.colorbar(im, label="Cell id")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()


def summarize_simulation(result: SimulationResult, model_name: str) -> None:
    """
    Print a compact textual summary.
    """
    print(f"Model: {model_name}")
    print(f"Final time: {result.times[-1]:.6f}")
    print(f"Initial cells: {result.cell_counts[0]}")
    print(f"Final cells: {result.cell_counts[-1]}")
    print(f"Number of events: {result.event_count}")


# ======= Animation utilities =======


def animate_snapshots(
    lattice: SquareLattice,
    result: SimulationResult,
    times_to_show: list[float],
    interval: int = 100,
    title_prefix: str = "",
    cmap: str = "binary",
):
    """
    Create a matplotlib animation from stored occupancy snapshots.

    Parameters
    ----------
    lattice : SquareLattice
        Lattice used in the simulation.
    result : SimulationResult
        Simulation output containing snapshots.
    times_to_show : list[float]
        Ordered list of times to animate.
    interval : int, default=100
        Delay between frames in milliseconds.
    title_prefix : str, default=""
        Prefix added to the title of each frame.
    cmap : str, default="binary"
        Colormap for occupancy. With 'binary':
        - 0 = empty -> white
        - 1 = occupied -> black

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object.
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    first_frame = lattice.reshape_state(result.snapshots[float(times_to_show[0])]).astype(int)

    im = ax.imshow(
        first_frame,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        animated=True,
    )

    title = ax.set_title(f"{title_prefix}t = {times_to_show[0]:.3f}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    def update(frame_idx: int):
        t = float(times_to_show[frame_idx])
        arr = lattice.reshape_state(result.snapshots[t]).astype(int)
        im.set_array(arr)
        title.set_text(f"{title_prefix}t = {t:.3f}")
        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(times_to_show),
        interval=interval,
        blit=False,
        repeat=False,
    )

    return anim, fig, ax


def save_gif(
    lattice: SquareLattice,
    result: SimulationResult,
    times_to_show: list[float],
    filename: str,
    fps: int = 10,
    title_prefix: str = "",
    cmap: str = "binary",
) -> None:
    """
    Save a GIF animation from stored occupancy snapshots.

    Parameters
    ----------
    lattice : SquareLattice
        Lattice used in the simulation.
    result : SimulationResult
        Simulation output containing snapshots.
    times_to_show : list[float]
        Ordered list of times to animate.
    filename : str
        Output GIF filename.
    fps : int, default=10
        Frames per second in the output GIF.
    title_prefix : str, default=""
        Prefix added to the title of each frame.
    cmap : str, default="binary"
        Colormap for occupancy.
    """
    filename = str(Path(filename))

    anim, fig, ax = animate_snapshots(
        lattice=lattice,
        result=result,
        times_to_show=times_to_show,
        interval=int(1000 / fps),
        title_prefix=title_prefix,
        cmap=cmap,
    )

    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)


def save_mp4(
    lattice: SquareLattice,
    result: SimulationResult,
    times_to_show: list[float],
    filename: str,
    fps: int = 10,
    title_prefix: str = "",
    cmap: str = "binary",
) -> None:
    """
    Save an MP4 animation from stored occupancy snapshots.

    Notes
    -----
    This requires ffmpeg to be installed and available in PATH.
    """
    from matplotlib.animation import FFMpegWriter

    filename = str(Path(filename))

    anim, fig, ax = animate_snapshots(
        lattice=lattice,
        result=result,
        times_to_show=times_to_show,
        interval=int(1000 / fps),
        title_prefix=title_prefix,
        cmap=cmap,
    )

    writer = FFMpegWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)