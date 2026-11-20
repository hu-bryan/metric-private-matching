"""Render the analytic two-point swap calibration figure on a compute node."""

from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    epsilon = np.geomspace(0.125, 16, 1000)
    radius = np.log(3) / (2 * epsilon)
    # Verify the displayed boundary and the SONAR calibration values.
    factor = np.exp(2 * epsilon * radius)
    assert np.allclose(factor / (1 + factor), 0.75)
    assert np.isclose(np.log(3) / 32, 0.03433163402087843)

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.97)
    ax.fill_between(epsilon, 0, radius, color="#4477AA", alpha=0.13, lw=0)
    ax.plot(epsilon, radius, color="#4477AA", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlim(0.125, 16)
    ax.set_ylim(0, 4.5)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8, 16],
                  ["0.125", "0.25", "0.5", "1", "2", "4", "8", "16"])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.minorticks_off()
    ax.set_xlabel(r"Privacy rate $\varepsilon$")
    ax.set_ylabel(r"Distance $r$")
    ax.grid(axis="y", color=".93", lw=0.5)
    import sys
    target = Path(sys.argv[1]) / "privacy_swap_calibration"
    for extension in ("pdf", "png"):
        fig.savefig(target.with_suffix("." + extension), dpi=230,
                    bbox_inches="tight", pad_inches=0.035,
                    metadata={"CreationDate": None, "ModDate": None}
                    if extension == "pdf" else None)
    plt.close(fig)
    print("Calibration boundary and numerical checks passed")


if __name__ == "__main__":
    main()
