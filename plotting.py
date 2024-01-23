import h5py
from matplotlib import pyplot as plt
import numpy as np 

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

FULL_WIDTH = 10

TITLE_SIZE = 16
LABEL_SIZE = 14
LEGEND_SIZE = 14
TEXT_SIZE = 12
TICK_SIZE = 12

LABEL_TRANS = "ln(Transmissivity) [ln(m$^{2}$ day$^{-1}$)]"

CMAP_TRANS = "turbo"

nx = 61
xmin = 0
xmax = 6000

def read_setup(fname):
    with h5py.File(fname, "r") as f:
        coords = f["locations"][:]
    return coords 

def read_prior(fname):

    with h5py.File(fname, "r") as f:
        results = {
            "particles" : f["particles"][:, :],
            "stds" : f["stds"][:, :]
        }

    results["particles"] = [
        np.reshape(p, (nx, nx)) for p in results["particles"]
    ]

    return results

def read_design(fname):
    with h5py.File(fname, "r") as f:
        sensors = f["sensors"][:]
    return sensors

def plot_stds(stds, fname, cmap="turbo"):

    figsize = (0.5 * FULL_WIDTH, 0.5 * FULL_WIDTH)
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_box_aspect(1)
    ax.axis("off")
    ax.pcolormesh(stds, cmap=cmap)

    plt.tight_layout()
    plt.savefig(fname)

def plot_particles(particles, fname, cmap="turbo", vmin=0.0, vmax=5.0):
    
    # print(np.min([np.min(p) for p in particles]))
    # print(np.max([np.max(p) for p in particles]))

    figsize = (FULL_WIDTH, 0.52 * FULL_WIDTH)
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    for i, ax in enumerate(axes.flat):

        ax.axis("off")
        ax.set_box_aspect(1)

        ax.pcolormesh(particles[i], cmap=cmap, vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.savefig(fname)

def plot_design(sensor_coords, sensors_opt, fname):

    figsize=(0.5 * FULL_WIDTH, 0.5 * FULL_WIDTH)
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_box_aspect(1)

    cxs = [c[0] for c in sensor_coords]
    cys = [c[1] for c in sensor_coords]

    cs_opt = sensor_coords[sensors_opt]
    cxs_opt = [c[0] for c in cs_opt]
    cys_opt = [c[1] for c in cs_opt]

    ax.scatter(cxs, cys, facecolors="none", edgecolors="k")
    for i, (x, y) in enumerate(zip(cxs_opt, cys_opt)):
        ax.scatter(x, y, c="k")
        ax.text(x+75.0, y+75.0, f"{i+1}", fontsize=TEXT_SIZE)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel("$x_{1}$ [m]", fontsize=LABEL_SIZE)
    # ax.set_ylabel("$x_{2}$ [m]", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(fname)

def plot_cbar(cmap, vmin, vmax, label, fname):
    
    plt.figure(figsize=(4, 3))
    plt.pcolormesh(np.ones((2, 2)), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(label=label)

    plt.savefig(fname)

prior = read_prior(f"data/prior.h5")
sensor_coords = read_setup(f"data/setup.h5")
sensors_opt = read_design(f"data/design.h5")

PLOT_STDS_PRI = True
PLOT_PARTICLES_PRI = True
PLOT_DESIGN = True

if PLOT_STDS_PRI:

    fname_plot = "plots/stds_pri.pdf"
    fname_cmap = "plots/cmap_std.pdf"
    vmin, vmax = np.min(prior["stds"]), np.max(prior["stds"])
    
    plot_stds(prior["stds"], fname_plot)
    plot_cbar(CMAP_TRANS, vmin, vmax, LABEL_TRANS, fname_cmap)

if PLOT_PARTICLES_PRI:
    
    fname_plot = "plots/samples_pri.pdf"
    fname_cmap = "plots/cmap_pri.pdf"
    vmin, vmax = 0.0, 5.0
    
    plot_particles(prior["particles"], fname_plot, CMAP_TRANS, vmin, vmax)
    plot_cbar(CMAP_TRANS, vmin, vmax, LABEL_TRANS, fname_cmap)

if PLOT_DESIGN:

    fname = "plots/design.pdf"
    plot_design(sensor_coords, sensors_opt, fname)