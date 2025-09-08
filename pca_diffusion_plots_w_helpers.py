import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.decomposition import PCA
import itertools
import json

import math, tempfile, uuid
from typing import List


from mlp_models_multilayer import DonutMLP
from persistent_homology_gpu import run_ph_for_point_cloud

try:
    from PyPDF2 import PdfMerger
except ImportError as e:
    PdfMerger = None
    _pdf2_err = e

# ---PCA Coordinates

def compute_pca_coords(embedding_weights, num_components=17):
    """
    Given embedding_weights (a NumPy array), compute and return the first num_components principal components.
    The data is centered and PCA is used so that the returned array has shape (n_points, num_components).
    Tries scikit-learn PCA first, falls back to manual SVD if sklearn is not available.
    """
    centered = embedding_weights - np.mean(embedding_weights, axis=0)

    pca = PCA(n_components=num_components, svd_solver="full")
    coords = pca.fit_transform(centered)
    return coords, pca

# ---Diffusion Coordinates

def compute_diffusion_coords(
    embedding_weights: np.ndarray,
    num_coords: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first *num_coords* non-trivial diffusion coordinates
    together with the full descending eigenvalue array (λ0…λn)."""
    # compute pairwise distances and kernel
    dists = squareform(pdist(embedding_weights, metric="euclidean"))
    epsilon = np.median(dists ** 2)
    A = np.exp(-dists ** 2 / epsilon)
    M = A / A.sum(axis=1, keepdims=True)

    # eigendecomposition
    eigenvalues, eigenvectors = eigh(M)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    if eigenvalues.shape[0] < num_coords + 1:
        raise ValueError(
            "Not enough eigenvalues to compute the requested diffusion coordinates."
        )

    # extract non-trivial coords
    coords = eigenvectors[:, 1 : num_coords + 1] * eigenvalues[1 : num_coords + 1]
    return coords, eigenvalues

def make_json(
    freq_list: list[int] | None,
    var_ratio: list[float],
    cum_ratio: list[float],
    save_dir: str
) -> None:
    """
    Dump frequency list and variance ratios as JSON into *save_dir*:
      {
        "freq_list": [...],
        "variance_ratio": [...],
        "cumulative_variance_ratio": [...]
      }
    The file is named "variance_explained.json" and is placed
    directly under *save_dir*.
    """
    os.makedirs(save_dir, exist_ok=True)
    data = {
        "freq_list": freq_list,
        "variance_ratio": var_ratio,
        "cumulative_variance_ratio": cum_ratio,
    }
    out_path = os.path.join(save_dir, "variance_explained.json")
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=4)

# ---------------- Plotting Functions ----------------
def generate_new_diffusion_plot(embedding_weights, output_file, p):
    """
    Compute the first 17 diffusion coordinates and plot 16 scatter plots of coordinate pairs:
    (Coord1 vs Coord2), (Coord2 vs Coord3), ..., (Coord16 vs Coord17) in a 4x4 grid.
    """
    diff_coords, _ = compute_diffusion_coords(embedding_weights, num_coords=17)
    num_plots = 16  # pairs: 1-2, 2-3, ..., 16-17
    fig = make_subplots(rows=4, cols=4,
                        subplot_titles=[f"Coord {i+1} vs {i+2}" for i in range(num_plots)])
    
    labels = np.arange(diff_coords.shape[0]) % p
    
    marker_args = dict(
        color=labels,
        colorscale=[(0.0, 'blue'), (0.5, 'red'), (1.0, 'blue')],
        cmin=0,
        cmax=p-1,
        size=6
    )
    
    plot_idx = 0
    for i in range(4):
        for j in range(4):
            x_coord = diff_coords[:, plot_idx]
            y_coord = diff_coords[:, plot_idx + 1]
            fig.add_trace(
                go.Scatter(x=x_coord, y=y_coord,
                           mode='markers', marker=marker_args),
                row=i+1, col=j+1
            )
            plot_idx += 1

    fig.update_layout(height=1000, width=1000,
                      title_text="New Diffusion Plot (16 coordinate pair plots)",
                      showlegend=False)
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"New diffusion plot saved to {output_file}")

def create_2d_diffusion_figure(embedding_weights, color_values, title_text, p):
    """
    Compute the first 17 diffusion coordinates and create a 4x4 grid of 16 scatter plots
    (Coord1 vs Coord2, Coord2 vs Coord3, ..., Coord16 vs Coord17). Each point shows custom
    hover text "a={a}, b={b}, y={y}" computed from its index.
    The marker colors are set by the provided color_values.
    """
    diff_coords, _ = compute_diffusion_coords(embedding_weights, num_coords=17)
    num_plots = 16
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=[f"Coord {i+1} vs {i+2}" for i in range(num_plots)]
    )
    
    n_points = diff_coords.shape[0]
    indices = np.arange(n_points)
    a_vals = indices // p
    b_vals = indices % p
    y_vals = (a_vals + b_vals) % p
    hover_texts = [f"a={a}, b={b}, y={y}" for a, b, y in zip(a_vals, b_vals, y_vals)]
    
    marker_args = dict(
        color=color_values,
        colorscale=[(0.0, 'blue'), (1.0, 'red')],
        cmin=0,
        cmax=p-1,
        size=6
    )
    
    plot_idx = 0
    for i in range(4):
        for j in range(4):
            x_coord = diff_coords[:, plot_idx]
            y_coord = diff_coords[:, plot_idx + 1]
            trace = go.Scatter(
                x=x_coord,
                y=y_coord,
                mode='markers',
                marker=marker_args,
                hovertext=hover_texts,
                hovertemplate='%{hovertext}<extra></extra>'
            )
            fig.add_trace(trace, row=i+1, col=j+1)
            plot_idx += 1

    fig.update_layout(
        height=1000,
        width=1000,
        title_text=title_text,
        showlegend=False
    )
    return fig

def create_3d_diffusion_figure(embedding_weights, color_values, title_text, p):
    """
    Compute the first 17 diffusion coordinates and create a 3x5 grid (15 subplots) of 3D scatter plots.
    Each subplot plots three consecutive coordinates:
      (Coord1, Coord2, Coord3), (Coord2, Coord3, Coord4), ..., (Coord15, Coord16, Coord17).
    """
    diff_coords, _ = compute_diffusion_coords(embedding_weights, num_coords=17)
    num_plots = 15
    rows, cols = 3, 5
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Coords {i+1}-{i+3}" for i in range(num_plots)],
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    n_points = diff_coords.shape[0]
    indices = np.arange(n_points)
    a_vals = indices // p
    b_vals = indices % p
    y_vals = (a_vals + b_vals) % p
    hover_texts = [f"a={a}, b={b}, y={y}" for a, b, y in zip(a_vals, b_vals, y_vals)]
    
    marker_args = dict(
        size=4,
        color=color_values,
        colorscale=[(0.0, 'blue'), (1.0, 'red')],
        cmin=0,
        cmax=p-1,
    )
    
    plot_idx = 0
    for i in range(rows):
        for j in range(cols):
            if plot_idx < num_plots:
                x_data = diff_coords[:, plot_idx]
                y_data = diff_coords[:, plot_idx + 1]
                z_data = diff_coords[:, plot_idx + 2]
                trace = go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=marker_args,
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>'
                )
                fig.add_trace(trace, row=i+1, col=j+1)
                scene_id = f'scene{(i * cols + j + 1) if (i * cols + j + 1) > 1 else ""}'
                fig.layout[scene_id].xaxis.title = f"diff coord {plot_idx + 1}"
                fig.layout[scene_id].yaxis.title = f"diff coord {plot_idx + 2}"
                fig.layout[scene_id].zaxis.title = f"diff coord {plot_idx + 3}"
                plot_idx += 1

    fig.update_layout(
        height=1200,
        width=1800,
        title_text=title_text,
        showlegend=False
    )
    return fig


def create_3d_pca_figure(embedding_weights, color_values, title_text, p):
    """
    Compute the principal components and create a grid of 3D scatter plots.
    Each subplot plots three consecutive components:
      (PC1, PC2, PC3), (PC2, PC3, PC4), ..., (PC_(n-2), PC_(n-1), PC_n).
    """
    pca_coords = compute_pca_coords(embedding_weights, num_components=17)
    available_components = pca_coords.shape[1]
    
    if available_components < 3:
        raise ValueError("Not enough PCA components to create a 3D plot.")

    num_plots = available_components - 2
    cols = min(5, num_plots)
    rows = (num_plots + cols - 1) // cols

    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"PCs {i+1}-{i+3}" for i in range(num_plots)],
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    n_points = pca_coords.shape[0]
    indices = np.arange(n_points)
    a_vals = indices // p
    b_vals = indices % p
    y_vals = (a_vals + b_vals) % p
    hover_texts = [f"a={a}, b={b}, y={y}" for a, b, y in zip(a_vals, b_vals, y_vals)]
    
    marker_args = dict(
        size=4,
        color=color_values,
        colorscale=[(0.0, 'blue'), (1.0, 'red')],
        cmin=0,
        cmax=p-1,
    )
    
    plot_idx = 0
    for i in range(rows):
        for j in range(cols):
            if plot_idx < num_plots:
                x_data = pca_coords[:, plot_idx]
                y_data = pca_coords[:, plot_idx + 1]
                z_data = pca_coords[:, plot_idx + 2]
                trace = go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=marker_args,
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>'
                )
                fig.add_trace(trace, row=i+1, col=j+1)
                scene_id = f'scene{(i * cols + j + 1) if (i * cols + j + 1) > 1 else ""}'
                fig.layout[scene_id].xaxis.title = f"PCA coord {plot_idx + 1}"
                fig.layout[scene_id].yaxis.title = f"PCA coord {plot_idx + 2}"
                fig.layout[scene_id].zaxis.title = f"PCA coord {plot_idx + 3}"
                plot_idx += 1

    fig.update_layout(
        height=1200,
        width=1800,
        title_text=title_text,
        showlegend=False
    )
    return fig

# ---Interactive embedding plot code

def generate_diffusion_map_figure(embedding_weights, epoch, p, f_multiplier=1, diffusion_coords=None):
    """
    Given embedding_weights and an epoch number, return a Plotly figure showing the diffusion map.
    If diffusion_coords is provided, use it instead of recomputing.
    """
    if diffusion_coords is None:
        diffusion_coords, _ = compute_diffusion_coords(embedding_weights)
    
    num_points = diffusion_coords.shape[0]
    if num_points == p:
        indices = np.arange(num_points)
        labels = (f_multiplier * indices) % p
    elif num_points == p*p:
        indices = np.arange(num_points)
        a = indices // p
        b = indices % p
        labels = (a + b) % p
    else:
        labels = np.zeros(num_points)
    
    custom_colorscale = [(0.0, 'blue'), (0.5, 'red'), (1.0, 'blue')]
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Coordinate 1 vs 2", "Coordinate 2 vs 3",
                                        "Coordinate 3 vs 4", "Coordinate 4 vs 5"))
    
    marker_args = dict(
        color=labels,
        colorscale=custom_colorscale,
        cmin=0,
        cmax=p-1,
        size=8,
        colorbar=dict(title="(f * index) mod p")
    )
    
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 0], y=diffusion_coords[:, 1],
                   mode='markers', marker=marker_args),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 1], y=diffusion_coords[:, 2],
                   mode='markers', marker=marker_args),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 2], y=diffusion_coords[:, 3],
                   mode='markers', marker=marker_args),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=diffusion_coords[:, 3], y=diffusion_coords[:, 4],
                   mode='markers', marker=marker_args),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Diffusion Coordinate 1", row=1, col=1)
    fig.update_yaxes(title_text="Diffusion Coordinate 2", row=1, col=1)
    
    fig.update_xaxes(title_text="Diffusion Coordinate 2", row=1, col=2)
    fig.update_yaxes(title_text="Diffusion Coordinate 3", row=1, col=2)
    
    fig.update_xaxes(title_text="Diffusion Coordinate 3", row=2, col=1)
    fig.update_yaxes(title_text="Diffusion Coordinate 4", row=2, col=1)
    
    fig.update_xaxes(title_text="Diffusion Coordinate 4", row=2, col=2)
    fig.update_yaxes(title_text="Diffusion Coordinate 5", row=2, col=2)
    
    fig.update_layout(height=800, width=800,
                      title_text=f"Diffusion Map (Epoch {epoch}, f_multiplier={f_multiplier})",
                      showlegend=False)
    
    return fig

def generate_interactive_diffusion_map_html(epoch_embedding_log, output_file, p, f_multiplier=1):
    """
    Given a dictionary keyed by epoch with embedding weight matrices,
    create an interactive Plotly figure using precomputed diffusion coordinates per epoch.
    """
    sorted_epochs = sorted(epoch_embedding_log.keys())
    frames = []
    for idx, epoch in enumerate(sorted_epochs):
        emb_weights = np.array(epoch_embedding_log[epoch])
        diff_coords, _ = compute_diffusion_coords(emb_weights)
        fig_epoch = generate_diffusion_map_figure(emb_weights, epoch, p, f_multiplier=f_multiplier, diffusion_coords=diff_coords)
        frame = go.Frame(data=fig_epoch.data, name=str(epoch))
        frames.append(frame)
        print(f"Made diffusion plot for epoch {epoch} (f_multiplier={f_multiplier}).")
    
    base_epoch = sorted_epochs[0]
    base_emb_weights = np.array(epoch_embedding_log[base_epoch])
    base_diff_coords, _ = compute_diffusion_coords(base_emb_weights)
    base_fig = generate_diffusion_map_figure(base_emb_weights, base_epoch, p, f_multiplier=f_multiplier, diffusion_coords=base_diff_coords)
    
    slider_steps = []
    for epoch in sorted_epochs:
        step = dict(
            label=str(epoch),
            method="animate",
            args=[[str(epoch)],
                  {"mode": "immediate",
                   "frame": {"duration": 300, "redraw": True},
                   "transition": {"duration": 200}}]
        )
        slider_steps.append(step)
    
    base_fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1,
            x=1.1,
            xanchor="right",
            yanchor="top",
            pad={"t": 0, "r": 10},
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {
                    "frame": {"duration": 300, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 200}
                }]
            )]
        )],
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            steps=slider_steps
        )]
    )
    
    base_fig.frames = frames
    base_fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Interactive diffusion map saved to {output_file}")

# Start of make .pdf code
def _write_multiplot_2d(coords: np.ndarray,
                        colour: np.ndarray,
                        ctitle: str,
                        out_path: str,
                        p: int,
                        seed,
                        label: str,
                        tag: str) -> None:
    """
    Generate 2-D scatter plots for *all* coordinate pairs.
    Each PDF “page” contains 8 × 4 = 32 sub-plots; pages are merged
    into a single multi-page PDF written to ``out_path``.

    Parameters
    ----------
    coords : (n_points, n_dims) array
        Point coordinates.
    colour : (n_points,) array
        Per-point colour values.
    ctitle : str
        Title for the colour-bar.
    out_path : str
        Destination PDF (multi-page).
    p : int
        Modulus for colour bar scaling.
    seed, label, tag
        Passed straight through to titles / figure text.
    """
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be 2-D with at least two columns.")
    if PdfMerger is None:
        raise ImportError(
            "PyPDF2 is required for PDF concatenation but could not be imported."
        ) from _pdf2_err

    # ── generate (i,j) pairs ────────────────────────────────────────────
    pairs = list(itertools.combinations(range(coords.shape[1]), 2))
    per_page = 32
    n_pages = math.ceil(len(pairs) / per_page)

    tmp_files: List[str] = []

    for page in range(n_pages):
        page_pairs = pairs[page * per_page:(page + 1) * per_page]
        n_cols, n_rows = 4, max(1, math.ceil(len(page_pairs) / 4))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{label}{i} vs {label}{j}" for i, j in page_pairs],
            horizontal_spacing=0.04,
            vertical_spacing=0.06,
        )

        for k, (i, j) in enumerate(page_pairs, 1):
            r, c = 1 + (k - 1) // n_cols, 1 + (k - 1) % n_cols
            fig.add_trace(
                go.Scatter(
                    x=coords[:, i],
                    y=coords[:, j],
                    mode="markers",
                    name="",
                    showlegend=False,
                    marker=dict(
                        size=4,
                        color=colour,
                        colorscale="Viridis",
                        cmin=0,
                        cmax=p - 1,
                        line=dict(width=0),
                        showscale=(k == 1),  # colour-bar only once per page
                        colorbar=dict(
                            title=ctitle,
                            tickvals=list(range(0, p, max(1, p // 10))),
                            ticktext=[str(v) for v in range(0, p, max(1, p // 10))],
                        ),
                    ),
                ),
                row=r,
                col=c,
            )
            fig.update_xaxes(title_text=f"{label}{i}", row=r, col=c)
            fig.update_yaxes(title_text=f"{label}{j}", row=r, col=c)

        fig.update_layout(
            width=1400,
            height=250 * n_rows + 100,  # dynamic vertical size
            title=f"{label} 2-D – seed {seed} – page {page + 1}/{n_pages} – {tag}",
            margin=dict(l=40, r=40, t=80, b=40),
        )

        # Write page to a temp file
        tmp_pdf = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.pdf")
        fig.write_image(tmp_pdf, format="pdf")
        tmp_files.append(tmp_pdf)

    # ── merge pages into one multi-page PDF ─────────────────────────────
    merger = PdfMerger()
    for pdf in tmp_files:
        merger.append(pdf)
    merger.write(out_path)
    merger.close()

    # clean-up
    for pdf in tmp_files:
        try:
            os.remove(pdf)
        except OSError:
            pass

    print(f"[{label} 2-D]  →  {out_path}")

def _write_multiplot_3d(coords: np.ndarray,
                        colour: np.ndarray,
                        ctitle: str,
                        out_path: str,
                        p: int,
                        seed,
                        label: str,
                        tag: str):
    """
    Four-panel 3-D scatter (all 4 choose 3 combos) → *PDF*.
    """
    triplets = list(itertools.combinations(range(4), 3))
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}]*2]*2,
        subplot_titles=[f"{label}{i} vs {label}{j} vs {label}{k}"
                        for i, j, k in triplets],
        horizontal_spacing=0.03, vertical_spacing=0.03
    )

    for idx, (i, j, k) in enumerate(triplets, 1):
        row, col = (1, idx) if idx <= 2 else (2, idx-2)
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, i], y=coords[:, j], z=coords[:, k],
                mode='markers',
                name='',  
                showlegend=False, 
                marker=dict(
                    size=3,
                    color=colour,
                    colorscale='Viridis',
                    cmin=0,
                    cmax=p-1,
                    showscale=(idx == 1),
                    colorbar=dict(
                        title=ctitle,
                        tickvals=list(range(0, p, max(1, p//10))),
                        ticktext=[str(v) for v in range(0, p, max(1, p//10))],
                    )
                )
            ),
            row=row, col=col
        )
        scene_id = f"scene{idx if idx > 1 else ''}"
        fig.layout[scene_id].xaxis.title.text = f"{label}{i}"
        fig.layout[scene_id].yaxis.title.text = f"{label}{j}"
        fig.layout[scene_id].zaxis.title.text = f"{label}{k}"

    fig.update_layout(
        width=1000, height=900,
        title=f"{label} 3-D (first 4) – seed {seed} - {tag}",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.write_image(out_path, format="pdf")
    print(f"[{label} 3-D]  →  {out_path}")


def save_homology_artifacts(coords: np.ndarray,
                            root_dir: str,
                            tag: str,
                            seed,
                            label: str,
                            num_dims: int | None = 2) -> None:
    """
    Thin wrapper that decides the sub-folder and filename stem,
    then calls `run_ph_for_point_cloud` on the first `num_dims` coords (if set).

    Parameters
    ----------
    num_dims : int | None
        Number of leading dimensions to keep for PH; if None, uses all dims.
    """
    # decide sub-folder and filename stem
    subdir = os.path.join(root_dir, "homology", tag)
    stem = f"{label.lower()}_seed_{seed}"

    # select only the first num_dims dimensions (if requested)
    n_nbrs = 150
    if num_dims is not None:
        if num_dims < 1 or num_dims > coords.shape[1]:
            raise ValueError(f"num_dims must be between 1 and {coords.shape[1]}")
        coords_to_use = coords[:, :num_dims]
    else:
        coords_to_use = coords

    if num_dims == 2:
        n_nbrs = 300
    
    run_ph_for_point_cloud(
        coords_to_use,
        maxdim=2,
        ph_sparse=True,
        n_nbrs=n_nbrs,
        save_dir=subdir,
        filename_stem=stem,
        title=f"{label}  (seed={seed})"
    )

#  Helper: phase-scatter & vector plots for the equal-frequency case
def _make_single_freq_phase_plots(mat: np.ndarray,
                                  p: int,
                                  f: int,
                                  save_dir: str,
                                  *,
                                  seed: int | str = "",
                                  tag: str = "",
                                  colour_scale: str = "Viridis",
                                  eps: float = 0.07) -> None:
    """
    Build a 2×2 PDF figure with
       (1,1) raw (φ_a, φ_b) scatter               coloured by amplitude
       (1,2) raw vectors from (0,0) to each point
       (2,1) merged-point scatter  (fat circles, label = Σ amps)
       (2,2) merged vectors (fat endpoints)

    A torus distance ≤ *eps* merges points; merged coordinates are
    the amplitude-weighted circular mean, amplitude = Σ amplitudes.
    """
    f = int(f) % p
    if f == 0:
        print("[phase-plots] f ≡ 0 (mod p) – skipped.")
        return

    # -- 1.  FFT → amplitudes & phases ----------------------------------
    n_neurons = mat.shape[1]
    amps  = np.empty(n_neurons)
    phi_a = np.empty(n_neurons)
    phi_b = np.empty(n_neurons)

    for n in range(n_neurons):
        grid   = mat[:, n].reshape(p, p).T
        F      = np.fft.fft2(grid) / (p * p)
        ca, cb = F[f, 0], F[0, f]

        amps[n]  = np.hypot(2*np.abs(ca), 2*np.abs(cb))
        phi_a[n] = (-np.angle(ca)) % (2*np.pi)
        phi_b[n] = (-np.angle(cb)) % (2*np.pi)

    # -- 2.  cluster / merge close points on the torus ------------------
    unpicked   = set(range(n_neurons))
    m_phi_a, m_phi_b, m_amp = [], [], []

    def torus_dist(x1, y1, x2, y2):
        dx = np.abs(x1 - x2); dx = np.minimum(dx, 2*np.pi - dx)
        dy = np.abs(y1 - y2); dy = np.minimum(dy, 2*np.pi - dy)
        return np.sqrt(dx*dx + dy*dy)

    while unpicked:
        i      = unpicked.pop()
        group  = [i]
        for j in list(unpicked):
            if torus_dist(phi_a[i], phi_b[i], phi_a[j], phi_b[j]) <= eps:
                unpicked.remove(j)
                group.append(j)

        # amplitude-weighted *circular* mean ----------------------------
        A      = amps[group]
        w_sum  = A.sum()
        ang_ax = np.arctan2((A*np.sin(phi_a[group])).sum(),
                            (A*np.cos(phi_a[group])).sum()) % (2*np.pi)
        ang_bx = np.arctan2((A*np.sin(phi_b[group])).sum(),
                            (A*np.cos(phi_b[group])).sum()) % (2*np.pi)

        m_phi_a.append(ang_ax)
        m_phi_b.append(ang_bx)
        m_amp .append(w_sum)

    m_phi_a = np.asarray(m_phi_a)
    m_phi_b = np.asarray(m_phi_b)
    m_amp   = np.asarray(m_amp)

    # -- 3.  build 2×2 Plotly figure -----------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "raw scatter", "raw vectors",
            "merged scatter", "merged vectors"
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )

    # row-1-col-1 : raw scatter
    fig.add_trace(
        go.Scatter(
            x=phi_a, y=phi_b, mode="markers",
            marker=dict(size=6, color=amps,
                        colorscale=colour_scale,
                        colorbar=dict(title="amplitude")),
            hovertemplate="φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=1, col=1
    )

    # row-1-col-2 : raw vectors
    for pa, pb in zip(phi_a, phi_b):
        fig.add_trace(
            go.Scatter(x=[0, pa], y=[0, pb],
                       mode="lines",
                       line=dict(width=1.5, color="rgba(0,0,0,0.5)"),
                       hoverinfo="skip"), row=1, col=2
        )
    fig.add_trace(
        go.Scatter(
            x=phi_a, y=phi_b, mode="markers",
            marker=dict(size=6, color=amps,
                        colorscale=colour_scale, showscale=False),
            hovertemplate="φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=1, col=2
    )

    # row-2-col-1 : merged scatter  (fat, annotated)
    fig.add_trace(
        go.Scatter(
            x=m_phi_a, y=m_phi_b,
            mode="markers+text",
            marker=dict(size=12, color=m_amp,
                        colorscale=colour_scale, showscale=False,
                        line=dict(width=1, color="black")),
            text=[f"{a:.1f}" for a in m_amp],
            textposition="top center",
            hovertemplate="[merged]<br>φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=2, col=1
    )

    # row-2-col-2 : merged vectors
    for pa, pb in zip(m_phi_a, m_phi_b):
        fig.add_trace(
            go.Scatter(
                x=[0, pa], y=[0, pb],
                mode="lines",
                line=dict(width=2, color="rgba(0,0,0,0.6)"),
                hoverinfo="skip"
            ), row=2, col=2
        )
    fig.add_trace(
        go.Scatter(
            x=m_phi_a, y=m_phi_b,
            mode="markers+text",
            marker=dict(size=12, color=m_amp,
                        colorscale=colour_scale, showscale=False,
                        line=dict(width=1, color="black")),
            text=[f"{a:.1f}" for a in m_amp],
            textposition="top center",
            hovertemplate="[merged]<br>φₐ=%{x:.2f}<br>φ_b=%{y:.2f}<br>|A|=%{marker.color:.3f}"
            "<extra></extra>"
        ), row=2, col=2
    )

    # common axes labels
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="φₐ (rad)", row=r, col=c)
            fig.update_yaxes(title_text="φ_b (rad)", row=r, col=c)

    fig.update_layout(
        width=1100, height=900,
        title=f"Seed {seed} – f = {f} – {tag}",
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=False
    )

    # -- 4.  save PDF ---------------------------------------------------
    phase_dir = os.path.join(save_dir, "phase_plots")
    os.makedirs(phase_dir, exist_ok=True)
    fname_pdf = f"seed_{seed}_f{f}{'_'+tag if tag else ''}.pdf"
    out_path  = os.path.join(phase_dir, fname_pdf)
    fig.write_image(out_path, format="pdf")   # requires Ka leido
    print(f"[phase-plots] wrote {out_path}")


def _generate_pdf_plots_for_matrix(mat: np.ndarray,
                                  p: int,
                                  save_dir: str,
                                  *,
                                  seed: int | str = "",
                                  freq_list: list[int] | None = None,
                                  tag: str = "",
                                  class_string: str = "",
                                  num_principal_components=2) -> None:
    """
    Create the same PCA & diffusion-map PDF plots as the research notebook,
    **without** any persistent-homology calls.

    Parameters
    ----------
    mat : np.ndarray
        Data matrix whose rows are the points to embed.
    p : int
        Alphabet size for colour-coding.
    save_dir : str
        Root folder into which PDFs are written.
    seed : int | str, optional
        Identifying seed for titles / file names.
    freq_list : list[int] | None, optional
        Frequency multipliers for the extra colourings
        (pass whatever you used in `final_grouping`).
    tag : str, optional
        Extra string inserted in file names (e.g. "embeds").
    """
    n_samples, n_features = mat.shape
    num_components = min(n_samples, n_features, 8)

    if num_components >= 4:
        append_to_title = f"{tag} & {class_string}"
        freq_list = sorted(freq_list or [])
        os.makedirs(save_dir, exist_ok=True)
        

        # ─── Colour bases (same as example) ───────────────────────────────────
        n_points = p**2
        indices = np.arange(n_points)
        a_vals = indices // p
        b_vals = indices % p
        coords_ab = np.stack([a_vals, b_vals], axis=1)
        colour_base = (coords_ab[:, 0] + coords_ab[:, 1]) % p

        # ─── Directory tree ──────────────────────────────────────────────────
        pca_root = os.path.join(save_dir, "pca_pdf_plots")
        dif_root = os.path.join(save_dir, "diffusion_pdf_plots")
        for root in (pca_root, dif_root):
            for sub in ("2d", "3d"):
                os.makedirs(os.path.join(root, sub, tag), exist_ok=True)
        print("computing PCA")
        # ─── PCA (4 comps) ───────────────────────────────────────────────────
        pcs, pca = compute_pca_coords(mat, num_components=num_components)

        base_2d_dir = os.path.join(pca_root, "2d", tag)
        # base_2d = os.path.join(
        #     base_2d_dir,
        #     f"pca_seed_{seed}{('_'+tag) if tag else ''}_base.pdf"
        # )
        # _write_multiplot_2d(pcs, colour_base, "(a+b) mod p",
        #                     base_2d, p, seed, "PC", append_to_title)
        var_ratio = pca.explained_variance_ratio_.tolist()
        cum_ratio = np.cumsum(pca.explained_variance_ratio_).tolist()
        make_json(freq_list, var_ratio, cum_ratio, base_2d_dir)
        
        base_3d_dir = os.path.join(pca_root, "3d", tag)
        # base_3d = os.path.join(
        #     base_3d_dir,
        #     f"pca_seed_{seed}{('_'+tag) if tag else ''}_base_3d.pdf"
        # )
        # _write_multiplot_3d(pcs, colour_base, "(a+b) mod p",
        #                     base_3d, p, seed, "PC", append_to_title)
        make_json(freq_list, var_ratio, cum_ratio, base_3d_dir)
        
        # Extra plots per-frequency
        for f in freq_list:
            if f % p == 0:
                continue
            colour_f = (f * (coords_ab[:, 0] + coords_ab[:, 1])) % p
            name_stub = f"pca_seed_{seed}_freq_{f}.pdf"
            _write_multiplot_2d(
                pcs,
                colour_f,
                f"{f}·(a+b) mod p",
                os.path.join(pca_root, "2d", tag, name_stub.replace(".pdf", "_2d.pdf")),
                p, seed, "PC", append_to_title
            )
            _write_multiplot_3d(
                pcs,
                colour_f,
                f"{f}·(a+b) mod p",
                os.path.join(pca_root, "3d", tag, name_stub.replace(".pdf", "_3d.pdf")),
                p, seed, "PC", append_to_title
            )

        save_homology_artifacts(
            pcs,
            root_dir=pca_root,
            tag=tag,
            seed=seed,
            label=f"PCA--{class_string}",
            num_dims=num_principal_components)

        # ─── Diffusion (first 4) ─────────────────────────────────────────────
        dmap, eigenvalues = compute_diffusion_coords(mat, num_coords=num_components)

        base_2d_d_dir = os.path.join(dif_root, "2d", tag)
        # base_2d_d = os.path.join(base_2d_d_dir,
        #                          f"diff_seed_{seed}{('_'+tag) if tag else ''}_base.pdf")
        
        # _write_multiplot_2d(dmap, colour_base, "(a+b) mod p",
        #                     base_2d_d, p, seed, "DM", append_to_title)
        make_json(freq_list, var_ratio, cum_ratio, base_2d_d_dir)
        
        nontriv = np.abs(eigenvalues[1:17])  # 1 to 16 (nontrivial)
        total = nontriv.sum()
        if total > 0:
            var_ratio = (nontriv / total).tolist()
            cum_ratio = np.cumsum(nontriv / total).tolist()
        else:
            var_ratio = [0.0] * 16
            cum_ratio = [0.0] * 16

        base_3d_d_dir = os.path.join(dif_root, "3d", tag)
        # base_3d_d = os.path.join(base_3d_d_dir,
        #                             f"diff_seed_{seed}{('_'+tag) if tag else ''}_base_3d.pdf")
        
        # _write_multiplot_3d(dmap, colour_base, "(a+b) mod p",
        #                     base_3d_d, p, seed, "DM", append_to_title)
        make_json(freq_list, var_ratio, cum_ratio, base_3d_d_dir)
        for f in freq_list:
            if f % p == 0:
                continue
            colour_f = (f * (coords_ab[:, 0] + coords_ab[:, 1])) % p
            name_stub = f"diff_seed_{seed}_freq_{f}.pdf"
            _write_multiplot_2d(
                dmap,
                colour_f,
                f"{f}·(a+b) mod p",
                os.path.join(dif_root, "2d", tag, name_stub.replace(".pdf", "_2d.pdf")),
                p, seed, "DM", append_to_title
            )
            _write_multiplot_3d(
                dmap,
                colour_f,
                f"{f}·(a+b) mod p",
                os.path.join(dif_root, "3d", tag, name_stub.replace(".pdf", "_3d.pdf")),
                p, seed, "DM", append_to_title
            )
        
        save_homology_artifacts(
            dmap,
            root_dir=dif_root,
            tag=tag,
            seed=seed,
            label=f"Dif--{class_string}",
            num_dims=num_principal_components)

        print("✔️  All PCA / diffusion PDF plots written.")

        if len(freq_list) == 1 and (mat.shape[0] == p ** 2):
            # make a 2d scatterplot lattice of the phases and a vector plot of them
            _make_single_freq_phase_plots(mat, p, freq_list[0], save_dir,
                                    seed=seed, tag=tag)
        
def generate_pdf_plots_for_matrix(
        mat: np.ndarray,
        p: int,
        save_dir: str,
        *,
        seed: int | str = "",
        freq_list: list[int] | None = None,
        tag: str = "",
        class_string: str = "",
        num_principal_components: int = 2,
        do_transposed: bool = False
) -> None:
    """
    Run all PCA / diffusion / homology plots for `mat`, and—if
    `do_transposed` is True—repeat on the transposed matrix.

    • The second run writes into exactly the same directory tree but with
      “_transposed” appended to every sub-folder via the `tag` argument.
    """
    # ---- first pass: original matrix ------------------------------------
    _generate_pdf_plots_for_matrix(
        mat, p, save_dir,
        seed=seed,
        freq_list=freq_list,
        tag=tag,
        class_string=class_string,
        num_principal_components=num_principal_components
    )

    # ---- optional second pass: transposed matrix ------------------------
    if do_transposed:
        new_tag = f"{tag}_transposed" if tag else "transposed"
        _generate_pdf_plots_for_matrix(
            mat.T, p, save_dir,
            seed=seed,
            freq_list=freq_list,
            tag=new_tag,
            class_string=class_string,
            num_principal_components=num_principal_components
        )


def generate_pca_information_scaling_experiment(mat: np.ndarray,
                                                p: int,
                                                save_dir: str,
                                                *,
                                                seed: int | str = "",
                                                freq_list: list[int] | None = None,
                                                tag: str = "") -> None:
    """
    For a given data matrix mat, compute:
      - cumulative PCA variance ratios for components 1–4
      - cumulative diffusion 'variance' ratios (via eigenvalues) for coords 1–4
    and save them as JSON.

    Parameters
    ----------
    mat : np.ndarray
        Data matrix whose rows are the points to embed.
    p : int
        Alphabet size (only recorded in JSON for provenance).
    save_dir : str
        Directory into which the JSON file will be written.
    seed : int | str, optional
        Identifier for this run, used in the filename and in the JSON.
    freq_list : list[int] | None, optional
        Ignored here (present only to mirror generate_pdf_plots_for_matrix).
    tag : str, optional
        Extra string inserted in the filename (e.g. "embeds").
    """
    # ensure output dir exists
    os.makedirs(save_dir, exist_ok=True)

    # --- PCA part (up to 4 components) ---
    centered = mat - np.mean(mat, axis=0)
    n_comp = min(4, centered.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(centered)
    var_ratio = pca.explained_variance_ratio_          # length = n_comp
    cum_var_ratio = np.cumsum(var_ratio).tolist()      # [v1, v1+v2, …]

    # --- Diffusion part (up to 4 nontrivial eigenvalues) ---
    dists = squareform(pdist(mat, metric="euclidean"))
    ε = np.median(dists**2)
    A = np.exp(-dists**2 / ε)
    M = A / A.sum(axis=1, keepdims=True)

    eigvals, _ = eigh(M)
    eigvals = eigvals[::-1]          # descending
    nontrivial = eigvals[1:1 + n_comp]    # skip the trivial λ₀=1
    total = np.sum(nontrivial)
    if total > 0:
        diff_ratios = (nontrivial / total)
    else:
        diff_ratios = np.zeros_like(nontrivial)
    cum_diff_ratio = np.cumsum(diff_ratios).tolist()

    # --- Assemble JSON ---
    info = {
        "seed": seed,
        "p": p,
        "num_pca_components": n_comp,
        "cumulative_pca_variance_ratio": cum_var_ratio,
        "num_diffusion_components": len(nontrivial),
        "cumulative_diffusion_eigenvalue_ratio": cum_diff_ratio,
    }

    # build filename
    fname = f"pca_info_seed_{seed}"
    if tag:
        fname += f"_{tag}"
    fname += ".json"
    out_path = os.path.join(save_dir, fname)

    with open(out_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"✔️  PCA & diffusion scaling info saved to {out_path}")