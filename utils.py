import jax
def compute_pytree_size(pytree):
    total_size = 0
    for array in jax.tree_util.tree_leaves(pytree):
        total_size += array.size * array.dtype.itemsize
    in_mb = total_size / (1024 ** 2) 
    print(f"model size in megabytes: {in_mb:.3f}")


# def put_neuron_preaccs_on_torus(
#         cluster_groupings,
#         neuron_data,
#         p: int,
#         base_dir: str,
#         layer_idx: int = 1,
#         major_radius: float = 3.0,
#         minor_radius: float = 1.0,
#         overlay_cutoff: float = 0.80,
#         mesh_res: int = 70,
#         max_neurons_per_cluster: int = 20,
#     ):
#         """
#         As before, but now also writes cluster_{freq}.json alongside each HTML.
#         """
#         out_root = os.path.join(base_dir, "cluster_preaccs_on_tori")
#         os.makedirs(out_root, exist_ok=True)

#         # Grab the right layer
#         try:
#             clusters   = cluster_groupings[layer_idx - 1]
#             layer_dict = neuron_data[layer_idx]
#         except (IndexError, KeyError):
#             print(f"[put_neuron_preaccs_on_torus] layer {layer_idx} unavailable.")
#             return

#         two_pi_over_p = 2.0 * np.pi / p

#         # Precompute torus surface once
#         th = np.linspace(0, 2*np.pi, mesh_res)
#         ph = np.linspace(0, 2*np.pi, mesh_res)
#         TH, PH = np.meshgrid(th, ph, indexing="ij")
#         Xsurf = (major_radius + minor_radius*np.cos(PH)) * np.cos(TH)
#         Ysurf = (major_radius + minor_radius*np.cos(PH)) * np.sin(TH)
#         Zsurf =  minor_radius * np.sin(PH)
#         TORUS_GHOST = go.Surface(
#             x=Xsurf, y=Ysurf, z=Zsurf,
#             colorscale="Greys", opacity=0.25, showscale=False, hoverinfo="skip"
#         )

#         px_cycle = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A",
#                     "#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"]

#         # Helper to build both the Plotly trace and the raw (a,b) arrays
#         def curve_ab(freq, kind):
#             a_vals = np.arange(p)
#             if kind == "sum0":
#                 b_vals = (-a_vals) % p
#                 color, name = "black", "a+b≡0"
#             else:
#                 b_vals = a_vals
#                 color, name = "red", "a=b"

#             ang_a = two_pi_over_p * freq * a_vals
#             ang_b = two_pi_over_p * freq * b_vals
#             X = (major_radius + minor_radius*np.cos(ang_b)) * np.cos(ang_a)
#             Y = (major_radius + minor_radius*np.cos(ang_b)) * np.sin(ang_a)
#             Z =  minor_radius * np.sin(ang_b)
#             # close loop
#             X = np.append(X, X[0]); Y = np.append(Y, Y[0]); Z = np.append(Z, Z[0])

#             trace = go.Scatter3d(
#                 x=X, y=Y, z=Z,
#                 mode="lines",
#                 line=dict(color=color, width=4),
#                 name=name,
#                 showlegend=True
#             )
#             return {"a_vals": a_vals.tolist(),
#                     "b_vals": b_vals.tolist(),
#                     "trace": trace}

#         # Iterate clusters exactly as before
#         for freq, neuron_ids in clusters.items():
#             if not neuron_ids:
#                 continue

#             # (A) build the cluster‑wide torus + neuron markers
#             cluster_traces = [
#                 TORUS_GHOST,
#                 curve_ab(freq, "sum0")["trace"],
#                 curve_ab(freq, "diag")["trace"]
#             ]

#             # Collect JSON data
#             overlay_points   = {}
#             neuron_plot_data = {}

#             # Add each neuron's cutoff points
#             for n_id in neuron_ids:
#                 real = np.asarray(layer_dict[n_id]["real_preactivations"])
#                 thr = overlay_cutoff * real.max()
#                 a_sel, b_sel = np.where(real >= thr)
#                 if a_sel.size == 0:
#                     continue

#                 overlay_points[n_id] = {
#                     "a_sel": a_sel.tolist(),
#                     "b_sel": b_sel.tolist()
#                 }

#                 # also add the Plotly scatter
#                 ang_a = two_pi_over_p * freq * a_sel
#                 ang_b = two_pi_over_p * freq * b_sel
#                 X = (major_radius + minor_radius*np.cos(ang_b)) * np.cos(ang_a)
#                 Y = (major_radius + minor_radius*np.cos(ang_b)) * np.sin(ang_a)
#                 Z =  minor_radius * np.sin(ang_b)
#                 cluster_traces.append(go.Scatter3d(
#                     x=X, y=Y, z=Z,
#                     mode="markers",
#                     marker=dict(size=4,
#                                 color=px_cycle[len(cluster_traces) % len(px_cycle)],
#                                 opacity=0.9),
#                     name=f"neuron {n_id}"
#                 ))

#             # (B) per‑neuron Viridis plots (up to max_neurons_per_cluster)
#             for n_id in neuron_ids[:max_neurons_per_cluster]:
#                 real = np.asarray(layer_dict[n_id]["real_preactivations"])
#                 relu = np.maximum(real, 0.0)
#                 if real.size == 0:
#                     continue

#                 # record JSON of the raw activations
#                 neuron_plot_data[n_id] = {
#                     "real": real.tolist(),
#                     "postactivations": relu.tolist()
#                 }

#                 # rebuild the figure exactly as before
#                 a_grid, b_grid = np.mgrid[0:p, 0:p]
#                 ang_a = two_pi_over_p * freq * a_grid.ravel()
#                 ang_b = two_pi_over_p * freq * b_grid.ravel()
#                 X = (major_radius + minor_radius*np.cos(ang_b)) * np.cos(ang_a)
#                 Y = (major_radius + minor_radius*np.cos(ang_b)) * np.sin(ang_a)
#                 Z =  minor_radius * np.sin(ang_b)
#                 pos = relu.ravel() > 0

#                 traces = [
#                     TORUS_GHOST,
#                     curve_ab(freq, "sum0")["trace"].update(showlegend=False) or curve_ab(freq, "sum0")["trace"],
#                     curve_ab(freq, "diag")["trace"].update(showlegend=False) or curve_ab(freq, "diag")["trace"],
#                 ]
#                 # non‑ReLU points
#                 if (~pos).any():
#                     traces.append(go.Scatter3d(
#                         x=X[~pos], y=Y[~pos], z=Z[~pos],
#                         mode="markers",
#                         marker=dict(size=2, color="black", opacity=0.6),
#                         hoverinfo="skip", showlegend=False
#                     ))
#                 # ReLU‑positive
#                 C = relu.ravel()[pos]
#                 if pos.any():
#                     traces.append(go.Scatter3d(
#                         x=X[pos], y=Y[pos], z=Z[pos],
#                         mode="markers",
#                         marker=dict(size=4, opacity=0.9,
#                                     color=C, colorscale="Viridis",
#                                     cmin=0.0, cmax=float(C.max()),
#                                     showscale=False),
#                         hovertemplate="ReLU=%{marker.color:.3f}<extra></extra>",
#                         showlegend=False
#                     ))

#                 fig_neuron = go.Figure(data=traces)
#                 fig_neuron.update_layout(
#                     scene=dict(xaxis=dict(visible=False),
#                             yaxis=dict(visible=False),
#                             zaxis=dict(visible=False),
#                             aspectmode="data"),
#                     margin=dict(l=0,r=0,t=0,b=0)
#                 )

#                 # append to cluster_traces so it's in the same HTML stream
#                 cluster_traces.extend(fig_neuron.data)

#             # Write out the HTML exactly as before
#             fig = go.Figure(data=cluster_traces)
#             fig.update_layout(
#                 title=f"Layer {layer_idx} – f={freq} (cutoff {overlay_cutoff})",
#                 scene=dict(xaxis=dict(visible=False),
#                         yaxis=dict(visible=False),
#                         zaxis=dict(visible=False),
#                         aspectmode="data"),
#                 margin=dict(l=0,r=0,t=40,b=0),
#                 showlegend=True
#             )
#             html_parts = [
#                 "<!DOCTYPE html><html><head><meta charset='utf-8'>",
#                 f"<title>Layer {layer_idx} – cluster f={freq}</title>",
#                 "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head><body>",
#                 "<h2>Cluster-wide torus with neurons</h2>",
#                 pio.to_html(fig, include_plotlyjs=False, full_html=False),
#             ]
#             # Insert each neuron panel too
#             for n_id in neuron_plot_data:
#                 html_parts.append(f"<h2>Neuron index: {n_id}</h2>")
#                 # regenerate that individual figure just for embedding
#                 # (or keep a small cache of fig_neuron if you like)
#                 # here, for clarity, we rebuild it:
#                 real = np.asarray(layer_dict[n_id]["real_preactivations"])
#                 relu = np.maximum(real, 0.0)
#                 # … same code as above to build fig_neuron …
#                 # but you can factor that out into a helper if you prefer
#                 # then embed:
#                 html_parts.append(pio.to_html(fig_neuron, include_plotlyjs=False, full_html=False))
#             html_parts.append("</body></html>")

#             path_html = os.path.join(out_root, f"cluster_{freq}.html")
#             with open(path_html, "w") as f:
#                 f.write("".join(html_parts))
#             print(f"[put_neuron_preaccs_on_torus] wrote {path_html}")

#             # Finally, write the JSON with *all* raw data
#             cluster_json = {
#                 "p": p,
#                 "major_radius": major_radius,
#                 "minor_radius": minor_radius,
#                 "overlay_cutoff": overlay_cutoff,
#                 "mesh_res": mesh_res,
#                 "freq": freq,
#                 "overlay_curves": {
#                     "sum0": {"a_vals": curve_ab(freq, "sum0")["a_vals"],
#                             "b_vals": curve_ab(freq, "sum0")["b_vals"]},
#                     "diag": {"a_vals": curve_ab(freq, "diag")["a_vals"],
#                             "b_vals": curve_ab(freq, "diag")["b_vals"]},
#                 },
#                 "cluster_overlay_points": overlay_points,
#                 "neuron_plot_data": neuron_plot_data,
#             }
#             path_json = os.path.join(out_root, f"cluster_{freq}.json")
#             with open(path_json, "w") as jf:
#                 json.dump(cluster_json, jf, indent=2)
#             print(f"[put_neuron_preaccs_on_torus] wrote {path_json}")