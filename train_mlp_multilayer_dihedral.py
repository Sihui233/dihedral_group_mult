#!/usr/bin/env python
# I commented out this files frequency .json log generation. 
import os
import numpy as np
try:
    import jax
    if all(d.platform != 'gpu' for d in jax.devices()):
        print("⚠️ No GPU detected — enabling multithreading for CPU.")
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=10"
except Exception:
    # If JAX isn't installed or fails to load, fall back safely
    pass
import jax
from jax import config
# config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from clu import metrics
from flax import struct
import optax  
import sys
import json
import flax.serialization as serialization
import jax.tree_util
import copy
from typing import Dict, Any, Tuple, Union, List

from itertools import product
import plotly.io as pio
pio.kaleido.scope.default_timeout = 60 * 5


jax.config.update("jax_traceback_filtering", 'off')
print("Devices available:", jax.devices())
import optimizers
import training
import collections
from collections import Counter 
import plotly.graph_objs as go, plotly.io as pio
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import report
# from persistent_homology_gpu import run_ph_for_point_cloud
from pca_diffusion_plots_w_helpers import (
    generate_pdf_plots_for_matrix,
    generate_interactive_diffusion_map_html,
)
from utils import compute_pytree_size
import DFT
import dihedral
from color_rules import colour_quad_mul_f        # ①  f·(a±b) mod p
from color_rules import colour_quad_mod_g      # ②  (a±b) mod g
from color_rules import colour_quad_a_only, colour_quad_b_only 
# import model MLP classes
from mlp_models_multilayer import DonutMLP, MLPOneEmbed, MLPOneHot, MLPTwoEmbed, MLPTwoEmbed_cheating, MLPOneEmbed_cheating, MLPOneHot_cheating, MLPOneEmbedResidual

if len(sys.argv) < 14:
    print("Usage: script.py <learning_rate> <weight_decay> <p> <batch_size> <optimizer> <epochs> <k> <batch_experiment> <num_neurons> <MLP_class> <features> <num_layers> <random_seed_int_1> [<random_seed_int_2> ...]")
    sys.exit(1)

print("start args parsing")
learning_rate = float(sys.argv[1])  # stepsize
weight_decay = float(sys.argv[2])     # L2 regularization penalty
p = int(sys.argv[3])
batch_size = int(sys.argv[4])
optimizer = sys.argv[5]
epochs = int(sys.argv[6])
k = int(sys.argv[7])
batch_experiment = sys.argv[8]
num_neurons = int(sys.argv[9])
MLP_class = sys.argv[10]
training_set_size = k * batch_size
features = int(sys.argv[11])
num_layers = int(sys.argv[12])
top_k = [1]
random_seed_ints = [int(arg) for arg in sys.argv[13:]]
num_models = len(random_seed_ints)
print(f"args: lr: {learning_rate}, wd: {weight_decay},nn: {num_neurons}, features: {features}, num_layer: {num_layers}")
print(f"Random seeds: {random_seed_ints}")

group_size = 2 * p

# generate for each seed
train_ds_list = []
for seed in random_seed_ints:
    x, y = dihedral.make_dihedral_dataset(p, batch_size, k, seed)
    train_ds_list.append((x, y))

x_batches = jnp.stack([x for (x, _) in train_ds_list])  # (num_models, k, batch_size, 2)
y_batches = jnp.stack([y for (_, y) in train_ds_list])  # (num_models, k, batch_size)
print("x_batches.shape =", x_batches.shape)
print("y_batches.shape =", y_batches.shape)

print(f"Number of training batches: {x_batches.shape[1]}")

# ---------------- FOURIER TRANSFORM (ADDED) ----------------
# REASON: implement custom group Fourier transform for D_n preacts

# assume `irreps` and `G` constructed same as mult
G, irreps = DFT.make_irreps_Dn(p)
freq_map = {}
for name, dim, R, freq in irreps:
    freq_map[name] = freq
    print(f"Checking {name}...")
    
    dihedral.check_representation_consistency(G, R, dihedral.mult, p)

print("made dataset")

dataset_size_bytes = (x_batches.size * x_batches.dtype.itemsize)
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
print(f"Dataset size per model: {dataset_size_mb:.2f} MB")

def positive_he_normal(key, shape, dtype=jnp.float32):
    init = jax.nn.initializers.he_normal()(key, shape, dtype)
    return jnp.abs(init)

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')
    l2_loss: metrics.Average.from_output('l2_loss')

model: DonutMLP
mlp_class_lower = f"{MLP_class.lower()}_{num_layers}"
model_class_map = {
    "no_embed": MLPOneHot,
    "one_embed": MLPOneEmbed,
    "one_embed_residual": MLPOneEmbedResidual,
    "two_embed": MLPTwoEmbed,
    "no_embed_cheating": MLPOneHot_cheating,
    "one_embed_cheating": MLPOneEmbed_cheating,
    "two_embed_cheating": MLPTwoEmbed_cheating,
    }
# Note these two maps can be replaced by better code checking if "cheating" in base_class_name, but for now I'm doing it this way cuz idk what I might add later
vector_addition_class_map = {
    "no_embed_cheating": MLPOneHot_cheating,
    "one_embed_cheating": MLPOneEmbed_cheating,
    "two_embed_cheating": MLPTwoEmbed_cheating,
    }
torus_class_map = {
    "no_embed": MLPOneHot,
    "one_embed": MLPOneEmbed,
    "two_embed": MLPTwoEmbed,
}
base_class_name = MLP_class.lower()

if base_class_name not in model_class_map:
    raise ValueError(f"Unknown MLP_class: {MLP_class}")
print(base_class_name)
if base_class_name not in torus_class_map and base_class_name not in vector_addition_class_map:
    raise ValueError(f"Unknown if MLP_class: {MLP_class} is vec add or torus")
if base_class_name in torus_class_map:
    num_principal_components = 4
if base_class_name in vector_addition_class_map:
    num_principal_components = 2

model_class = model_class_map[base_class_name]

kwargs = dict(p=group_size, num_neurons=num_neurons, num_layers=num_layers)
if "embed" in base_class_name:
    kwargs["features"] = features

model = model_class(**kwargs)
dummy_x = jnp.zeros(shape=(batch_size, 2), dtype=jnp.int32)

def cross_entropy_loss(y_pred, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).mean()

def total_loss(y_pred_and_l2, y):
    y_pred, pre_activation, l2_loss = y_pred_and_l2
    return cross_entropy_loss(y_pred, y) + l2_loss * weight_decay

def apply(variables, x, training=False):
    params = variables['params']
    batch_stats = variables.get("batch_stats", None)
    if batch_stats is None:
        batch_stats = {}
    outputs, updates = model.apply({'params': params, 'batch_stats': batch_stats}, x, training=training,
                                   mutable=['batch_stats'] if training else [])
    x_out, pre_activation, _, _ = outputs
    l2_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    return x_out, updates, l2_loss

def sample_hessian(prediction, sample):
    return (optimizers.sample_crossentropy_hessian(prediction, sample[0]), 0.0, 0.0)

def compute_metrics(metrics, *, loss, l2_loss, outputs, labels):
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    metric_updates = metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss, l2_loss=l2_loss)
    return metrics.merge(metric_updates)


print("model made")

def init_model(seed):
    rng_key = jax.random.PRNGKey(seed)
    variables = model.init(rng_key, dummy_x, training=False)
    return variables

variables_list = []
for seed in random_seed_ints:
    variables = init_model(seed)
    variables_list.append(variables)

compute_pytree_size(variables_list[0]['params'])

variables_batch = {}
variables_batch['params'] = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *(v['params'] for v in variables_list))
variables_batch['batch_stats'] = None

params_batch = variables_batch['params']

if optimizer == "adam":
    tx = optax.adam(learning_rate)
elif optimizer[:3] == "SGD":
    tx = optax.sgd(learning_rate, 0.0)
else:
    raise ValueError("Unsupported optimizer type")

def init_opt_state(params):
    return tx.init(params)

opt_state_list = []
for i in range(num_models):
    params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
    opt_state = init_opt_state(params_i)
    opt_state_list.append(opt_state)

opt_state_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *opt_state_list)

def create_train_state(params, opt_state, rng_key, batch_stats):
    state = training.TrainState(
        apply_fn=apply, params=params, tx=tx,
        opt_state=opt_state,
        loss_fn=total_loss,
        loss_hessian_fn=sample_hessian,
        compute_metrics_fn=compute_metrics,
        rng_key=rng_key,
        initial_metrics=Metrics,
        batch_stats=batch_stats,
        injected_noise=0.0
    )
    return state

states_list = []
for i in range(num_models):
    seed = random_seed_ints[i]
    rng_key = jax.random.PRNGKey(seed)
    params_i = jax.tree_util.tree_map(lambda x: x[i], params_batch)
    opt_state_i = jax.tree_util.tree_map(lambda x: x[i], opt_state_batch)
    batch_stats = None
    state = create_train_state(params_i, opt_state_i, rng_key, batch_stats)
    states_list.append(state)

states = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states_list)

initial_metrics_list = [state.initial_metrics.empty() for state in states_list]
initial_metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *initial_metrics_list)

### Added for test evaluation ###
# -----------------------------------------------------------
#  Build FULL evaluation grid for D_n  (all g,h ∈ D_n)
# -----------------------------------------------------------
# 1. build D_n elements list and index list
idx  = {g: i for i, g in enumerate(G)}

group_size = len(G)               # == 2 * p

# 2. build x_eval:  shape = (|G|², 2)   —— every row is (idx_g, idx_h)
x_eval = jnp.array(
    [[idx[g], idx[h]] for g in G for h in G],
    dtype=jnp.int32
)                                  # (4 p², 2)

# 3. build y_eval:  shape = (|G|²,)     —— every row is idx[g * h]
y_eval = jnp.array(
    [idx[dihedral.mult(g, h, p)] for g in G for h in G],
    dtype=jnp.int32
)                                  # (4 p²,)

# 4. duplicate eval data to every model (num_models) and do padding → batch structure
x_eval = jax.device_put(x_eval)
y_eval = jax.device_put(y_eval)

x_eval_expanded = jnp.tile(x_eval[None, :, :], (num_models, 1, 1))
y_eval_expanded = jnp.tile(y_eval[None, :],       (num_models, 1))

eval_batch_size   = batch_size 
total_eval_points = x_eval.shape[0]
num_full_batches  = total_eval_points // eval_batch_size
remain            = total_eval_points % eval_batch_size

if remain > 0:
    pad = eval_batch_size - remain
    x_pad = x_eval_expanded[:, :pad, :]
    y_pad = y_eval_expanded[:, :pad]
    x_eval_padded = jnp.concatenate([x_eval_expanded, x_pad], axis=1)
    y_eval_padded = jnp.concatenate([y_eval_expanded, y_pad], axis=1)
    num_eval_batches = num_full_batches + 1
else:
    x_eval_padded   = x_eval_expanded
    y_eval_padded   = y_eval_expanded
    num_eval_batches = num_full_batches

# → (num_models, num_eval_batches, eval_batch_size, …)
x_eval_batches = x_eval_padded.reshape(num_models, num_eval_batches,
                                       eval_batch_size, 2)
y_eval_batches = y_eval_padded.reshape(num_models, num_eval_batches,
                                       eval_batch_size)
print("eval grid:", x_eval.shape, "batches:", x_eval_batches.shape, "\n")

BASE_DIR = f"/home/mila/w/weis/scratch/DL/qualitative_{p}_{mlp_class_lower}_{num_neurons}_features_{features}_k_{k}"
os.makedirs(BASE_DIR, exist_ok=True)
model_dir = os.path.join(
    BASE_DIR,
    f"{p}_models_embed_{features}",
    f"p={p}_bs={batch_size}_nn={num_neurons}_wd={weight_decay}_epochs={epochs}_training_set_size={training_set_size}"
)
os.makedirs(model_dir, exist_ok=True)

# Logging dictionaries for metrics (per epoch)
log_by_seed = {seed: {} for seed in random_seed_ints}
epoch_dft_logs_by_seed = { seed: {} for seed in random_seed_ints }

# logs for effective embeddings, preactivations, and logits.
epoch_embedding_log = {}
epoch_preactivation_log = {}
epoch_logits_log = {}

# Build per-seed train vs test coords
coords_full = np.array(x_eval)   
train_coords_by_seed = {}
test_coords_by_seed  = {}

for i, seed in enumerate(random_seed_ints):
    x, y = train_ds_list[i]
    train_flat = np.array(x.reshape(-1, 2))
    seen = set(map(tuple, train_flat.tolist()))
    train_coords = np.array([xy for xy in coords_full if tuple(xy) in seen], dtype=int)
    test_coords  = np.array([xy for xy in coords_full if tuple(xy) not in seen], dtype=int)
    train_coords_by_seed[seed] = train_coords
    test_coords_by_seed[seed]  = test_coords


# Training and Evaluation Loops
@jax.jit
def train_epoch(states, x_batches, y_batches, initial_metrics):
    def train_step(state_metrics, batch):
        states, metrics = state_metrics
        x, y = batch
        new_states, new_metrics = jax.vmap(
            lambda state, metric, x, y: state.train_step(metric, (x, y)),
            in_axes=(0, 0, 0, 0)
        )(states, metrics, x, y)
        return (new_states, new_metrics), None
    initial_state_metrics = (states, initial_metrics)
    transposed_x = x_batches.transpose(1, 0, 2, 3)
    transposed_y = y_batches.transpose(1, 0, 2)
    (new_states, new_metrics), _ = jax.lax.scan(
        train_step,
        initial_state_metrics,
        (transposed_x, transposed_y)
    )
    return new_states, new_metrics

@jax.jit
def eval_model(states, x_batches, y_batches, initial_metrics):
    def eval_step(metrics, batch):
        x, y = batch
        new_metrics = jax.vmap(
            lambda state, metric, x, y: state.eval_step(metric, (x, y)),
            in_axes=(0, 0, 0, 0)
        )(states, metrics, x, y)
        return new_metrics, None
    metrics = initial_metrics
    transposed_x = x_batches.transpose(1, 0, 2, 3)
    transposed_y = y_batches.transpose(1, 0, 2)
    final_metrics, _ = jax.lax.scan(
        eval_step,
        metrics,
        (transposed_x, transposed_y)
    )
    return final_metrics

# # build all p² inputs [a, b]
# a_grid, b_grid = jnp.mgrid[0:p, 0:p]
# x_freq_all = jnp.stack([a_grid.ravel(), b_grid.ravel()], axis=-1).astype(jnp.int32)
# # num positive frequencies we care about
# max_freq = p // 2  
x_all = jnp.array(
    [[i, j] for i in range(group_size) for j in range(group_size)],
    dtype=jnp.int32
)

# @jax.jit
# def compute_dft_max_all_layers(params):
#     # run model on all inputs at once
#     _, pre_acts_all, _, _ = model.apply({'params': params}, x_freq_all, training=False)
#     # pre_acts_all is a list/tuple of arrays, one per layer: each of shape (p^2, num_neurons_layer)
#     all_mag_max = []
#     for pre in pre_acts_all:
#         # reshape to (p, p, num_neurons)
#         pre_grid = pre.reshape(p, p, pre.shape[-1])
#         # FFT along the 'a' axis
#         fft_grid = jnp.fft.fft(pre_grid, axis=0)
#         mag = jnp.abs(fft_grid)
#         # slice out positive frequencies 1…max_freq
#         mag_sub = mag[1 : max_freq+1, :, :]
#         # take max over b axis → (max_freq, num_neurons)
#         mag_max = jnp.max(mag_sub, axis=1)
#         all_mag_max.append(mag_max)
#     return all_mag_max  # list of (max_freq, num_neurons_layer)

# @jax.jit
# def compute_group_dft_energy_all_layers(params: dict) -> list[dict]:
#     """
#     For each hidden layer, compute the energy of pre-activations in each (irrep_r,irrep_s) basis.
#     Returns a list of dicts: one dict per layer mapping neuron_idx -> {(r,s): energy}
#     """
#     # build all (g,h) input pairs for D_n x D_n
#     x_all = jnp.array(
#         [[i, j] for i in range(group_size) for j in range(group_size)],
#         dtype=jnp.int32
#     )  # shape (4*n^2, 2)

#     # forward pass to get pre-activations for each layer: list of (group_size^2, num_neurons)
#     _, pre_acts_all, _, _ = model.apply({'params': params}, x_all, training=False)


#     energies_all = []
#     for pre in pre_acts_all:
#         # pre: jnp.ndarray of shape (group_size^2, num_neurons)
#         # compute group DFT coefficients and remap to energy
#         Fhat = DFT.group_dft_preacts(pre,G,irreps,group_size)
#         energy_map = DFT.remap_to_energy(Fhat)
#         # energy_map: {neuron_idx: {(r,s): float}}
#         energies_all.append(energy_map)
#     return energies_all

# @jax.jit
# def compute_margin_stats(params, xs, ys):
#     # xs: (N,2) int32; ys: (N,) int32
#     logits = model.apply({'params': params}, xs, training=False)[0]  # (N, C)
#     # correct‐class logits
#     correct = logits[jnp.arange(xs.shape[0]), ys]
#     # mask out correct class
#     one_hot = jax.nn.one_hot(ys, logits.shape[1], dtype=bool)
#     masked = jnp.where(one_hot, -1e9, logits)
#     runner = jnp.max(masked, axis=1)
#     margins = correct - runner  # shape (N,)
#     return jnp.min(margins), jnp.mean(margins)

energy_batch_size = 10 * batch_size



# === Training Loop ===
first_100_acc_epoch_by_seed = {seed: None for seed in random_seed_ints}
first_epoch_loss_by_seed = {seed: None for seed in random_seed_ints}
first_epoch_ce_loss_by_seed = {seed: None for seed in random_seed_ints} 

# for model_idx in range(num_models):
#     params_i = jax.tree_util.tree_map(lambda x: x[model_idx], states.params)
#     all_mag_max = compute_dft_max_all_layers(params_i)
#     mag_np_list  = [np.array(m) for m in all_mag_max]
#     seed = random_seed_ints[model_idx]
#     # record under epoch 0
#     epoch_dft_logs_by_seed[seed].setdefault(0, {})
#     for layer_idx, mag_np in enumerate(mag_np_list):
#         layer_dict = epoch_dft_logs_by_seed[seed][0].setdefault(layer_idx, {})
#         for neuron_idx in range(mag_np.shape[1]):
#             freq_dict = { str(f): float(mag_np[f-1, neuron_idx])
#                           for f in range(1, max_freq+1) }
#             layer_dict[neuron_idx] = freq_dict
# print("Logged initial (random‐init) DFT → epoch 0")
# for model_idx in range(num_models):
#     params_i = jax.tree_util.tree_map(lambda x: x[model_idx], states.params)
#     energies_all = compute_group_dft_energy_all_layers(params_i)
#     seed = random_seed_ints[model_idx]
#     epoch_dft_logs_by_seed[seed].setdefault(0, {})
#     for layer_idx, energy_map in enumerate(energies_all):
#         layer_dict = epoch_dft_logs_by_seed[seed][0].setdefault(layer_idx, {})
#         for neuron_idx, freq_eng in energy_map.items():
#             layer_dict[neuron_idx] = {k: float(v) for k, v in freq_eng.items()}
# print("Logged initial (random-init) group DFT energies → epoch 0")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    states, train_metrics = train_epoch(states, x_batches, y_batches, initial_metrics)
    train_losses = []
    train_accuracies = []

    do_eval = (epoch + 1) % 5000 == 0 or (epoch + 1) == epochs
    if do_eval:
        print(f"\n--- Test Evaluation at Epoch {epoch + 1} ---")
        test_metrics = eval_model(states, x_eval_batches, y_eval_batches, initial_metrics)
        test_losses = []
        test_accuracies = []

    for i in range(num_models):
        seed = random_seed_ints[i]
        # Train metrics
        train_metric = jax.tree_util.tree_map(lambda x: x[i], train_metrics)
        train_metric = train_metric.compute()
        train_loss = float(train_metric['loss'])
        train_acc = float(train_metric['accuracy'])
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Model {i + 1}/{num_models}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.2%}")

        # Test metrics
        if do_eval:
            test_metric = jax.tree_util.tree_map(lambda x: x[i], test_metrics)
            test_metric = test_metric.compute()
            test_loss = float(test_metric['loss'])
            test_accuracy = float(test_metric['accuracy'])
            test_l2_loss = float(test_metric['l2_loss'])
            test_ce_loss = test_loss - weight_decay * test_l2_loss
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Model {i + 1}/{num_models}: Test CE Loss: {test_ce_loss:.6f}, Test Total Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2%}")

            if first_100_acc_epoch_by_seed[seed] is None and test_accuracy >= 1.0:
                first_100_acc_epoch_by_seed[seed] = epoch + 1
                first_epoch_loss_by_seed[seed] = test_loss
                first_epoch_ce_loss_by_seed[seed] = test_ce_loss

                print(
                    f"*** Seed {seed} first reached 100% accuracy at epoch {epoch + 1} "
                    f"with total loss {test_loss:.6f} and CE-only loss {test_ce_loss:.6f} ***"
                )

            # Log to dictionary 
            params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)
            weight_norm = float(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params_i)))

            log_by_seed[seed][epoch + 1] = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_ce_loss": test_ce_loss, 
                "test_accuracy": test_accuracy,
                "l2_weight_norm": weight_norm,
                "first_reach_100%": epoch+1
            }
            params_i = jax.tree_util.tree_map(lambda x: x[i], states.params)

            # train margin commented out for scaling 
            # tc = train_coords_by_seed[seed]
            # ty = jnp.mod(tc[:, 0] + tc[:, 1], p)
            # train_min, train_avg = compute_margin_stats(params_i, jnp.array(tc), ty)

            # # test margin
            # if(k**2!=p**2):
            #     vc = test_coords_by_seed[seed]
            #     vy = jnp.mod(vc[:, 0] + vc[:, 1], p)
            # else:
            #     vc = tc
            #     vy = ty
            # test_min, test_avg = compute_margin_stats(params_i, jnp.array(vc), vy)

            # # total margin (use full eval grid)
            # total_min, total_avg = compute_margin_stats(params_i, x_eval, y_eval)

            # # update log
            # log_by_seed[seed][epoch + 1].update({
            #     "train_margin":      float(train_min),
            #     "train_avg_margin":  float(train_avg),
            #     "test_margin":       float(test_min),
            #     "test_avg_margin":   float(test_avg),
            #     "min_total_margin":      float(total_min),
            #     "total_avg_margin":  float(total_avg),
            # })


    current_epoch = epoch + 1 

# === Final Evaluation on Test Set ===
print("Starting final evaluation...")
test_metrics = eval_model(states, x_eval_batches, y_eval_batches, initial_metrics)
network_metrics = {}  # To store loss and l2_loss for each seed.
for i in range(num_models):
    test_metric = jax.tree_util.tree_map(lambda x: x[i], test_metrics)
    test_metric = test_metric.compute()
    test_loss = float(test_metric["loss"])
    test_accuracy = float(test_metric["accuracy"])
    test_l2_loss = float(test_metric["l2_loss"])  # extract l2_loss from metrics
    print(f"Model {i + 1}/{num_models}: Final Test Loss: {test_loss:.6f}, Final Test Accuracy: {test_accuracy * 100:.2f}%")
    network_metrics[random_seed_ints[i]] = {"loss": test_loss, "l2_loss": test_l2_loss}
    # if test_accuracy >= 0.999:
    #     params_file_path = os.path.join(
    #         model_dir,
    #         f"params_p_{p}_{optimizer}_ts_{training_set_size}_"
    #         f"bs={batch_size}_nn={num_neurons}_lr={learning_rate}_wd={weight_decay}_"
    #         f"rs_{random_seed_ints[i]}.params"
    #     )
    #     os.makedirs(os.path.dirname(params_file_path), exist_ok=True)
    #     with open(params_file_path, 'wb') as f:
    #         f.write(serialization.to_bytes(jax.tree_util.tree_map(lambda x: x[i], states.params)))
    #     print(f"Model {i + 1}: Parameters saved to {params_file_path}")
    # else:
    #     print(f"Model {i + 1}: Test accuracy did not exceed 99.9%. Model parameters wont be saved")
    #     print(f"\n--- Misclassified Test Examples for Model {i + 1} ---")
    #     logits, _, _, _ = model.apply({'params': jax.tree_util.tree_map(lambda x: x[i], states.params)}, x_eval, training=False)
    #     predictions = jnp.argmax(logits, axis=-1)
    #     y_true = y_eval
    #     incorrect_mask = predictions != y_true
    #     incorrect_indices = jnp.where(incorrect_mask)[0]
    #     if incorrect_indices.size > 0:
    #         misclassified_x = x_eval[incorrect_indices]
    #         misclassified_y_true = y_true[incorrect_indices]
    #         misclassified_y_pred = predictions[incorrect_indices]
    #         print(f"Total Misclassifications: {len(incorrect_indices)}")
    #         for idx, (x_vals, true_label, pred_label) in enumerate(zip(misclassified_x, misclassified_y_true, misclassified_y_pred), 1):
    #             a_val, b_val = x_vals
    #             print(f"{idx}. a: {int(a_val)}, b: {int(b_val)}, True: {int(true_label)}, Predicted: {int(pred_label)}")

# Build new dictionaries based on final epoch grouping for DFT logs ===
final_epoch = epochs
seed_dict_freqs_list = {}
for seed in random_seed_ints:
    seed_dict_freqs_list[seed] = set()
    # Grab the per-layer logs at the final epoch
    final_epoch_log = epoch_dft_logs_by_seed[seed].get(final_epoch, {})
    # For each layer, group neurons by their strongest frequency
    for layer_idx, neuron_dict in final_epoch_log.items():
        grouping = {}
        for neuron_idx, dft_dict in neuron_dict.items():
            # find the freq with max magnitude
            max_key = max(dft_dict, key=dft_dict.get)
            grouping.setdefault(max_key, []).append(neuron_idx)
            seed_dict_freqs_list[seed].add(max_key)

        # For each frequency, build an epoch‐by‐epoch log for this layer
        for freq, neuron_list in grouping.items():
            new_dict = {}
            for epoch_num, layers_log in epoch_dft_logs_by_seed[seed].items():
                layer_logs = layers_log.get(layer_idx, {})
                filtered = {
                    str(n): layer_logs[n]
                    for n in neuron_list
                    if n in layer_logs
                }
                if filtered:
                    new_dict[epoch_num] = filtered

            # Write out JSON with layer and freq in the filename
            output_filepath = os.path.join(
                model_dir,
                f"layer_{layer_idx}_frequency_{freq}_log_seed_{seed}.json"
            )
            with open(output_filepath, 'w') as f:
                json.dump(new_dict, f, indent=2)
            print(
                f"Frequency log for layer {layer_idx}, freq {freq} "
                f"(seed {seed}) saved to {output_filepath}"
            )

for seed in random_seed_ints:
    # get the unique freqs from earlier for filename
    freq_set = seed_dict_freqs_list.get(seed, set())       # a Python set of ints
    if not freq_set:                                       # safety check
        freqs_str = "none"                                 # fallback 
    else:
        freqs_sorted = sorted(freq_set)                    # e.g. [1, 3, 5]
        freqs_str = ",".join(map(str, freqs_sorted))       # "1,3,5"
    # write the JSON
    log_file_path = os.path.join(
        model_dir,
        f"log_features_{features}_({freqs_str})_seed_{seed}.json"
    )
    with open(log_file_path, "w") as f:
        json.dump(log_by_seed[seed], f, indent=2)
    print(f"Final log for seed {seed} saved to {log_file_path}")

from plots_multilayer import (
    plot_cluster_preactivations,
    summed_preactivations,
    summed_postactivations,
    plot_cluster_to_logits,
    plot_all_clusters_to_logits,
    reconstruct_sine_fits_multilayer_logn_fits_layers_after_2,
    fit_sine_wave_multi_freq
)

def convert_to_builtin_type(obj):
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_builtin_type(v) for v in obj]
    else:
        return obj

def get_all_preacts_and_embeddings(
        *,                        
        model: DonutMLP,
        params: dict,
        p: int | None = None,
        clusters_by_layer: list[dict[int, list[int]]] | None = None,
):
    
    """
    Build the (p², d_in) matrix that actually feeds the first Dense layer
    and return

    Returns
    -------
    preacts : list[np.ndarray]
        A list of length `model.num_layers`, where
        `preacts[L]` has shape `(p², width_L)` of the raw pre-ReLU activations.
    X_in : np.ndarray
        The `(p², d_in)` input matrix formed by all (a,b) embedding pairs.
    weights : list[np.ndarray]
        Hidden-layer weight kernels; `weights[L]` has shape
        `(in_dim_L, width_L)`.
    cluster_contribs : dict[int, np.ndarray]
        For every frequency `f` in the **last** hidden layer this returns
        a matrix of shape `(p², p)`:
    cluster_weights : dict[int, np.ndarray]
        For every frequency `f` in the last hidden layer, this is the slice
        of the output-layer kernel that feeds the logits from the neurons
        in cluster `f`.  Shape: `(|cluster_f|, p)`.
    
            H_cluster @ W_block
        where
        - `H_cluster` is the ReLU’d activations of the neurons in cluster `f`
          at every of the `p²` inputs, and
        - `W_block` is the slice of the output layer’s weight matrix
          corresponding to those same neurons.
    """
    if clusters_by_layer is None:
        raise ValueError("clusters_by_layer cannot be None")

    p = p or model.p
    X_in = model.all_p_squared_embeddings(params)                # (p², d_in)

    # forward pass once to get *pre-activations*
    _, preacts = model.call_from_embedding(jnp.asarray(X_in), params)
    preacts_np = [np.asarray(layer) for layer in preacts]        # list[(p², width_L)]
    # convert last layer to *post-ReLU activations*
    H_last = np.maximum(preacts_np[-1], 0.0)                     # (p², width_{L})

    # collect hidden-layer kernels 
    weights_np = [np.asarray(params[f"dense_{l}"]["kernel"])
                  for l in range(1, model.num_layers + 1)]

    # build cluster-wise *logit contributions*
    W_out = np.asarray(params["output_dense"]["kernel"])         # (width_L, p)
    cluster_contribs: dict[int, np.ndarray] = {}
    cluster_weights : dict[int, np.ndarray] = {}
    last_layer_clusters = clusters_by_layer[-1]                  # freq → [ids]
    for freq, neuron_ids in last_layer_clusters.items():
        if not neuron_ids:                  # skip empty clusters
            continue
        H_cluster = H_last[:, neuron_ids]               # (p², |cluster|)
        W_block   = W_out[neuron_ids, :]                # (|cluster|, p)
        C_freq    = H_cluster @ W_block                 # (p², p)
        cluster_contribs[freq] = C_freq
        cluster_weights[freq]  = W_block  

    return preacts_np, X_in, weights_np, cluster_contribs, cluster_weights

def make_some_jsons(
    *,
    preacts: list[np.ndarray],
    p: int,
    clusters_by_layer: list[dict[int, list[int]]],
    cluster_weights_to_logits: dict[int, np.ndarray],
    save_dir: str,
    subdir: str = "json",
    float_dtype=np.float32,
    sanity_check: bool = True,
    cluster_contribs_to_logits: dict[int, np.ndarray] | None = None,
) -> str:
    """
    Writes one JSON per *last layer* cluster: cluster_{freq}.json
    For each neuron in the cluster (keyed by its neuron_idx as a string), stores:
      - "preactivations": (p^2,)
      - "w_out":          (p,)
      - "contribs_to_logits": (p^2, p) = ReLU(preacts)[:,None] * w_out[None,:]

    Safety checks:
      • Ensures preacts[-1] is (p^2, width_last)
      • Ensures W_block is (|cluster|, p)
      • Ensures neuron_ids are within [0, width_last)
      • Optional exactness check vs. cluster_contribs_to_logits[freq]
    """
    # ---- global shape checks
    if not preacts:
        raise ValueError("make_some_jsons: empty `preacts`.")
    Z_last = np.asarray(preacts[-1])  # (p^2, width_last)
    n_rows, width_last = Z_last.shape
    if n_rows != p * p:
        raise ValueError(f"make_some_jsons: expected p^2={p*p} rows, got {n_rows}.")
    if not clusters_by_layer:
        raise ValueError("make_some_jsons: empty `clusters_by_layer`.")

    last_layer_clusters = clusters_by_layer[-1] or {}
    if not isinstance(last_layer_clusters, dict):
        raise TypeError("make_some_jsons: clusters_by_layer[-1] must be a dict {freq -> [neuron_ids]}.")

    json_root = os.path.join(save_dir, subdir)
    os.makedirs(json_root, exist_ok=True)

    for freq, neuron_ids in last_layer_clusters.items():
        if not neuron_ids:
            continue

        # Pull the aligned output weights block (built with the SAME order as neuron_ids)
        W_block = cluster_weights_to_logits.get(freq, None)
        if W_block is None:
            # Nothing to write if we don't have this cluster's output weights
            continue
        W_block = np.asarray(W_block)  # (|cluster|, p)

        # ---- index validation & alignment
        ids = np.asarray(neuron_ids, dtype=int)  # (|cluster|)
        valid_mask = (ids >= 0) & (ids < width_last)
        if not np.all(valid_mask):
            bad = ids[~valid_mask].tolist()
            # Filter both ids and W_block rows to keep alignment
            ids = ids[valid_mask]
            W_block = W_block[valid_mask, :]
            if ids.size == 0:
                # No valid neurons remain
                continue
            # (optional) log: print(f"[make_some_jsons] freq={freq}: dropped invalid neuron ids {bad}")

        # ---- shape checks after filtering
        if W_block.shape[0] != ids.shape[0]:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block rows ({W_block.shape[0]}) "
                f"≠ number of neuron ids ({ids.shape[0]})."
            )
        if W_block.shape[1] != p:
            raise ValueError(
                f"make_some_jsons: for freq={freq}, W_block has {W_block.shape[1]} columns, expected p={p}."
            )

        # Gather per-neuron preacts and ReLU
        Z_cluster = Z_last[:, ids]                 # (p^2, |cluster|)
        H_cluster = np.maximum(Z_cluster, 0.0)     # (p^2, |cluster|)

        # Vectorized per-neuron contributions: (p^2, |cluster|, p)
        contribs = H_cluster[:, :, None] * W_block[None, :, :]

        # Optional correctness check against provided cluster_contribs_to_logits
        if sanity_check and (cluster_contribs_to_logits is not None):
            C_freq_expected = np.asarray(cluster_contribs_to_logits.get(freq))
            if C_freq_expected is not None and C_freq_expected.size:
                C_sum = contribs.sum(axis=1)  # (p^2, p)
                if C_freq_expected.shape != C_sum.shape:
                    raise ValueError(
                        f"make_some_jsons: cluster_contribs_to_logits[{freq}] has shape {C_freq_expected.shape}, "
                        f"expected {C_sum.shape}."
                    )
                if not np.allclose(C_sum, C_freq_expected, rtol=1e-5, atol=1e-6):
                    raise ValueError(
                        f"make_some_jsons: contribution mismatch for freq={freq} "
                        f"(sum of per-neuron ≠ cluster total)."
                    )

        # Build JSON payload { "<neuron_idx>": {...}, ... } preserving original order
        payload = {}
        for j, nid in enumerate(ids.tolist()):
            payload[str(int(nid))] = {
                "preactivations": Z_cluster[:, j].astype(float_dtype).tolist(),   # (p^2,)
                "w_out":          W_block[j, :].astype(float_dtype).tolist(),     # (p,)
                "contribs_to_logits": contribs[:, j, :].astype(float_dtype).tolist(),  # (p^2, p)
            }

        out_path = os.path.join(json_root, f"cluster_{freq}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)

    return json_root

print("starting main analysis loop")
rho_cache  = DFT.build_rho_cache(G, irreps)
dft_fn     = DFT.jit_wrap_group_dft(rho_cache, irreps, group_size)
subgroups = dihedral.enumerate_subgroups_Dn(p)   # n = group_size//2
for seed_idx, seed in enumerate(random_seed_ints):
    graph_dir = os.path.join(model_dir, f"graphs_seed_{seed}_refined")
    paper_graph_dir = os.path.join(model_dir, f"p_graphs_seed_{seed}_refined")

    os.makedirs(graph_dir, exist_ok=True)
    model_params_seed = jax.tree_util.tree_map(lambda x: x[seed_idx], states.params)
    x_all = jnp.array([[g, h]
                       for g in range(group_size)
                       for h in range(group_size)],
                      dtype=jnp.int32)
    _, pre_acts_all, left, right = model.apply({'params': model_params_seed},
                                        x_all, training=False)

    tol = 6e-6
    layers_freq = []
    for layer_idx in range(num_layers):
        prei      = pre_acts_all[layer_idx]
        prei_grid = prei.reshape(group_size, group_size, -1)
        # ### test DFT by verifying reconstruction
        # Fhat      = dft_fn(prei)
        # recon = DFT.inverse_group_dft(Fhat, rho_cache, irreps, group_size, prei_grid.shape[-1])
        # abs_err = jnp.max(jnp.abs(recon - prei_grid))
        # rel_err = (jnp.linalg.norm(recon - prei_grid) /
        #         (jnp.linalg.norm(prei_grid) + 1e-12))

        # print(f"max abs error = {abs_err:.2e} | rel error = {rel_err:.2e}")
        # assert abs_err < tol, f"ABS error too large: {abs_err}"
        
        
        cluster_tau = 1e-3
        color_rule = colour_quad_a_only
        # color_rule = colour_quad_mod_g
        t1 = 2.0 if group_size < 50 else 3
        t2 = 2.0 if group_size < 50 else 3
        artifacts = report.prepare_layer_artifacts(prei_grid, #(G, G, N)
                            left, right, #(G*G, N)
                            dft_fn, irreps, freq_map,
                            prune_cfg={"thresh1": t1, "thresh2": t2, "seed": 0})
        
        coset_masks_L = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="left")
        coset_masks_R = dihedral.build_coset_masks(G, subgroups, dihedral.mult, p, side="right")
        report.make_layer_report(prei_grid,left,right,p,
                                 dft_fn, irreps, 
                                 coset_masks_L, coset_masks_R,
                                 graph_dir, cluster_tau, color_rule,
                                 artifacts
                                 )
        clusters_layer = artifacts["freq_cluster"]
        layers_freq.append(clusters_layer)
        
        # report.export_cluster_neuron_pages_2x4(prei_grid,left,right,
        #                          dft_fn, irreps, 
        #                          paper_graph_dir,
        #                          artifacts,
        #                          rounding_scale=10
        #                          )
        
        diag_labels = artifacts["diag_labels"]
        names = artifacts["names"]
        approx = report.summarize_diag_labels(diag_labels,p,names)
        filename = f"approx_summary_p{p}.json"
        filepath = os.path.join(graph_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(approx, f, ensure_ascii=False, indent=2)

        print(f"Saved summary to {filepath}")

####################
    preacts, X_emb, input_weights, cluster_contribs_to_logits, cluster_weights_to_logits = get_all_preacts_and_embeddings(
        model=model,
        params=model_params_seed,
        p=group_size,
        clusters_by_layer=layers_freq,
    )

    pdf_root = os.path.join(graph_dir, "pdf_plots", f"seed_{seed}")
    os.makedirs(pdf_root, exist_ok=True)

    json_root = make_some_jsons(
        preacts=preacts,
        p=group_size,
        clusters_by_layer=layers_freq,                  # == dominant_freq_clusters
        cluster_weights_to_logits=cluster_weights_to_logits,  # dict[freq] -> (|cluster|, p)
        cluster_contribs_to_logits=cluster_contribs_to_logits,# optional correctness check
        save_dir=pdf_root,
        sanity_check=True,
    )

    # clusters to logits
    for freq, C_freq in cluster_contribs_to_logits.items():
        # C_freq is (p², p): the total contribution of cluster “freq” to each logit
        generate_pdf_plots_for_matrix(
            C_freq, p, save_dir=pdf_root, seed=seed,
            freq_list=[freq],
            tag=f"cluster_contributions_to_logits_freq={freq}",
            tag_q = "full",
            colour_rule = colour_quad_mod_g,
            class_string=mlp_class_lower,
            num_principal_components=num_principal_components,
        )
    