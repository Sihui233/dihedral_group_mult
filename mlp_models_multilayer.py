from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class DonutMLP(nn.Module):
    """Base class shared by all MLP variants used in the project."""

    p: int               # size of the modular alphabet (labels 0 … p‑1)
    num_neurons: int      # width of every hidden layer

    # ---------------------------------------------------------------------
    # Interfaces that the training / analysis code relies on
    # ---------------------------------------------------------------------
    def _combine_embeddings(self,
                             vec_a: np.ndarray,
                             vec_b: np.ndarray) -> np.ndarray:
        """Default = concatenation.  Cheating variants override."""
        return np.concatenate([vec_a, vec_b], axis=0)
    
    def bias(self, params):
        """Return the bias vector of the *first* hidden layer."""
        raise NotImplementedError

    def extract_embeddings_ab(self, params):
        """Return (embedding_a, embedding_b) — each of shape (p, D)."""
        raise NotImplementedError

    def all_p_squared_embeddings(self, params):
        """Return the (p², d_in) matrix the network really sees."""
        emb_a, emb_b = self.extract_embeddings_ab(params)
        return np.stack([
            self._combine_embeddings(emb_a[i], emb_b[j])
            for i in range(self.p)            # row-major on (a,b)
            for j in range(self.p)
        ], axis=0)
    
    def call_from_embedding(self, emb: jnp.ndarray, params: dict):
        """
        Given a batch of concatenated embeddings (shape: [N, 2D]) and the full params dict,
        apply the MLP via direct linear algebra (no submodules) and return (logits, preacts).
        """
        hidden = emb
        preacts = []
        # apply each hidden layer: dense_i
        for layer_idx in range(1, self.num_layers + 1):
            key = f"dense_{layer_idx}"
            w = params[key]["kernel"]    # shape (in_features, num_neurons)
            b = params[key]["bias"]      # shape (num_neurons,)
            pre_act = jnp.dot(hidden, w) + b
            hidden = nn.relu(pre_act)
            preacts.append(pre_act)
        # output layer
        w_out = params["output_dense"]["kernel"]  # shape (num_neurons, p)
        b_out = params["output_dense"]["bias"]    # shape (p,)
        logits = jnp.dot(hidden, w_out) + b_out
        return logits, preacts

# =====================================================================
# Helper to build an arbitrary‑depth feed‑forward tower
# =====================================================================

def _forward_tower(x, num_layers, num_neurons, first_layer_name_prefix="dense"):
    """Utility: build *num_layers* Dense → ReLU blocks.

    Returns
    -------
    activation : jnp.ndarray
        Output after the final ReLU.
    preactivations : list[jnp.ndarray]
        List of pre‑ReLU tensors, one per hidden layer (index 0 == layer 1).
    dense0_kernel : jnp.ndarray
        Kernel of the *first* Dense layer — needed for contribution splits.
    """
    preacts = []
    activation = x

    for layer_idx in range(1, num_layers + 1):
        dense = nn.Dense(
            features=num_neurons,
            kernel_init=nn.initializers.he_normal(),
            name=f"{first_layer_name_prefix}_{layer_idx}")
        pre_act = dense(activation)
        activation = nn.relu(pre_act)
        preacts.append(pre_act)

        if layer_idx == 1:
            first_kernel = dense.variables["params"]["kernel"]

    return activation, preacts, first_kernel

# =====================================================================
# 1) One‑Hot concatenation
# =====================================================================
class MLPOneHot(DonutMLP):
    num_layers: int = 1
    features: int = 128
    @nn.compact
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        a_onehot = jax.nn.one_hot(a, self.p)
        b_onehot = jax.nn.one_hot(b, self.p)
        concat = jnp.concatenate([a_onehot, b_onehot], axis=-1)

        # Build the hidden tower
        hidden, preacts, kernel1 = _forward_tower(concat, self.num_layers, self.num_neurons,
                                                  first_layer_name_prefix="dense")
        # Split contributions of the first layer --------------------------
        contribution_a = jnp.dot(a_onehot, kernel1[: self.p, :])
        contribution_b = jnp.dot(b_onehot, kernel1[self.p : 2 * self.p, :])

        # Output layer -----------------------------------------------------
        logits = nn.Dense(features=self.p,
                          kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)
        return logits, preacts, contribution_a, contribution_b

    # Interfaces -----------------------------------------------------------
    def bias(self, params):
        return params["dense_1"]["bias"]
    
    def all_p_squared_embeddings(self, params=None):
        """
        Return a (p², 2 p) matrix where each row is the concatenation
        of the one-hot code for token a and token b.
        """
        eye = np.eye(self.p, dtype=np.float32)        # (p, p)

        # create the index lists for the Cartesian product (a, b)
        a_idx = np.repeat(np.arange(self.p), self.p)  # p²
        b_idx = np.tile  (np.arange(self.p), self.p)  # p²

        return np.concatenate([eye[a_idx], eye[b_idx]], axis=1)

    def extract_embeddings_ab(self, params):
        W = params["dense_1"]["kernel"]      # (2p, num_neurons)
        return W[: self.p, :], W[self.p : 2 * self.p, :]

# =====================================================================
# 2) One shared embedding (duplicated)
# =====================================================================
class MLPOneEmbed(DonutMLP):
    features: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        shared = nn.Embed(self.p, self.features, name="shared_embed",
                          embedding_init=nn.initializers.he_normal())
        a_emb = shared(a)
        b_emb = shared(b)
        concat = jnp.concatenate([a_emb, b_emb], axis=-1)

        hidden, preacts, kernel1 = _forward_tower(concat, self.num_layers, self.num_neurons,
                                                  first_layer_name_prefix="dense")
        contribution_a = jnp.dot(a_emb, kernel1[: self.features, :])
        contribution_b = jnp.dot(b_emb, kernel1[self.features :, :])

        logits = nn.Dense(self.p, kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)
        return logits, preacts, contribution_a, contribution_b

    def bias(self, params):
        return params["dense_1"]["bias"]

    def extract_embeddings_ab(self, params):
        emb = params["shared_embed"]["embedding"]
        return emb, emb

# =====================================================================
# 3) Two independent embeddings
# =====================================================================
class MLPTwoEmbed(DonutMLP):
    features: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        emb_a = nn.Embed(self.p, self.features, name="embedding_a",
                         embedding_init=nn.initializers.he_normal())
        emb_b = nn.Embed(self.p, self.features, name="embedding_b",
                         embedding_init=nn.initializers.he_normal())
        a_emb = emb_a(a)
        b_emb = emb_b(b)
        concat = jnp.concatenate([a_emb, b_emb], axis=-1)

        hidden, preacts, kernel1 = _forward_tower(concat, self.num_layers, self.num_neurons,
                                                  first_layer_name_prefix="dense")
        contribution_a = jnp.dot(a_emb, kernel1[: self.features, :])
        contribution_b = jnp.dot(b_emb, kernel1[self.features :, :])

        logits = nn.Dense(self.p, kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)
        return logits, preacts, contribution_a, contribution_b

    def bias(self, params):
        return params["dense_1"]["bias"]

    def extract_embeddings_ab(self, params):
        return (params["embedding_a"]["embedding"],   
                params["embedding_b"]["embedding"])

# =====================================================================
# "Cheating" variants (add instead of concatenate)
# =====================================================================
class MLPOneHot_cheating(DonutMLP):
    num_layers: int = 1
    features: int = 128
    @nn.compact
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        a_oh, b_oh = jax.nn.one_hot(a, self.p), jax.nn.one_hot(b, self.p)
        added = a_oh + b_oh

        hidden, preacts, kernel1 = _forward_tower(added, self.num_layers, self.num_neurons,
                                                  first_layer_name_prefix="dense")
        contribution_a = jnp.dot(a_oh, kernel1)
        contribution_b = jnp.dot(b_oh, kernel1)

        logits = nn.Dense(self.p, kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)
        return logits, preacts, contribution_a, contribution_b

    def bias(self, params):
        return params["dense_1"]["bias"]

    def extract_embeddings_ab(self, params):
        # one‑hot is identity
        eye = np.eye(self.p)
        return eye, eye
    
    def _combine_embeddings(self, vec_a, vec_b):
        return vec_a + vec_b

class MLPOneEmbed_cheating(DonutMLP):
    features: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        shared = nn.Embed(self.p, self.features, name="shared_embed",
                          embedding_init=nn.initializers.he_normal())
        a_emb, b_emb = shared(a), shared(b)
        added = a_emb + b_emb

        hidden, preacts, kernel1 = _forward_tower(added, self.num_layers, self.num_neurons,
                                                  first_layer_name_prefix="dense")
        contribution_a = jnp.dot(a_emb, kernel1)
        contribution_b = jnp.dot(b_emb, kernel1)

        logits = nn.Dense(self.p, kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)
        return logits, preacts, contribution_a, contribution_b

    def bias(self, params):
        return params["dense_1"]["bias"]

    def extract_embeddings_ab(self, params):
        emb = params["shared_embed"]["embedding"]
        return emb, emb
    
    def _combine_embeddings(self, vec_a, vec_b):
        return vec_a + vec_b

class MLPTwoEmbed_cheating(DonutMLP):
    features: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x, training: bool = False):
        a, b = x[:, 0], x[:, 1]
        emb_a = nn.Embed(self.p, self.features, name="embedding_a",
                         embedding_init=nn.initializers.he_normal())
        emb_b = nn.Embed(self.p, self.features, name="embedding_b",
                         embedding_init=nn.initializers.he_normal())
        a_emb, b_emb = emb_a(a), emb_b(b)
        added = a_emb + b_emb

        hidden, preacts, kernel1 = _forward_tower(added, self.num_layers, self.num_neurons,
                                                  first_layer_name_prefix="dense")
        contribution_a = jnp.dot(a_emb, kernel1)
        contribution_b = jnp.dot(b_emb, kernel1)

        logits = nn.Dense(self.p, kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)
        return logits, preacts, contribution_a, contribution_b

    def bias(self, params):
        return params["dense_1"]["bias"]

    def extract_embeddings_ab(self, params):
        return (params["embedding_a"]["embedding"],
                   params["embedding_b"]["embedding"])
    
    def _combine_embeddings(self, vec_a, vec_b):
        return vec_a + vec_b

class MLPOneEmbedResidual(DonutMLP):
    """
    Two-token MLP that mirrors the constant-attention transformer:

        • shared embedding  E
        • learned position biases  p₀ , p₁
        • token-wise linear   V : ℝᴰ → ℝᴴ
        • add        v_sum = V(a) + V(b)
        • project    mix   = W_O · v_sum        (second linear)
        • add mix to *token-1* stream only
        • Dense→ReLU tower on that stream
        • final logits = W_U · h_final
        • keeps the same public API (bias, extract_embeddings_ab, …)
    """
    features: int
    num_layers: int = 1            # for compatibility

    # ------------------------------------------------------------
    @nn.compact
    def __call__(self, x, *, training: bool = False):
        a, b = x[:, 0], x[:, 1]

        # (1) shared embedding + learned position offsets
        shared = nn.Embed(self.p, self.features,
                          embedding_init=nn.initializers.he_normal(),
                          name="shared_embed")
        pos = self.param("pos_bias", nn.initializers.zeros, (2, self.features))
        a_emb = shared(a) + pos[0]
        b_emb = shared(b) + pos[1]

        # (2) first linear “value” projection  V
        V_proj = nn.Dense(
            self.features,                 # 128  ← not num_neurons
            use_bias=False,
            kernel_init=nn.initializers.he_normal(),
            name="V_proj",
        )
        v_a = V_proj(a_emb)          # (B,H)
        v_b = V_proj(b_emb)
        v_sum = v_a + v_b            # (B,H)

        # (3) second linear “output” projection  W_O
        O_proj = nn.Dense(
            self.features,                 # 128
            use_bias=False,
            kernel_init=nn.initializers.he_normal(),
            name="O_proj",
        )
        mix = O_proj(v_sum)          # (B,H)

        # (4) build the residual stream for TOKEN 1 (b-position)
        resid1 = b_emb + mix         # exactly what the transformer adds

        # (5) optional post-ReLU tower (you had num_layers=1 anyway)
        hidden, preacts, kernel1 = _forward_tower(
            resid1,
            num_layers=self.num_layers,
            num_neurons=self.num_neurons,
            first_layer_name_prefix="dense"
        )

        # (6) final unembed  W_U   (same shape as transformer)
        logits = nn.Dense(self.p,
                          kernel_init=nn.initializers.he_normal(),
                          name="output_dense")(hidden)

        # contributions for analysis ----------------------------
        contribution_a = jnp.dot(a_emb, kernel1)   # (B,H)
        contribution_b = jnp.dot(b_emb, kernel1)   # (B,H)

        return logits, preacts, contribution_a, contribution_b

    # ---------------- analysis helpers -------------------------
    def bias(self, params):
        return params["dense_1"]["bias"]

    def extract_embeddings_ab(self, params):
        emb = params["shared_embed"]["embedding"]
        pos = params["pos_bias"]
        return emb + pos[0], emb + pos[1]

    def call_from_embedding(self, emb: jnp.ndarray, params: dict):
        """
        emb must be [..., 2*D] where D == dense_1.kernel.shape[0].
        first D dims → token-a embedding,
        next  D dims → token-b embedding.
        """
        D = params["dense_1"]["kernel"].shape[0]          # 128
        if emb.shape[-1] != 2 * D:
            raise ValueError(
                f"call_from_embedding expects {2*D} features, got {emb.shape[-1]}"
            )

        a_emb, b_emb = jnp.split(emb, 2, axis=-1)         # (…,D) , (…,D)

        # shared Dense → ReLU -------------------------------------------------
        W1, b1 = params["dense_1"]["kernel"], params["dense_1"]["bias"]  # (D,H), (H,)
        a_pre = a_emb @ W1 + b1
        b_pre = b_emb @ W1 + b1
        a_act = jnp.maximum(a_pre, 0.)
        b_act = jnp.maximum(b_pre, 0.)

        h_sum  = a_act + b_act                            # (…,H)

        # optional residual branch -------------------------------------------
        if "residual_proj" in params:
            W_res, b_res = params["residual_proj"]["kernel"], params["residual_proj"]["bias"]
            h_final = b_act + (b_emb @ W_res + b_res) + h_sum
        else:
            h_final = b_act + h_sum

        # output head ---------------------------------------------------------
        W_out, b_out = params["output_dense"]["kernel"], params["output_dense"]["bias"]
        logits = h_final @ W_out + b_out                               # (…,p)
        return logits, [b_pre]
    
