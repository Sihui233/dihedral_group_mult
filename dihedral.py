import jax
import jax.numpy as jnp
import math
import numpy as np

def dn_elements(n: int):
    """Return list G of D_n elements and index dict idx."""
    G = [('rot', i) for i in range(n)] + [('ref', i) for i in range(n)]
    idx = {g: i for i, g in enumerate(G)}
    return G, idx

def mult(g, h, n):
    # r_k=r^k, s_l=s r^l
    # D_n multiplication logic
    tg, k = g; th, l = h
    if tg=='rot' and th=='rot': return ('rot', (k + l) % n)
    if tg=='rot' and th=='ref': return ('ref', (l - k) % n)
    if tg=='ref' and th=='rot': return ('ref', (k + l) % n)
    if tg=='ref' and th=='ref': return ('rot', (l - k) % n)
    raise ValueError

# DIHEDRAL: helpers to map indices <-> group elements and to multiply indices
# `G` is the list of D_n elements and `idx` is its inverse map (built once).
def idx_mul(i: int, j: int, G, idx, p: int) -> int:
    """Return the index of the group product G[i] * G[j] in D_n."""
    g, h = G[i], G[j]
    return idx[mult(g, h, p)]

def make_dihedral_dataset(n, batch_size, num_batches, seed):
    # enumerate D_n elements
    G = [('rot',i) for i in range(n)] + [('ref',i) for i in range(n)]
    idx = {g:i for i,g in enumerate(G)}
    total = num_batches * batch_size
    # all pairs and labels
    pairs = []
    labels = []
    for g in G:
        for h in G:
            pairs.append((idx[g], idx[h]))
            labels.append(idx[mult(g,h,n)])
    # shuffle and batch
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, len(pairs))[:total]
    arr = jnp.array(pairs)[perm]
    lbl = jnp.array(labels)[perm]
    arr = arr.reshape(num_batches, batch_size, 2)
    lbl = lbl.reshape(num_batches, batch_size)
    return arr, lbl

def check_representation_consistency(G, R, mult, n, tol=1e-6):
    for g in G:
        for h in G:
            lhs = R(mult(g, h, n))
            rhs = R(g) @ R(h)
            err = jnp.linalg.norm(lhs - rhs)
            if err > tol:
                print(f"Inconsistency at g={g}, h={h}, error={err:.2e}")


def enumerate_subgroups_Dn(n):
    subs = []
    seen = set()                      

    def add(name, H):
        key = frozenset(H)            
        if key not in seen:
            seen.add(key)
            subs.append((name, H))
    # 1) rotation sg C_d
    for d in range(1, n+1):
        if n % d == 0:
            step = n // d
            H = [('rot', i*step) for i in range(d)]
            add(f"C_{d}", H)

    # 2) simple ref subgroup order 2 {e, s r^m}
    for m in range(n):
        H = [('rot', 0), ('ref', m)]   # ('ref', m) = s r^m
        add(f"Refl2_{m}", H)

    # 3) dihedral subgroup of order 2d  D_d = ⟨ r^{n/d}, s r^m ⟩
    for d in range(2, n+1):
        if n % d == 0:
            step = n // d
            for m in range(step):  
                rots  = [('rot', t*step) for t in range(d)]
                refls = [('ref', (m + t*step) % n) for t in range(d)]
                H = rots + refls
                add(f"Dih_{d}_axis_{m}", H)

    return subs

def is_subgroup(H, mult, inv, n):
    e = ('rot', 0)
    if e not in H:
        return False
    # closed form
    for a in H:
        for b in H:
            if mult(a, b, n) not in H:
                return False
    # inv exists
    for a in H:
        if inv(a, n) not in H:
            return False
    return True

def inv(g, n):
    t,k = g
    return ('rot', (-k)%n) if t=='rot' else ('ref', k)

def build_coset_masks(G, subgroups, mult, p, side="left"):
    index_of = {g:i for i,g in enumerate(G)}
    coset_masks = {}
    for H_name, H_elems in subgroups:
        seen = set(); cid = 0
        for g in G:
            gi = index_of[g]
            if gi in seen: 
                continue
            if side == "left":          # gH
                coset_idx = [ index_of[mult(g, h, p)] for h in H_elems ]
            else:                        # Hg
                coset_idx = [ index_of[mult(h, g, p)] for h in H_elems ]
            for j in coset_idx: 
                seen.add(j)
            mask = np.zeros(len(G), dtype=bool)
            mask[coset_idx] = True
            coset_masks[(H_name, cid)] = mask
            cid += 1
    return coset_masks

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple

def mult_chain(elems: List[Tuple[str,int]], n: int):
    """
    Multiply a sequence of D_n elements left-to-right: elems[0]*elems[1]*...*elems[-1].
    elems are ('rot', k) or ('ref', k).
    """
    if not elems:
        return ('rot', 0)
    acc = elems[0]
    for g in elems[1:]:
        acc = mult(acc, g, n)
    return acc

def build_cayley_table(n: int):
    """
    Returns:
      G       : list of group elements in fixed order
      idx     : dict mapping element -> index
      table   : jnp.int32 array of shape (|G|, |G|) with table[i,j] = idx[G[i]*G[j]]
    """
    G = [('rot',i) for i in range(n)] + [('ref',i) for i in range(n)]
    idx = {g:i for i,g in enumerate(G)}
    m = len(G)
    table_np = np.zeros((m, m), dtype=np.int32)
    for i, g in enumerate(G):
        for j, h in enumerate(G):
            table_np[i, j] = idx[mult(g, h, n)]
    return G, idx, jnp.array(table_np, dtype=jnp.int32)

def make_dihedral_dataset_k_ary(n: int,
                                batch_size: int,
                                num_batches: int,
                                seed: int,
                                arity: int = 3,
                                exhaustive: bool = False):
    """
    Build training batches for sequences of length = arity.
    Inputs x: shape (num_batches, batch_size, arity) with indices in [0, 2n).
    Labels y: shape (num_batches, batch_size) with index of the product.
    
    If exhaustive=True, we enumerate all (2n)^arity tuples in random order
    and then take the first (num_batches*batch_size) of them (only sensible for small n, arity).
    Otherwise we uniform-sample with replacement.
    """
    assert arity >= 2, "arity must be >= 2"
    G, idx, table = build_cayley_table(n)
    m = len(G)  # = 2n
    total = num_batches * batch_size
    key = jax.random.PRNGKey(seed)

    if exhaustive:
        # Enumerate ALL tuples and shuffle. Be careful: size explodes as (2n)^arity
        all_seq = np.array(np.meshgrid(*[np.arange(m) for _ in range(arity)], indexing='ij'))
        all_seq = all_seq.reshape(arity, -1).T  # shape: (m^arity, arity)
        perm = jax.random.permutation(key, all_seq.shape[0])[:total]
        arr = jnp.array(all_seq, dtype=jnp.int32)[perm]
    else:
        arr = jax.random.randint(key, shape=(total, arity), minval=0, maxval=m, dtype=jnp.int32)

    # Labels via associative reduction over the Cayley table
    # Start with first column, then fold with table
    res = arr[:, 0]
    for t in range(1, arity):
        res = table[res, arr[:, t]]

    x = arr.reshape(num_batches, batch_size, arity)
    y = res.reshape(num_batches, batch_size)
    return x, y, G, idx, table

def make_eval_grid_k_ary(G, idx, table, n: int, arity: int, batch_size: int):
    """
    Build the full evaluation set over all |G|^arity tuples.
    Returns x_eval_batches, y_eval_batches, with padding to multiples of batch_size.
    """
    m = len(G)
    # All tuples: shape (m^arity, arity)
    axes = [np.arange(m) for _ in range(arity)]
    all_seq = np.array(np.meshgrid(*axes, indexing='ij'))
    all_seq = all_seq.reshape(arity, -1).T  # (m^arity, arity)
    x_eval = jnp.array(all_seq, dtype=jnp.int32)

    # Compute labels with table-reduction
    res = x_eval[:, 0]
    for t in range(1, arity):
        res = table[res, x_eval[:, t]]
    y_eval = res.astype(jnp.int32)

    # Pad to batches
    total_eval_points = x_eval.shape[0]
    num_full_batches  = total_eval_points // batch_size
    remain            = total_eval_points % batch_size
    if remain > 0:
        pad = batch_size - remain
        x_pad = jnp.zeros((pad, arity), dtype=jnp.int32)
        y_pad = jnp.zeros((pad,), dtype=jnp.int32)
        x_eval = jnp.concatenate([x_eval, x_pad], axis=0)
        y_eval = jnp.concatenate([y_eval, y_pad], axis=0)
        num_eval_batches = num_full_batches + 1
    else:
        num_eval_batches = num_full_batches

    x_eval_batches = x_eval.reshape(num_eval_batches, batch_size, arity)
    y_eval_batches = y_eval.reshape(num_eval_batches, batch_size)
    return x_eval_batches, y_eval_batches

if __name__ == "__main__":
    Ns = [3,4,5,6,8,10,18]  # batch of n
    for n in Ns:
        G = [('rot', i) for i in range(n)] + [('ref', i) for i in range(n)]
        e = ('rot', 0)

        # group basics test
        for a in G:
            for b in G:
                assert mult(a, b, n) in G
        for g in G:
            ig = inv(g, n)
            assert mult(g, ig, n) == e and mult(ig, g, n) == e

        # subgroup verification
        bad = []
        for name, H in enumerate_subgroups_Dn(n):
            if not is_subgroup(H, mult, inv, n):
                bad.append(name)
        if bad:
            print(f"[n={n}] Not subgroups: {bad}")
        else:
            print(f"[n={n}] all enumerated subgroups pass")

    print("Done.")

