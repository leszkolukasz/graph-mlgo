from graph_mlgo.constants import MAX_NODES
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import VariableDict

if TYPE_CHECKING:
    from graph_mlgo.graph import Graph
    from graph_mlgo.graph.embedding import GraphSAGENet


def graphsage_loss(
    params: VariableDict,
    model: "GraphSAGENet",
    h_in: jnp.ndarray,
    neighbor_indices: list[jnp.ndarray],
    u_idx: jnp.ndarray,
    v_idx: jnp.ndarray,
    neg_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    z = model.apply(params, h_in, neighbor_indices)

    z_u = cast(jnp.ndarray, z[u_idx])  # (Batch, dim)
    z_v = cast(jnp.ndarray, z[v_idx])  # (Batch, dim)
    z_neg = cast(jnp.ndarray, z[neg_idx])  # (Batch, Q, dim)

    pos_score = jnp.sum(z_u * z_v, axis=-1)
    pos_loss = -jax.nn.log_sigmoid(pos_score)  # (Batch,)

    z_u_expanded = jnp.expand_dims(z_u, axis=1)  # (Batch, 1, dim)
    neg_score = jnp.sum(z_u_expanded * z_neg, axis=-1)  # (Batch, Q)
    neg_loss = -jax.nn.log_sigmoid(-neg_score)

    total_loss = jnp.mean(pos_loss + jnp.sum(neg_loss, axis=1))

    metrics = {
        "loss": total_loss,
        "pos_loss": jnp.mean(pos_loss),
        "neg_loss": jnp.mean(neg_loss),
    }

    return total_loss, metrics


# Returns (neighborhood_size, node_feat_dim), list of (depth, num_targets, num_neighbours)
def extract_neighborhood(
    *, graph: "Graph", batch: list[str], depth: int, num_neighbours: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    neighbourhood_per_depth: list[defaultdict[str, list[str]]] = [
        defaultdict(list) for _ in range(depth)
    ]
    targets_per_depth: list[list[str]] = [[] for _ in range(depth)]
    targets_per_depth[depth - 1] = batch
    all_nodes: list[str] = []

    for d in reversed(range(depth)):
        current_targets = targets_per_depth[d]

        for node in current_targets:
            # To handle case when batch has duplicates
            if node not in neighbourhood_per_depth[d]:
                neighbours = sample_neighbors(
                    graph=graph, node=node, num_neighbours=num_neighbours
                )
                neighbourhood_per_depth[d][node].extend(neighbours)

        next_targets = list(current_targets)
        next_set = set(current_targets)

        for neighs in neighbourhood_per_depth[d].values():
            for n in sorted(list(neighs)):
                if n not in next_set:
                    next_targets.append(n)
                    next_set.add(n)
        if d > 0:
            targets_per_depth[d - 1] = next_targets
        else:
            all_nodes = next_targets

    node_to_idx = {}
    for idx, node in enumerate(all_nodes):
        if node not in node_to_idx:
            node_to_idx[node] = idx

    feat_dim = graph.nodes[all_nodes[0]].features.shape[0]
    features = np.zeros((len(all_nodes), feat_dim), dtype=np.float32)

    # First few elements in all_nodes may contain duplicates from `batch`
    for idx, node in enumerate(all_nodes):
        features[idx] = graph.nodes[node].features

    neighbor_indices = []

    for d in range(depth):
        target_nodes = targets_per_depth[d]
        layer_indices = np.zeros((len(target_nodes), num_neighbours), dtype=np.int32)

        for i, node in enumerate(target_nodes):
            sampled_neighs = sorted(list(neighbourhood_per_depth[d][node]))
            layer_indices[i] = [node_to_idx[n] for n in sampled_neighs]

        neighbor_indices.append(layer_indices)

    return features, neighbor_indices

def pad_neighborhood(
    feat_np: np.ndarray,
    indices_np: list[np.ndarray],
    max_nodes: int = MAX_NODES,
    pad_id: int = -1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    current_num_nodes = feat_np.shape[0]

    if current_num_nodes > max_nodes:
        raise ValueError(
            f"Subgraph exceeds MAX_NODES: {current_num_nodes} > {max_nodes}"
        )

    pad_size = max_nodes - current_num_nodes

    feat_padded = np.pad(
        feat_np,
        ((0, pad_size), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )

    indices_padded = []
    for ind in indices_np:
        num_targets = ind.shape[0]
        pad_targets = max_nodes - num_targets

        ind_padded = np.pad(
            ind,
            ((0, pad_targets), (0, 0)),
            mode="constant",
            constant_values=pad_id,
        )
        indices_padded.append(ind_padded)

    return feat_padded, indices_padded


def sample_neighbors(*, graph: "Graph", node: str, num_neighbours: int) -> list[str]:
    direct_neighbors = list(graph.nodes[node].neighbours)

    if len(direct_neighbors) == 0:
        return [node] * num_neighbours

    replace = len(direct_neighbors) < num_neighbours
    sampled = np.random.choice(direct_neighbors, size=num_neighbours, replace=replace)

    return sampled.tolist()


def sample_training_batches(
    graph: "Graph", num_batches: int, batch_size: int, num_negatives: int
) -> list[tuple[list[str], list[str], list[list[str]]]]:
    all_nodes = [node for node, data in graph.nodes.items() if np.any(data.features)]
    num_nodes = len(all_nodes)

    if num_nodes < 2 + num_negatives:
        return []

    adj_list = {node: [] for node in all_nodes}
    for src, dst in graph.edges.keys():
        if src != dst and src in adj_list and dst in adj_list:
            adj_list[src].append(dst)

    valid_starts = [node for node in all_nodes if len(adj_list[node]) > 0]

    if not valid_starts:
        return []

    batches = []
    for _ in range(num_batches):
        batch_u = []
        batch_v = []
        batch_neg = []

        starts = np.random.choice(valid_starts, size=batch_size, replace=True)

        for u in starts:
            v = np.random.choice(adj_list[u])

            negatives = set()
            while len(negatives) < num_negatives:
                cand_idx = np.random.randint(0, num_nodes)
                cand = all_nodes[cand_idx]

                if cand != u and cand != v:
                    negatives.add(cand)

            batch_u.append(u)
            batch_v.append(v)
            batch_neg.append(list(negatives))

        batches.append((batch_u, batch_v, batch_neg))

    return batches
