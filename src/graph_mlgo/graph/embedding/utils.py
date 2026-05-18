from collections import defaultdict
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import VariableDict

from graph_mlgo.graph.embedding import GraphSAGENet

if TYPE_CHECKING:
    from graph_mlgo.graph import Graph


def graphsage_loss(
    params: VariableDict,
    model: GraphSAGENet,
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
    neighbourhood_per_depth: list[defaultdict[str, set]] = [
        defaultdict(set) for _ in range(depth)
    ]
    targets_per_depth: list[list[str]] = [[] for _ in range(depth)]
    targets_per_depth[depth - 1] = batch
    all_nodes: list[str] = []

    for d in reversed(range(depth)):
        current_targets = targets_per_depth[d]

        for node in current_targets:
            neighbours = sample_neighbors(
                graph=graph, node=node, num_neighbours=num_neighbours
            )
            neighbourhood_per_depth[d][node].update(neighbours)

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

    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    feat_dim = graph.nodes[all_nodes[0]].features.shape[0]
    features = np.zeros((len(all_nodes), feat_dim), dtype=np.float32)
    for node in all_nodes:
        idx = node_to_idx[node]
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


def sample_neighbors(*, graph: "Graph", node: str, num_neighbours: int) -> list[str]:
    direct_neighbors = list(graph.nodes[node].neighbours)

    if len(direct_neighbors) == 0:
        return [node] * num_neighbours

    replace = len(direct_neighbors) < num_neighbours
    sampled = np.random.choice(direct_neighbors, size=num_neighbours, replace=replace)

    return sampled.tolist()


def sample_training_batches(
    graph: Graph, num_batches: int, batch_size: int, num_negatives: int
) -> list[tuple[list[str], list[str], list[list[str]]]]:
    all_nodes = list(graph.nodes.keys())
    nodes_with_neighbours = [
        node for node in all_nodes if len(graph.nodes[node].neighbours) > 0
    ]

    adj_list = {node: [] for node in all_nodes}
    for src, dst in graph.edges.keys():
        adj_list[src].append(dst)

    batches = []
    for _ in range(num_batches):
        batch_u = []
        batch_v = []
        batch_neg = []

        starts = np.random.choice(nodes_with_neighbours, size=batch_size, replace=True)

        for u in starts:
            neighbors = adj_list[u]
            v = np.random.choice(neighbors)

            negatives = np.random.choice(
                a=all_nodes, size=num_negatives, replace=True
            ).tolist()

            batch_u.append(u)
            batch_v.append(v)
            batch_neg.append(negatives)

        batches.append((batch_u, batch_v, batch_neg))

    return batches
