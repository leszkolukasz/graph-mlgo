from collections import defaultdict
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.typing import VariableDict

from graph_mlgo.constants import MAX_NODES
from graph_mlgo.graph.embedding.constants import GLOBAL_FEATURES_DIM

if TYPE_CHECKING:
    from graph_mlgo.graph import Graph
    from graph_mlgo.graph.embedding.networks import EmbeddingNet


class EdgeType:
    FORWARD = 0
    BACKWARD = 1
    SELF_LOOP = 2
    PAD = 3


@struct.dataclass
class EmbeddingAux:
    h: jnp.ndarray
    indices: list[jnp.ndarray]
    edge_types: list[jnp.ndarray]

    def to_device(self, device: jax.Device) -> "EmbeddingAux":
        return EmbeddingAux(
            h=jax.device_put(self.h, device),
            indices=[jax.device_put(idx, device) for idx in self.indices],
            edge_types=[jax.device_put(et, device) for et in self.edge_types],
        )

    def to_cpu(self) -> "EmbeddingAux":
        return self.to_device(jax.devices("cpu")[0])

    def to_gpu(self) -> "EmbeddingAux":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices available for EmbeddingAux")
        return self.to_device(gpus[0])


@struct.dataclass
class EmbeddingParts:
    global_feat: jnp.ndarray
    edge_embed: jnp.ndarray
    edge_mult: jnp.ndarray
    const_ratio: jnp.ndarray
    aux: EmbeddingAux | None = None

    def to_device(self, device: jax.Device) -> "EmbeddingParts":
        return EmbeddingParts(
            global_feat=jax.device_put(self.global_feat, device),
            edge_embed=jax.device_put(self.edge_embed, device),
            edge_mult=jax.device_put(self.edge_mult, device),
            const_ratio=jax.device_put(self.const_ratio, device),
            aux=self.aux.to_device(device) if self.aux is not None else None,
        )

    def to_cpu(self) -> "EmbeddingParts":
        return self.to_device(jax.devices("cpu")[0])

    def to_gpu(self) -> "EmbeddingParts":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices available for EmbeddingParts")
        return self.to_device(gpus[0])

    @classmethod
    def empty(cls, embed_dim: int) -> "EmbeddingParts":
        return cls(
            global_feat=jnp.zeros(GLOBAL_FEATURES_DIM, dtype=jnp.float32),
            edge_embed=jnp.zeros(embed_dim, dtype=jnp.float32),
            edge_mult=jnp.zeros(1, dtype=jnp.float32),
            const_ratio=jnp.zeros(1, dtype=jnp.float32),
            aux=None,
        )


def contrastive_loss(
    params: VariableDict,
    model: "EmbeddingNet",
    h_in: jnp.ndarray,
    neighbor_indices: list[jnp.ndarray],
    edge_types: list[jnp.ndarray],
    u_idx: jnp.ndarray,
    v_idx: jnp.ndarray,
    neg_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    z = model.apply(params, h_in, neighbor_indices, edge_types)

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


# Returns (neighborhood_size, node_feat_dim), list of (depth, num_targets, num_neighbours), list of (depth, num_targets, num_neighbours)
def extract_neighborhood(
    *,
    graph: "Graph",
    batch: list[str],
    depth: int,
    num_neighbours: int,
    use_in_edges: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    neighbourhood_per_depth: list[defaultdict[str, list[str]]] = [
        defaultdict(list) for _ in range(depth)
    ]
    edge_types_per_depth: list[defaultdict[str, list[int]]] = [
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
                neighbours, edge_types = sample_neighbors(
                    graph=graph,
                    node=node,
                    num_neighbours=num_neighbours,
                    use_in_edges=use_in_edges,
                )
                neighbourhood_per_depth[d][node].extend(neighbours)
                edge_types_per_depth[d][node].extend(edge_types)

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
    neighbor_edge_types = []

    for d in range(depth):
        target_nodes = targets_per_depth[d]
        layer_indices = np.zeros((len(target_nodes), num_neighbours), dtype=np.int32)
        layer_edge_types = np.zeros((len(target_nodes), num_neighbours), dtype=np.int32)

        for i, node in enumerate(target_nodes):
            sampled_neighs = neighbourhood_per_depth[d][node]
            sampled_types = edge_types_per_depth[d][node]

            layer_indices[i] = [node_to_idx[n] for n in sampled_neighs]
            layer_edge_types[i] = sampled_types

        neighbor_indices.append(layer_indices)
        neighbor_edge_types.append(layer_edge_types)

    return features, neighbor_indices, neighbor_edge_types


def pad_neighborhood(
    feat_np: np.ndarray,
    indices_np: list[np.ndarray],
    edge_types_np: list[np.ndarray],
    max_nodes: int = MAX_NODES,
    pad_id: int = -1,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
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
    edge_types_padded = []

    for i in range(len(indices_np)):
        ind = indices_np[i]
        num_targets = ind.shape[0]
        pad_targets = max_nodes - num_targets

        ind_padded = np.pad(
            ind,
            ((0, pad_targets), (0, 0)),
            mode="constant",
            constant_values=pad_id,
        )
        indices_padded.append(ind_padded)

        et = edge_types_np[i]
        et_padded = np.pad(
            et,
            ((0, pad_targets), (0, 0)),
            mode="constant",
            constant_values=EdgeType.PAD,
        )

        edge_types_padded.append(et_padded)

    return feat_padded, indices_padded, edge_types_padded


def sample_neighbors(
    *, graph: "Graph", node: str, num_neighbours: int, use_in_edges: bool = False
) -> tuple[list[str], list[int]]:
    forward_neighbors = list(graph.nodes[node].neighbours)

    if use_in_edges:
        backward_edges = list(graph.edges_by_callee.get(node, set()))
        backward_neighbors = [caller for caller, _ in backward_edges]
    else:
        backward_edges = []
        backward_neighbors = []

    all_neighbors = forward_neighbors + backward_neighbors
    edge_types = [EdgeType.FORWARD] * len(forward_neighbors) + [
        EdgeType.BACKWARD
    ] * len(backward_neighbors)

    for idx, (src, dest) in enumerate(backward_edges):
        if src == dest:
            edge_types[len(forward_neighbors) + idx] = EdgeType.SELF_LOOP

    if len(all_neighbors) == 0:
        return [node] * num_neighbours, [EdgeType.PAD] * num_neighbours

    replace = len(all_neighbors) < num_neighbours

    indices = np.random.choice(len(all_neighbors), size=num_neighbours, replace=replace)

    sampled_nodes = [all_neighbors[i] for i in indices]
    sampled_types = [edge_types[i] for i in indices]

    return sampled_nodes, sampled_types


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


def concatenate_parts(parts: EmbeddingParts) -> jnp.ndarray:
    return jnp.concatenate(
        [parts.global_feat, parts.edge_embed, parts.edge_mult, parts.const_ratio]
    )
