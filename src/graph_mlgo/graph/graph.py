from typing import Iterator
from llvmlite import binding as llvm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ctypes

from graph_mlgo.ir import compile_module
from graph_mlgo.graph.embedding import Embedder
from graph_mlgo import cpp_bindings # ty: ignore

FEATURE_DIM = 3

type Edge = tuple[str, str]

class Node:
    features: np.ndarray

    def __init__(self, name: str):
        self.name: str = name
        self.neighbours: set[str] = set()
        self.features: np.ndarray = np.zeros(FEATURE_DIM, dtype=np.float32)

    @staticmethod
    def get_features_dim() -> int:
        return FEATURE_DIM


class Graph:
    nodes: dict[str, Node]
    edges: dict[Edge, int] # multiplicity of edge
    module: llvm.ModuleRef

    def __init__(self, bitcode: bytes):
        self.nodes = {}
        self.edges = dict()

        self.module = llvm.parse_bitcode(bitcode)

        self._build_from_bitcode()
        self._scc()
        self._compute_node_features()

    def calc_native_size(self) -> int:
        return compile_module(str(self.module), enable_inlining=False)[0]

    def get_edge_embedding(self, edge: Edge, embedder: Embedder) -> np.ndarray:
        return embedder.embed(edge, self)

    def get_global_features(self) -> np.ndarray:
        num_nodes = len(self.nodes)
        num_edges = sum(self.edges.values())
        num_unique_edges = len(self.edges)
        feat= np.array([float(num_nodes), float(num_edges), float(num_unique_edges)], dtype=np.float32)

        assert len(feat) == Graph.get_global_embedding_dim()
        return feat

    @staticmethod
    def get_global_embedding_dim() -> int:
        return 3

    def inline(self, edge: Edge) -> None:
        caller, callee = edge

        module_ptr = ctypes.cast(self.module._ptr, ctypes.c_void_p).value
        cpp_bindings.inline_edges(module_ptr, caller, callee)
        
        self.edges.pop(edge, None)
        
        self._refresh_node_neighbours(caller)
        self._update_node_features(caller)
        self._update_node_features(callee)

    def get_inline_order(self) -> Iterator[Edge]:
        visited = set()
        ordered_edges = []
        
        edges_by_callee: dict[str, list[Edge]] = {}
        for edge in self.edges:
            _, callee = edge
            if callee not in edges_by_callee:
                edges_by_callee[callee] = []
            edges_by_callee[callee].append(edge)

        def dfs(u):
            visited.add(u)
            
            for v in self.nodes[u].neighbours:
                if v not in visited:
                    dfs(v)
            
            ordered_edges.extend(edges_by_callee.get(u, []))

        for node_name in list(self.nodes.keys()):
            if node_name not in visited:
                dfs(node_name)
        
        return iter(ordered_edges)

    def _build_from_bitcode(self) -> None:
        for caller_func in self.module.functions:
            caller = caller_func.name
            
            if caller not in self.nodes:
                self.nodes[caller] = Node(caller)

            if caller_func.is_declaration:
                continue

            for block in caller_func.blocks:
                for instr in block.instructions:
                    callee = self._get_instr_callee(instr)
                    if callee:
                        assert isinstance(callee, str), f"Expected callee name as string, got {type(callee)}"

                        if callee not in self.nodes:
                            self.nodes[callee] = Node(callee)
                        
                        edge: Edge = (caller, callee)
                        if edge not in self.edges:
                            self.edges[edge] = 1
                            self.nodes[caller].neighbours.add(callee)
                        else:
                            self.edges[edge] += 1

    def _scc(self) -> None:
        components = self._find_sccs()
        
        node_to_scc_id: dict[str, int] = {}
        for scc_id, component in enumerate(components):
            for node in component:
                node_to_scc_id[node] = scc_id

        filtered_edges: list[Edge] = []

        # For simplicity, inlining is only performed
        # on edges between different SCCs or recursive calls.
        def should_keep_edge(caller: str, callee: str) -> bool:
            caller_scc = node_to_scc_id.get(caller)
            callee_scc = node_to_scc_id.get(callee)
            return caller_scc != callee_scc or caller == callee
        
        for caller, callee in self.edges:
            if should_keep_edge(caller, callee):
                filtered_edges.append((caller, callee))

        self.edges = {edge: self.edges[edge] for edge in filtered_edges}

        for node_name, node in self.nodes.items():
            filtered_neighbours = set()
            for neighbour in node.neighbours:
                if should_keep_edge(node_name, neighbour):
                    filtered_neighbours.add(neighbour)
                    
            node.neighbours = filtered_neighbours

    def _find_sccs(self) -> list[set[str]]:
        index_counter = 0
        time_in: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        sccs: list[set[str]] = []

        def strong_connect(node_name: str) -> None:
            nonlocal index_counter
            time_in[node_name] = index_counter
            lowlinks[node_name] = index_counter
            index_counter += 1
            stack.append(node_name)
            on_stack.add(node_name)

            node = self.nodes[node_name]
            
            for neighbour in node.neighbours:
                if neighbour not in time_in:
                    strong_connect(neighbour)
                    lowlinks[node_name] = min(lowlinks[node_name], lowlinks[neighbour])
                elif neighbour in on_stack:
                    lowlinks[node_name] = min(lowlinks[node_name], time_in[neighbour])

            if lowlinks[node_name] == time_in[node_name]:
                current_scc = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    current_scc.add(w)
                    if w == node_name:
                        break
                sccs.append(current_scc)

        for node_name in self.nodes:
            if node_name not in time_in:
                strong_connect(node_name)

        return sccs

    def _get_instr_callee(self, instr) -> str | None:
        if instr.opcode not in ("call", "invoke"):
            return None
        
        operands = list(instr.operands)
        print(operands)
        if not operands:
            return None
        
        callee_val = operands[-1]
        callee_name = callee_val.name

        assert isinstance(callee_name, str), f"Expected callee name as string, got {type(callee_name)}"

        return callee_name

    def _refresh_node_neighbours(self, name: str) -> None:
        node = self.nodes[name]
        func = self.module.get_function(name)

        node.neighbours.clear()

        for block in func.blocks:
            for instr in block.instructions:
                callee_name = self._get_instr_callee(instr)
                if callee_name is not None:
                    node.neighbours.add(callee_name)

    def _update_node_features(self, name: str) -> None:
        node = self.nodes[name]
        func = self.module.get_function(name)
        
        num_instr = sum(len(list(block.instructions)) for block in func.blocks)
        is_recursive = 1.0 if name in node.neighbours else 0.0
        out_deg = len(node.neighbours)

        node.features[0] = float(num_instr)
        node.features[1] = float(out_deg)
        node.features[2] = is_recursive

    def _compute_node_features(self) -> None:
        for name in self.nodes:
            self._update_node_features(name)

    def visualize(self):
        G = nx.DiGraph()

        for node_name in self.nodes:
            G.add_node(node_name)

        for (caller, callee), multiplicity in self.edges.items():
            G.add_edge(caller, callee, weight=multiplicity)

        pos = nx.spring_layout(G, seed=42, k=5.0)

        plt.figure(figsize=(12, 8))
        
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

        nx.draw_networkx_edges(
            G, pos, 
            arrowstyle="-|>", 
            arrowsize=30,
            edge_color="gray", 
            width=1.5,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=25,
            min_target_margin=25
        )

        edge_labels = {(u, v): f"x{d["weight"]}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

        plt.axis("off")
        plt.show()

def run_test_visualization():
    longer_ir = """
    define internal i32 @square(i32 %x) {
        %res = mul i32 %x, %x
        ret i32 %res
    }

    define internal i32 @transform_data(i32 %val) {
        %1 = call i32 @square(i32 %val)
        %2 = add i32 %1, 10
        ret i32 %2
    }

    define internal void @log_info() {
        ret void
    }

    define i32 @aggregate_results(i32 %a, i32 %b) {
        %1 = call i32 @transform_data(i32 %a)
        %2 = call i32 @transform_data(i32 %b)
        %3 = call i32 @transform_data(i32 42)
        call void @log_info()
        %res = add i32 %1, %2
        %final = add i32 %res, %3
        ret i32 %final
    }

    define i32 @main() {
        call void @log_info()
        %1 = call i32 @aggregate_results(i32 1, i32 2)
        call void @log_info()
        ret i32 %1
    }
    """

    mod = llvm.parse_assembly(longer_ir)
    bitcode = mod.as_bitcode()

    graph = Graph(bitcode)    
    graph.visualize()

    graph.inline(("aggregate_results", "transform_data"))
    graph.visualize()

if __name__ == "__main__":
    run_test_visualization()