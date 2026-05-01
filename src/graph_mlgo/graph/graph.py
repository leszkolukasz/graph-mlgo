from llvmlite import binding as llvm

from graph_mlgo.ir import compile_module

type Edge = tuple[str, str]

class Node:
    def __init__(self, name: str):
        self.name: str = name
        self.neighbours: set[str] = set()


class Graph:
    nodes: dict[str, Node]
    edges: set[Edge]
    module: llvm.ModuleRef

    def __init__(self, bitcode: bytes):
        self.nodes = {}
        self.edges = set()

        self.module = llvm.parse_bitcode(bitcode)

        self._build_from_bitcode()
        self._scc()

    def calc_native_size(self) -> int:
        return compile_module(str(self.module), enable_inlining=False)[0]

    def _build_from_bitcode(self) -> None:
        for caller_func in self.module.functions:
            caller = caller_func.name
            
            if caller not in self.nodes:
                self.nodes[caller] = Node(caller)

            if caller_func.is_declaration:
                continue

            for block in caller_func.blocks:
                for instr in block.instructions:
                    if instr.opcode in ("call", "invoke"):
                        operands = list(instr.operands)
                        if not operands:
                            continue
                            
                        callee_val = operands[-1]
                        print(operands)
                        callee = callee_val.name

                        if callee:
                            assert isinstance(callee, str), f"Expected callee name as string, got {type(callee)}"

                            if callee not in self.nodes:
                                self.nodes[callee] = Node(callee)
                            
                            edge: Edge = (caller, callee)
                            if edge not in self.edges:
                                self.edges.add(edge)
                                self.nodes[caller].neighbours.add(callee)

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

        self.edges = set(filtered_edges)

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