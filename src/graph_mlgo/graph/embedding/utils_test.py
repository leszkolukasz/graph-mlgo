from llvmlite import binding as llvm

from graph_mlgo.graph import Graph
from graph_mlgo.graph.embedding.utils import extract_neighborhood

if __name__ == "__main__":
    simple_ir = """
    define internal i32 @square(i32 %x) {
        ret i32 %x
    }

    define internal i32 @transform_data(i32 %val) {
        %1 = call i32 @square(i32 %val)
        ret i32 %1
    }

    define i32 @main() {
        %1 = call i32 @transform_data(i32 1)
        %2 = call i32 @square(i32 2)
        ret i32 %1
    }
    """

    mod = llvm.parse_assembly(simple_ir)
    bitcode = mod.as_bitcode()

    graph = Graph(bitcode)

    print("Graph nodes:", list(graph.nodes.keys()))

    batch = ["main", "transform_data", "main"]
    print(f"\nInput batch: {batch}")

    depth = 2
    num_neighbours = 3

    features, neighbor_indices, edge_types = extract_neighborhood(
        graph=graph,
        batch=batch,
        depth=depth,
        num_neighbours=num_neighbours,
        use_in_edges=True,
    )

    print("\n=== EXTRACT_NEIGHBORHOOD ===")
    print(
        f"Shape features (h): {features.shape} -> (Number of all necessary nodes, Feature dimension)"
    )

    for d, indices in enumerate(neighbor_indices):
        print(f"\nLayer {d} (indices):")
        print(
            f" - Shape: {indices.shape} -> (Number of targets in the layer, num_neighbours)"
        )
        print(f" - Index matrix:\n{indices}")

    print(f"\nEdge types:\n{edge_types}")
