import os
import sys
import argparse
import multiprocessing as mp

from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset, DatasetDict
import llvmlite.binding as llvm

CPP_CRASH_LOG = "logs/cpp_crashes.log"
LOG_FILE = "logs/dataset_preparation.log"

def _validation_worker(ir_text: str, ir_bitcode: bytes, max_nodes: int, max_edges: int, max_ir_len: int, queue: mp.Queue):
    os.makedirs("logs", exist_ok=True)
    crash_log_fd = os.open(CPP_CRASH_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    
    os.dup2(crash_log_fd, sys.stdout.fileno())
    os.dup2(crash_log_fd, sys.stderr.fileno())

    logger.remove()
    logger.add(
    LOG_FILE, 
        rotation="50 MB",
        enqueue=True,
        level="DEBUG"
    )

    try:
        ir_text_lines = ir_text.splitlines()
        if len(ir_text_lines) > max_ir_len:
            logger.warning(f"IR length {len(ir_text_lines)} exceeds max {max_ir_len}. Skipping sample.")
            queue.put(False)
            return

        from graph_mlgo.graph.graph import Graph

        graph = Graph(ir_bitcode)
        graph.module.verify()
        
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)

        if num_edges == 0 or num_edges > max_edges:
            queue.put(False)
            logger.warning(f"Number of edges {num_edges} is invalid (0 or exceeds max {max_edges}). Skipping sample.")
            return
            
        if num_nodes == 0 or num_nodes > max_nodes:
            queue.put(False)
            logger.warning(f"Number of nodes {num_nodes} is invalid (0 or exceeds max {max_nodes}). Skipping sample.")
            return

        logger.debug(f"IR has {num_nodes} nodes and {num_edges} edges.")

        edge_iterator = graph.get_inline_order()
        for edge in edge_iterator:
            graph.inline(edge)

        _ = graph.calc_native_size()

        queue.put(True)
        
    except Exception as e:
        logger.exception(f"Validation worker encountered an error: {e}")
        queue.put(False)

def is_valid(ir_text: str, ir_bitcode: bytes, max_nodes: int = 200, max_edges: int = 500, max_ir_len: int = 15000) -> bool:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    
    p = ctx.Process(target=_validation_worker, args=(ir_text, ir_bitcode, max_nodes, max_edges, max_ir_len, queue))
    p.start()
    p.join()
    
    if p.exitcode != 0:
        # logger.warning(f"Worker crashed with exit code {p.exitcode}. Skipping sample.")
        return False
        
    if not queue.empty():
        return queue.get()
        
    return False

def ir_generator(target_gb: float, split_name: str, skip_rows: int, parse_bitcode: bool, apply_filter: bool = True):
    ds = load_dataset("llvm-ml/ComPile", split="train", streaming=True)
    
    if skip_rows > 0:
        ds = ds.skip(skip_rows)

    target_bytes = target_gb * 1024 * 1024 * 1024
    current_bytes = 0

    with tqdm(total=target_gb, desc=f"Processing {split_name}", unit="GB") as bar:
        for sample in ds:
            ir_bitcode = sample["content"]
            ir_text = str(llvm.parse_bitcode(ir_bitcode))

            if apply_filter:
                if not is_valid(ir_text, ir_bitcode):
                    continue

            if parse_bitcode:
                output = ir_text
            else:
                output = ir_bitcode

            sample["content"] = output
            size_bytes = len(ir_text)
            
            yield sample
            
            bar.update(size_bytes / (1024**3))

            current_bytes += size_bytes            
            if current_bytes >= target_bytes:
                break


def prepare_dataset(size_GB: float, output_path: str = "data", parse_bitcode: bool = False, apply_filter: bool = True) -> None:
    logger.info(f"Generating train dataset ({size_GB} GB)...")
    
    train_dataset = Dataset.from_generator(
        ir_generator,
        gen_kwargs={
            "target_gb": float(size_GB),
            "split_name": "train",
            "skip_rows": 0,
            "parse_bitcode": parse_bitcode,
            "apply_filter": apply_filter
        }
    )
    
    skip_rows = len(train_dataset)
    
    test_size_gb = size_GB / 10.0
    logger.info(f"Generating test dataset ({test_size_gb} GB)...")
    
    test_dataset = Dataset.from_generator(
        ir_generator,
        gen_kwargs={
            "target_gb": test_size_gb,
            "split_name": "test",
            "skip_rows": skip_rows,
            "parse_bitcode": parse_bitcode,
            "apply_filter": apply_filter
        }
    )

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    save_path = f"{output_path}/ComPile-{size_GB}GB" + ("-parsed" if parse_bitcode else "")
    dataset_dict.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--size_GB", type=float, default=10, help="Size of the train dataset in GB")
    parser.add_argument("--output_path", type=str, default="data", help="Path to save")
    parser.add_argument("--parse_bitcode", action="store_true", help="Parse bitcode to text IR", default=False)
    parser.add_argument("--filter", action="store_true", help="Run sanity checks", default=True)

    args = parser.parse_args()

    os.remove(LOG_FILE) if os.path.exists(LOG_FILE) else None
    os.remove(CPP_CRASH_LOG) if os.path.exists(CPP_CRASH_LOG) else None
    prepare_dataset(size_GB=args.size_GB, output_path=args.output_path, parse_bitcode=args.parse_bitcode, apply_filter=args.filter)