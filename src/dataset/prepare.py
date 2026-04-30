import argparse
import llvmlite.binding as llvm
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from loguru import logger

def ir_generator(target_gb: float, split_name: str, skip_rows: int, parse_bitcode: bool):
    ds = load_dataset("llvm-ml/ComPile", split="train", streaming=True)
    
    if skip_rows > 0:
        ds = ds.skip(skip_rows)

    target_bytes = target_gb * 1024 * 1024 * 1024
    current_bytes = 0

    with tqdm(total=target_gb, desc=f"Processing {split_name}", unit="GB") as bar:
        for sample in ds:
            bitcode = sample["content"]

            if parse_bitcode:
                try:
                    mod = llvm.parse_bitcode(bitcode)
                    output = str(mod)
                except Exception as e:
                    logger.warning(f"Failed to parse bitcode: {e}")
                    continue
            else:
                output = bitcode.decode("utf-8", errors="ignore")

            sample["content"] = output
            size_bytes = len(output.encode("utf-8"))
            
            yield sample
            
            bar.update(size_bytes / (1024**3))

            current_bytes += size_bytes            
            if current_bytes >= target_bytes:
                break


def prepare_dataset(size_GB: int, output_path: str = "data", parse_bitcode: bool = False):
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    logger.info(f"Generating train dataset ({size_GB} GB)...")
    
    train_dataset = Dataset.from_generator(
        ir_generator,
        gen_kwargs={
            "target_gb": float(size_GB),
            "split_name": "train",
            "skip_rows": 0,
            "parse_bitcode": parse_bitcode
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
            "parse_bitcode": parse_bitcode
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
    parser.add_argument("--size_GB", type=int, default=10, help="Size of the train dataset in GB")
    parser.add_argument("--output_path", type=str, default="data", help="Path to save")
    parser.add_argument("--parse_bitcode", action="store_true", help="Parse bitcode to text IR")
    args = parser.parse_args()

    prepare_dataset(size_GB=args.size_GB, output_path=args.output_path, parse_bitcode=args.parse_bitcode)