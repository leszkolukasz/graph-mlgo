from datasets import Dataset, DatasetDict, load_from_disk

class ComPileDataset(DatasetDict):
    def __init__(self, path: str):
        loaded_dict = load_from_disk(path)
        
        super().__init__(loaded_dict)
        
    @property
    def train(self) -> Dataset:
        return self["train"]
        
    @property
    def test(self) -> Dataset:
        return self["test"]

if __name__ == "__main__":
    dataset = ComPileDataset("data/ComPile-1GB")
    
    with open("sample_ir.txt", "w") as f:
        for i, sample in enumerate(dataset.train):
            f.write(f"--- Train Sample {i} ---\n")
            f.write(sample["content"])
            f.write("\n\n")
            
            if i >= 4:
                break

        for i, sample in enumerate(dataset.test):
            f.write(f"--- Test Sample {i} ---\n")
            f.write(sample["content"])
            f.write("\n\n")
            
            if i >= 4:
                break