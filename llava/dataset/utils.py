import glob
import transformers
from .webdataset import WebDataset, DataCollatorForWebDataset

def count_shards(shard_pattern: str) -> int:
    shard_files = glob.glob(shard_pattern.replace("{00000..00558}", "*"))
    return len(shard_files)

def make_web_data_module(tokenizer: transformers.PreTrainedTokenizer,
                         data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning using webdataset."""
    train_dataset = WebDataset(shard_pattern=data_args.shard_pattern,
                               tokenizer=tokenizer,
                               data_args=data_args)
    data_collator = DataCollatorForWebDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)