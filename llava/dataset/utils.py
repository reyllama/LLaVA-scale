import glob
import transformers
from .webdataset import WebDataset, DataCollatorForWebDataset

def make_web_data_module(tokenizer: transformers.PreTrainedTokenizer,
                         data_args):
    """Make dataset and collator for supervised fine-tuning using webdataset."""
    train_dataset = WebDataset(shard_pattern=data_args.shard_pattern,
                               tokenizer=tokenizer,
                               data_args=data_args)
    data_collator = DataCollatorForWebDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)