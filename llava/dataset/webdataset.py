import os
import json
import copy
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
import webdataset as wds
from typing import Dict, Sequence, Tuple, Union
from .preprocessing import preprocess, preprocess_multimodal, _process_video
from ..constants import IGNORE_INDEX

class WebDataset(Dataset):
    """Dataset for supervised fine-tuning using webdataset."""

    def __init__(self, shard_pattern: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: Dict):
        super(WebDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_shards = count_shards(shard_pattern)
        self.shard_size = data_args.shard_size
        self.est_len = int( self.num_shards * self.shard_size )
        self.dataset = (
            wds.WebDataset(shard_pattern, nodesplitter=wds.split_by_worker)
            .decode("pil")
            .map(self.decode_sample)
        )
        self.iterator = iter(self.dataset)
        self.modals = ["image", "video", "audio"]
        # self.modals = ["video", "audio"] # Placeholder data for video+audio

    def decode_sample(self, sample):

        metadata = sample['json']
        conversations = metadata["conversations"]
        decoded_sample = dict()
        modal_list = list()

        if 'mp4' in sample:
            video = sample["mp4"]
            video = self.process_video(video) # 
            decoded_sample["video"] = video
            modal_list.append('VIDEO')

        if 'jpg' in sample:
            image = sample["jpg"]
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x * 255) for x in self.data_args.image_processor.image_mean))
                image = self.data_args.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = self.data_args.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            decoded_sample["image"] = image
            modal_list.append('IMAGE')

        if 'mp3' in sample:
            audio = sample["mp3"]
            audio = self.process_audio(audio)
            decoded_sample["audio"] = audio
            modal_list.append('AUDIO')

        decoded_sample['conversations'] = conversations
        decoded_sample['modal_list'] = modal_list

        return decoded_sample

    def __len__(self):
        return self.est_len

    def __getitem__(self, idx):
        try:
            sample = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset)
            sample = next(self.iterator)

        ################# placeholder for video+audio ###################
        # sample['modal_list'] = ['VIDEO', 'AUDIO']
        # sample['video'] = torch.stack([sample['image']] * self.data_args.num_frames)
        # sample['audio'] = torch.randn(10000)
        #################################################################

        conversations = preprocess_multimodal(copy.deepcopy([sample["conversations"]]), self.data_args)
        data_dict = preprocess(conversations, self.tokenizer, sample['modal_list']) # TODO
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0],
                        instructions=data_dict['instructions'][0])
        for modal in self.modals:
            if modal in sample:
                data_dict[modal] = sample[modal]
            
        return data_dict

    def process_video(self, video):

        return _process_video(video, self.data_args.image_processor, num_frames=self.data_args.num_frames)

    def preprocess_audio(file_path, target_sr=16000):
    
        # y, sr = librosa.load(file_path, sr=None)
        
        # if sr != target_sr:
        #     y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # if len(y.shape) > 1:
        #     y = librosa.to_mono(y)
        
        # y = y / np.max(np.abs(y))

        # audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        
        audio_tensor = torch.randn(10000)

        return audio_tensor

@dataclass
class DataCollatorForWebDataset(object):
    """Collate examples for supervised fine-tuning using webdataset."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for modal in ['image', 'video', 'audio']:
            if modal in instances[0]:
                modal_data = [instance[modal] for instance in instances]
                if all(x is not None and x.shape == modal_data[0].shape for x in modal_data):
                    batch[f"{modal}s"] = torch.stack(modal_data)
                else:
                    batch[f"{modal}s"] = modal_data

        batch['instructions'] = [instance['instructions'] for instance in instances]

        return batch

def count_shards(shard_pattern: str) -> int:
    shard_files = glob.glob(shard_pattern.replace("{00000..00558}", "*"))
    return len(shard_files)