# LLaVA-scale

`Codebase to scale-up LLaVA.`

## 🎯 Potential Issues for Scale-up

### \# 1-1. Data Scale-up

> The current data pipeline assumes all images (videos) and their corresponding metadata to be stored locally. However, this will become easily infeasible when the dataset is scaled. Plus, there are more elegant ways to implement *lazy dataloading* without explicitly defining custom dataloading pipeline as in the original code base.

### \# 1-2. Model Scale-up

> LLaVA is essentially made up of three architectural components: vision encoder, projector and the language model. In most (if not all) model variants, the language model with a huge number of parameters (7B~) takes up most of the compute resources. Therefore, to control compute resources with model scale-up, we need to pay careful attention to the number of tokens being computed by the language model. Current implementation takes little care of this, hence with higher spatial/temporal resolution input, the compute for the language model will become very expensive.

### \# 1-3. Team Scale-up

> The code itself is overall well-written, but when more people are to contribute to this codebase, we need to make some parts more modular, especially the `train.py`. This file alone contains code for argument parsing, model configuration, dataset classs definition, data preprocessing, and the actual training, which could be refactored to enhance readability and reusability.

## 🎯 Proposed Modifications

### \# 2-1. Webdataset

[Webdataset](https://webdataset.github.io/webdataset/) is a pytorch implementation of [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) that provides efficient access to dataset stored in `.tar` archives in streaming fashion. It has advantages over classical map-based dataset in terms of scalability as it naturally supports `streaming` and `sharding`, relieving potential memory/communication bottlenecks.

Webdataset class is implemented in [webdataset.py](). Thanks to the flexible mapping supported by webdatasets, we can take care of the basic preprocessing of the audio-visual input using the `decode_sample()` method.

```python
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
                    ...
                    return square_img
        
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
```

Because we already have the 558k dataset locally, we convert them to `.tar` archive format to use webdataset pipeline. This code is provided in [convert_data_to_wds.py]().

```python
def create_sharded_tars_image(json_file, image_dir, output_dir, shard_size=1000):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    for i in range(0, len(data), shard_size):
        shard_data = data[i:i + shard_size]
        tar_path = os.path.join(output_dir, f"shard-{shard_idx:05d}.tar")
        with tarfile.open(tar_path, 'w') as tar:
            for item in shard_data:
                image_path = os.path.join(image_dir, item['image'])
                with open(image_path, 'rb') as img:
                    img_data = img.read()
                img_name = os.path.basename(image_path)
                tarinfo = tarfile.TarInfo(name=f"{item['id']}.jpg")
                tarinfo.size = len(img_data)
                tar.addfile(tarinfo, io.BytesIO(img_data))
                meta_data = {
                    'id': item['id'],
                    'conversations': item['conversations']
                }
                meta_str = json.dumps(meta_data)
                tarinfo = tarfile.TarInfo(name=f"{item['id']}.json")
                tarinfo.size = len(meta_str)
                tar.addfile(tarinfo, io.BytesIO(meta_str.encode('utf-8')))
        shard_idx += 1
        if shard_idx % 10 == 0:
            print(f"{shard_idx} / {len(data) // shard_size} completed.")
```

### \# 2-2. Local Token Merging

[Token Merging](https://arxiv.org/abs/2210.09461) was originally proposed for classifier ViTs to improve the model throughput. However, as visual instruction following requires a much more fine-grained understanding of the visual content compared to fixed-label classification, we take inspirations from [Token Merging for Stable Diffusions](https://arxiv.org/abs/2303.17604) and make some modifications.

- ToMe-SD (Token Merging for SD) introduces a window-based anchor selection method (window size of 2x2) to adapt token merging to the dense prediction task (image generation).
- We modify ToMe-SD to preserve relative order of the visual tokens. Although the original ToMe-SD did not care so much about the sequence order (because older SDs do not embed position information anyways), for our task, the relative position order of visual tokens is crucial.
- 

### \# 2-3. Instruction-aware Spatio-temporal Resampling

## 🎯 Video + Audio Utilization

### \# 3-1. Video

### \# 3-2. Audio