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

The training logs for base (original) and webdataset pipeline are shown below.

![webdataset](./assets/wds.jpg)

### \# 2-2. Local Token Merging

[Token Merging](https://arxiv.org/abs/2210.09461) was originally proposed for classifier ViTs to improve the model throughput. However, as visual instruction following requires a much more fine-grained understanding of the visual content compared to fixed-label classification, we take inspirations from [Token Merging for Stable Diffusions](https://arxiv.org/abs/2303.17604) and make some modifications.

- ToMe-SD (Token Merging for SD) introduces a window-based anchor selection method (window size of 2x2) to adapt token merging to the dense prediction task (image generation).
- We modify ToMe-SD to preserve relative order of the visual tokens. Although the original ToMe-SD did not care so much about the sequence order (because older SDs do not embed position information anyways), for our task, the relative position order of visual tokens is crucial.
- We introduce cosine scheduling for the token merging ratio `r` (proportion of tokens to merge). We observe training instability otherwise.

```python
class LocalTokenMerging:
    def __init__(self, w: int, h: int, sx: int = 2, sy: int = 2, r: float = 0.5):
        self.w, self.h = w, h
        self.sx, self.sy = sx, sy
        self.r = 0.0
        self.max_r = r
        self.tik = 0.0

        if torch.distributed.get_rank() == 0:
            print(f"[ToMe] Initializing Local Token Merging with w: {w}, h: {h}, sx: {sx}, sy: {sy}, r: {r}")

    def adjust_r(self):
        self.r = self.max_r * (1 - math.cos((self.tik / 4000) * math.pi / 2)) if self.tik < 4000 else self.max_r
        
    def __call__(self, x: torch.Tensor):
        B, N, C = x.shape
        r = int(self.r * N)

        self.tik += 1
        self.adjust_r()

        if self.tik % 100 == 0:
            print(f"** ToMe info: self.r: {self.r}, self.max_r: {self.max_r}, self.tik: {self.tik}, N: {N}, r: {r}")

        if r <= 0:
            return x

        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

        with torch.no_grad():
            hsy, wsx = self.h // self.sy, self.w // self.sx

            rand_idx = torch.randint(self.sy*self.sx, size=(hsy, wsx, 1)).to(x.device)
            
            idx_buffer_view = torch.zeros(hsy, wsx, self.sy*self.sx, device=x.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
            idx_buffer_view = idx_buffer_view.view(hsy, wsx, self.sy, self.sx).transpose(1, 2).reshape(hsy * self.sy, wsx * self.sx)

            if (hsy * self.sy) < self.h or (wsx * self.sx) < self.w:
                idx_buffer = torch.zeros(self.h, self.w, device=x.device, dtype=torch.int64)
                idx_buffer[:(hsy * self.sy), :(wsx * self.sx)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

            del idx_buffer, idx_buffer_view

            num_dst = hsy * wsx
            a_idx = rand_idx[:, num_dst:, :]
            b_idx = rand_idx[:, :num_dst, :]

            def split(x):
                C = x.shape[-1]
                src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

            metric = x / x.norm(dim=-1, keepdim=True)
            a, b = split(metric)
            scores = a @ b.transpose(-1, -2)

            r = min(a.shape[1], r)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            src_idx = edge_idx[..., :r, :]

            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            for b in range(B):
                mask[b, src_idx[b, :, 0]] = False
            out = x[mask].view(B, N - r, -1)

        return out
```

The training logs for different Token Merging methods are shown below. (Green: base, Red: ToMe (r=0.25), Teal: ToMe (r=0.5), Yellow: ToMe w/o scheduling)

![ToMe](./assets/tome.jpg)

### \# 2-3. Instruction-aware Spatio-temporal Resampling

When we deal with high-resolution data (both spatial and temporal), naive prepending approach of LLaVA results in excess number of visual tokens fed to the language model. To overcome this limitation, several works have proposed resampling techniques to handle language modeling on (long) videos. [BLIP2](https://arxiv.org/abs/2301.12597) and [InstructBLIP](https://arxiv.org/abs/2305.06500), for instance, introduce Q-former to extract instruction-aware visual information from the input. However, [a recent work](https://arxiv.org/abs/2406.07476) has pointed out that its naive resampling hurts video understanding capability by ignoring the spatio-temporal order of audio-visual tokens. 

## 🎯 Video + Audio Utilization

### \# 3-1. Video

### \# 3-2. Audio