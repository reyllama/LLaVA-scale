import json
import os
import io
import tarfile
from glob import glob
import webdataset as wds

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

def create_sharded_tars_video(json_file, image_dir, video_dir, audio_dir, output_dir, shard_size=1000):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    for i in range(0, len(data), shard_size):
        shard_data = data[i:i + shard_size]
        tar_path = os.path.join(output_dir, f"shard-{shard_idx:05d}.tar")
        with tarfile.open(tar_path, 'w') as tar:
            for item in shard_data:
                                
                video_path = os.path.join(video_dir, item['video'])
                with open(video_path, 'rb') as vid:
                    vid_data = vid.read()
                tarinfo = tarfile.TarInfo(name=f"{item['id']}.mp4")
                tarinfo.size = len(vid_data)
                tar.addfile(tarinfo, io.BytesIO(vid_data))
                
                audio_path = os.path.join(audio_dir, item['audio'])
                with open(audio_path, 'rb') as aud:
                    aud_data = aud.read()
                tarinfo = tarfile.TarInfo(name=f"{item['id']}.mp3")
                tarinfo.size = len(aud_data)
                tar.addfile(tarinfo, io.BytesIO(aud_data))
                
                # Add metadata
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


if __name__ == "__main__":


    json_file = './data/blip_laion_cc_sbu_558k.json'
    image_dir = './data/images'
    output_dir = './data/shards'
    create_sharded_tars_image(json_file, image_dir, output_dir, shard_size=1000)
