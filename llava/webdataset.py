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
from typing import Dict, Sequence