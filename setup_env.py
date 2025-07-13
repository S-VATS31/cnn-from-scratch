import os
import logging

# Set up CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")
