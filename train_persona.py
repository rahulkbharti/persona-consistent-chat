import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define special tokens
add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
num_added_toks = tokenizer.add_special_tokens(add_special_tokens)
logger.info(f'Added {num_added_toks} new tokens')

# Define cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/models")

# Load model from cache (or download if not available)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", cache_dir=cache_dir)

# Resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# Set decoder start token
response_token_id = tokenizer.convert_tokens_to_ids('<response>')
if response_token_id is None:
    raise ValueError("'<response>' token was not added properly.")

model.config.decoder_start_token_id = response_token_id

# Ensure model runs on CPU
device = torch.device("cpu")
model.to(device)

logger.info("Model loaded and cached at: {}".format(cache_dir))
logger.info("Model is ready on CPU")
