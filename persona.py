from .LMDER import LMEDRModel
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = LMEDRModel.from_pretrained("facebook/bart-large", num_labels=1,
                                           num_token=len(tokenizer),
                                           num_latent=10, num_latent2=10)