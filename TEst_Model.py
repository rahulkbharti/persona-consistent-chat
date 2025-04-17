import torch

print(torch.__version__)

# from build_data_PersonaChat import create_data, build_dataloader, build_infer_dataset

from transformers import BartTokenizer
# from model.modeling_bart import LMEDRModel
from model_paper.LMEDR import LMEDRModel 
data_from = "_original"

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
# add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}
# num_added_toks = tokenizer.add_special_tokens(add_special_tokens)

# print('We have added {} tokens'.format(num_added_toks))
# model = LMEDRModel.from_pretrained("facebook/bart-large", num_labels=1,
#                                            num_token=len(tokenizer),
#                                            num_latent=10, num_latent2=10)
# model.resize_token_embeddings(len(tokenizer))
# model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids('<response>')
# model.config.forced_bos_token_id = None


# # persona, query, response, cand = create_data(f"data/ConvAI2/valid_self{data_from}.txt")
# # train_data = build_dataloader(persona, query, response, cand, tokenizer, max_history=4, n_cand=5)

# # train_data

# print(model)