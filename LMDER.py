import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartPretrainedModel, BartModel, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss

class LMEDRModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, num_token=None, num_latent=10, num_latent2=10):
        super().__init__(config)
        self.model = BartModel(config)
        self.num_latent = num_latent
        self.num_latent2 = num_latent2
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        self.bow_head = nn.Linear(config.d_model, num_token if num_token else config.vocab_size)
        
        self.classification_head = nn.Linear(config.d_model, config.num_labels)
        self.latent_head_m1 = nn.Linear(config.d_model, num_latent)
        self.latent_head_m2 = nn.Linear(config.d_model, num_latent2)
        
        self.memory1 = nn.Parameter(torch.randn(self.num_latent, config.d_model))
        self.memory2 = nn.Parameter(torch.randn(self.num_latent2, config.d_model))

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        extra_bias = torch.zeros((1, max(0, new_num_tokens - old_num_tokens)), device=self.final_logits_bias.device)
        self.register_buffer("final_logits_bias", torch.cat([self.final_logits_bias, extra_bias], dim=1))

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                infer_input_ids=None, infer_attention_mask=None, infer_decoder_input_ids=None,
                infer_decoder_attention_mask=None, lmlabels=None, return_dict=True):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if infer_input_ids is not None:
            infer_encoder_outputs = self.model.encoder(
                input_ids=infer_input_ids, attention_mask=infer_attention_mask, return_dict=return_dict
            )
            infer_latent_hidden_state = infer_encoder_outputs.last_hidden_state[:, 0, :]
            infer_latent_logits = self.latent_head_m1(infer_latent_hidden_state)
            weight_memory = torch.mm(F.softmax(infer_latent_logits, dim=-1), self.memory1)
            
            infer_decoder_outputs = self.model.decoder(
                input_ids=infer_decoder_input_ids,
                attention_mask=infer_decoder_attention_mask,
                encoder_hidden_states=infer_encoder_outputs.last_hidden_state,
                return_dict=return_dict,
                latent_memory=weight_memory
            )
            infer_lm_logits = self.lm_head(infer_decoder_outputs.last_hidden_state) + self.final_logits_bias
            loss_fct = CrossEntropyLoss()
            infer_masked_lm_loss = loss_fct(infer_lm_logits.view(-1, self.config.vocab_size), lmlabels.view(-1))
            return Seq2SeqLMOutput(loss=infer_masked_lm_loss, logits=infer_lm_logits)
        
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        latent_hidden_state_m2 = encoder_outputs.last_hidden_state[:, 0, :]
        latent_logits_m2 = self.latent_head_m2(latent_hidden_state_m2)
        dialog_latent_variable = torch.mm(F.softmax(latent_logits_m2, dim=-1), self.memory2)
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            return_dict=return_dict,
            latent_memory=dialog_latent_variable
        )
        
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lmlabels.view(-1))
        
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits)

    def prepare_inputs_for_generation(self, decoder_input_ids, past=None, attention_mask=None, encoder_outputs=None,
                                      latent_variable=None, **kwargs):
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
            "latent_variable": latent_variable,
        }
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return torch.roll(labels, shifts=1, dims=-1)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(tuple(past_state.index_select(0, beam_idx) for past_state in layer_past) for layer_past in past)
