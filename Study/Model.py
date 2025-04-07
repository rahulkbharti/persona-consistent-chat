import torch
import torch.nn as nn
from transformers import BartModel, BartConfig

class MemoryModule(nn.Module):
    def __init__(self, hidden_dim, memory_size):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.attention = nn.Linear(hidden_dim, memory_size)
    
    def forward(self, last_hidden_state):
        attn_weights = torch.softmax(self.attention(last_hidden_state), dim=-1)
        memory_output = torch.matmul(attn_weights, self.memory)
        return memory_output

class ModifiedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bart_encoder = BartModel(config).encoder
        self.memory_module = MemoryModule(config.d_model, memory_size=10)
    
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state.clone()
        modified_last_hidden = self.memory_module(hidden_states[:, -1, :])
        hidden_states[:, -1, :] = modified_last_hidden
        return hidden_states

class ModifiedDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bart_decoder = BartModel(config).decoder
    
    def forward(self, input_ids, encoder_hidden_states, attention_mask, decoder_attention_mask, memory_output):
        input_embeddings = self.bart_decoder.embed_tokens(input_ids).clone()
        input_embeddings[:, 0, :] += memory_output
        decoder_outputs = self.bart_decoder(
            inputs_embeds=input_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
        return decoder_outputs.last_hidden_state

class CustomBART(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ModifiedEncoder(config)
        self.decoder = ModifiedDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_hidden_states = self.encoder(input_ids, attention_mask)
        memory_output = encoder_hidden_states[:, -1, :]
        decoder_hidden_states = self.decoder(
            decoder_input_ids, encoder_hidden_states, attention_mask, decoder_attention_mask, memory_output
        )
        logits = self.lm_head(decoder_hidden_states)
        return logits

config = BartConfig.from_pretrained("facebook/bart-base")
model = CustomBART(config)

print(model)
