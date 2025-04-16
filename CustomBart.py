import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel

class CustomBART(nn.Module):
    def __init__(self):
        super(CustomBART, self).__init__()

        # Load tokenizer and model
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = BartModel.from_pretrained("facebook/bart-base")

        # Extract encoder and decoder
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder


    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        """Pass inputs through encoder and decoder."""
        # Encoder forward pass
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Decoder forward pass with cross-attention
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )

        return decoder_outputs.last_hidden_state  # Decoder output embeddings

# Example Usage
model = CustomBART()

# Dummy inputs (batch size=2, sequence length=5)
input_ids = torch.randint(0, 50265, (2, 5))
attention_mask = torch.ones_like(input_ids)
decoder_input_ids = torch.randint(0, 50265, (2, 5))
decoder_attention_mask = torch.ones_like(decoder_input_ids)

# Forward pass
output = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
print(output.shape)  # (batch_size, seq_len, hidden_size)

#changes done by gaurav