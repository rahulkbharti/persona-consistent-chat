import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForQuestionAnswering

class CustomBART(nn.Module):
    def __init__(self):
        super(CustomBART, self).__init__()

        # Load tokenizer and pre-trained BART for QA
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model = BartForQuestionAnswering.from_pretrained("facebook/bart-base")

        # Extract encoder & decoder
        self.encoder = self.model.model.encoder
        self.decoder = self.model.model.decoder

        # LM Head (for answer prediction)
        self.qa_outputs = self.model.qa_outputs  # Linear layer for start & end logits

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        """Forward pass for QA."""
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

        # Compute start & end logits for answer span prediction
        logits = self.qa_outputs(decoder_outputs.last_hidden_state)

        return logits  # (batch_size, seq_len, 2)

# Example Usage
model = CustomBART()

inputs = model.tokenizer("Hello How are You?", return_tensors="pt")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Dummy inputs (batch size=2, sequence length=5)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

decoder_input_ids = model.tokenizer("<s>", return_tensors="pt").input_ids
decoder_attention_mask = torch.ones_like(decoder_input_ids)

# Forward pass
output = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
print(output.shape)  # (batch_size, seq_len, 2)  -> Start & end logits
