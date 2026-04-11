import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import math
import os
import argparse

class TemporalEncoder(nn.Module):
    def __init__(self, c_in, d_model):
        super(TemporalEncoder, self).__init__()
        
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)

        self.batchNorm = nn.BatchNorm1d(num_features=d_model)
        self.drop = nn.Dropout(0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):

        x = self.tokenConv(x.permute(0, 2, 1))
        x = self.batchNorm(x)
        x = F.gelu(x)
        x=x.permute(0, 2, 1)
        x = self.drop(x)
        return x


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=768, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class EEGEncoder(nn.Module):
    def __init__(
        self,
        class_number
    ):
        super().__init__()
        if class_number == 1:
            self.n_fft = 200
            self.hop_length = 100
            self.n_channels = 16
        else:
            self.n_fft = 100
            self.hop_length = 50
            self.n_channels = 10
        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=768, n_freq=self.n_fft // 2 + 1
        )

        self.channel_tokens = nn.Embedding( self.n_channels, 768)
        self.index = nn.Parameter(
            torch.LongTensor(range(self.n_channels)), requires_grad=False
        )
        self.positional_encoding = PositionalEncoding(768)
    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, token_numbers ,token_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            # channel_spec_emb=self.TemporalEncoder(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            channel_emb = channel_token_emb+channel_spec_emb
            emb_seq.append(channel_emb)
        emb = torch.cat(emb_seq, dim=1)
        return emb


class gpt4ts(nn.Module):
    def __init__(self,class_number):
        super(gpt4ts, self).__init__()
        self.gpt_layers = 6
        self.num_classes = class_number
        self.d_model = 768

        self.biot = EEGEncoder(class_number)
        gpt2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'gpt2')
        self.gpt2 = GPT2Model.from_pretrained(gpt2_path, local_files_only=True,output_attentions=True, output_hidden_states=True,attn_implementation='eager')


        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name  or 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.ln_proj = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_enc):
        outputs = self.biot(x_enc)
        if self.num_classes == 5 or self.num_classes == 1:
            outputs=outputs.mean(dim=1)
        # print(outputs.shape)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        # print(outputs.shape)
        if self.num_classes == 4:
            outputs=outputs.mean(dim=1)
        # outputs=outputs.mean(dim=1)

        outputs = self.ln_proj(outputs)
        outputs = self.classifier(outputs)
        outputs = self.dropout(outputs)
        return outputs
    


