import numpy as np
import torch
import torch.nn as nn
from base import BaseModel


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


class GradientClip(nn.Module):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = torch.tensor(-alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AudioTextClap(BaseModel):
    def __init__(self,
                 audio_encoder,
                 text_encoder,
                 audio_dim,
                 text_dim,
                 shared_dim,
                 gradient_clip=1):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gradient_clip = gradient_clip
        if gradient_clip != 1:
            self.audio_gradient_clip = GradientClip(gradient_clip)
            self.text_gradient_clip = GradientClip(gradient_clip)
 
    def forward(self, input_dict):

        batch_size = input_dict["waveform"].size(0)
        num_captions = input_dict["num_captions"]

        audio_keys = ["waveform", "wave_length"]
        audio_input = {k: input_dict[k] for k in audio_keys}
        audio_emb = self.audio_encoder(**audio_input)["clip_emb"]
        if self.gradient_clip != 1:
            audio_emb = self.audio_gradient_clip(audio_emb)
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
 
        text_keys = ["input_ids", "token_type_ids", "attention_mask"]
        text_input = {}
        for k in text_keys:
            v = input_dict[k]
            text_input[k] = v.reshape(batch_size * num_captions, *v.size()[2:])
        text_emb = self.text_encoder(**text_input)["clip_emb"]
        if self.gradient_clip != 1:
            text_emb = self.text_gradient_clip(text_emb)
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        text_emb = text_emb.view(batch_size, num_captions, -1)
                
        output = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "logit_scale": self.logit_scale.exp()
        }

        return output

    def evaluate_retrieval(self, inputs):
        return self.forward(inputs)
    
    def encode_audio(self, waveform, wave_length):
        audio_emb = self.audio_encoder(waveform, wave_length)["clip_emb"]
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        return audio_emb

    def encode_text(self, **text):
        text_emb = self.text_encoder(**text)["clip_emb"]
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        return text_emb


class AudioSingleTextClap(AudioTextClap):

    def forward(self, input_dict):
        audio_keys = ["waveform", "wave_length"]
        audio_input = {k: input_dict[k] for k in audio_keys}
        audio_emb = self.audio_encoder(**audio_input)["clip_emb"]
        if self.gradient_clip != 1:
            audio_emb = self.audio_gradient_clip(audio_emb)
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
 
        text_keys = ["input_ids", "token_type_ids", "attention_mask"]
        text_input = {k: input_dict[k] for k in text_keys}
        text_emb = self.text_encoder(**text_input)["clip_emb"]
        if self.gradient_clip != 1:
            text_emb = self.text_gradient_clip(text_emb)
        text_emb = self.text_proj(text_emb)
        norm = text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.div(norm + 1e-7).clip(-1e3, 1e3)
                
        output = {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "logit_scale": self.logit_scale.exp()
        }

        return output

    def evaluate_retrieval(self, input_dict):
        if "num_captions" in input_dict:
            return super().forward(input_dict)
        else:
            return self.forward(input_dict)


