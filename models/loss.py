import torch 
import torch.nn as nn


class InfoNceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, audio_emb=None, text_emb=None, logit_scale=None, sims=None):
        if sims is None:
            batch_size = audio_emb.size(0)
            if text_emb.ndim == 3:
                logit = logit_scale * audio_emb @ text_emb[:, 0, :].T
            else:
                logit = logit_scale * audio_emb @ text_emb.T
            # (batch_size, batch_size)
        else:
            logit = logit_scale * sims.T
            batch_size = sims.size(0)
        label = torch.arange(batch_size).to(logit.device)
        loss_a = self.loss_fn(logit.T, label)
        loss_t = self.loss_fn(logit, label)
        loss = (loss_a + loss_t) / 2
        return loss

