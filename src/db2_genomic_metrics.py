# genomic_metrics.py - Shared metrics for Composer-based training - DNABERT2 versions
import math
import torch
from torchmetrics import Metric
"""
Perplexity from mean cross-entropy over masked tokens only (labels != -100).
Consistent with LanguageCrossEntropy(ignore_index=-100).
Adding this to get_metrics() is all that is needed - Composer handles logging to WandB automatically under train/MaskedPerplexity and val/MaskedPerplexity.
"""
class MaskedPerplexity(Metric):
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state('sum_loss',      default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('n_masked_toks', default=torch.tensor(0),   dist_reduce_fx='sum')
 
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        B, L, V = logits.shape
        #sum reduction over masked positions only;
        loss = torch.nn.functional.cross_entropy(logits.reshape(B * L, V).float(),
            labels.reshape(B * L), ignore_index=-100,
            reduction='sum')
        n_masked = (labels != -100).sum()
        self.sum_loss += loss.detach()
        self.n_masked_toks += n_masked.detach()
 
    def compute(self) -> torch.Tensor:
        mean_loss = self.sum_loss / self.n_masked_toks.clamp(min=1) #mean_loss = self.sum_loss / self.n_batches.clamp(min=1) - alternative
        return torch.exp(mean_loss.clamp(max=20))