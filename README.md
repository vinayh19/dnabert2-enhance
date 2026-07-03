# DNABERT-2 Reproduction and Modernization

Reproduction and infrastructure modernization of the DNABERT-2 (Zhou et al., 2023) pretraining pipeline feeding onto a related work on Personalized Human Genome modeling, with this version restricted to the multi-species genome sequence corpus provided by the DNABERT-2 authors, with a controlled ablation over MLM train/eval masking rates. Weights and code released under Apache-2.0. 

> **Scope.** This repository reproduces the DNABERT-2 pretraining recipe on the released multi-species corpus (135 genomes, ~32.5 Gbp). It is not a drop-in replacement for the released
> `zhihan1996/DNABERT-2-117M` checkpoint. This artifact is intended as a controlled baseline for MLM masking-rate and context-length ablations, and as a modernized starting point for further pretraining and fine-tuning.

## TL;DR
- Reproduced DNABERT-2 pretraining architecture (BPE tokenizer, 117M-parameter BERT variant) on the DNABERT-2 multi-species corpus.
- Modernized the stack: PyTorch 2.9, FlashAttention-2, CUDA 12.8, NVIDIA H100/A100/B200 support, deterministic best-eval-loss checkpoint capture.
- MosaicML Composer training framework.
- MLM masking-rate ablation across three configurations: {170 ctx, 30/15 train/eval mask}, {256 ctx, 15/15}, {256 ctx, 30/30}.
- Downstream fine-tuning evaluated on GUE binary tasks (EMP, transcription factor prediction human, splice site prediction, promoter core).


## Corpus and expected performance gap 
[explicit statement: human-only vs multi-species; Applicable only for further trained model]

## Pretraining
### Data
- Source: DNABERT-2 released multi-species pretraining corpus (135 genomes, ~32.5 GB).
- Distribution: https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view (as linked from the DNABERT-2 repository)
- No individualized, clinical, or human-subjects data.
- Sequence length: 1000 bp windows, 256 BPE tokens max.

### Model
- Architecture: BERT encoder, 117M parameters, matching DNABERT-2 configuration.
- Tokenizer: reused from `zhihan1996/DNABERT-2-117M` (BPE, 4096 vocab). Not retrained.
- Attention: FlashAttention-2, replacing the Triton attention dependency in the upstream DNABERT-2 codebase.

### Objective
- Masked language modeling.
- Ablation cells: [Applicable only for further trained model]
  | Cell | Train mask | Eval mask | Motivation                   |
  |------|------------|-----------|------------------------------|
  | A    | 15%        | 15%       | DNABERT-2 default; baseline. |
  | B    | 30%        | 30%       | hypothesis; symmetric high mask. |
  | C    | 30%        | 15%       | Asymmetric; higher signal per step, standard eval. |

### Training
- Optimizer: DecoupledAdamW, betas (0.9, 0.98), eps 1e-6, weight decay 1e-4.
- LR schedule: linear decay with warmup, peak LR 7.07e-4, warmup 8% of duration, decay to 0.02× peak.
- Global batch size: 8192.
- Duration: 250,000 batches (~70 epochs over corpus).
- Precision: `amp_bf16`.
- Hardware: NVIDIA B200.
- Wall-clock: ~28 hours per configuration.

## Downstream evaluation
- Benchmark: GUE (Zhou et al., 2023).
- Tasks: EMP, transcription factor prediction (human), splice site prediction, promoter core — all binary.
- Fine-tuning protocol: LR 1e-4, up to 50k batches.
- Baseline: published `zhihan1996/DNABERT-2-117M` under identical fine-tuning protocol.
- Results: (TBA)

## Reproducing this work
See [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Model checkpoints available at https://huggingface.co/vinayh19

## Limitations
- The ablation is for testing purpose only: context length (170 - (mean token length) vs 256) covaries with masking rate across the three configurations. The clean masking comparison is between the two 256-token configurations (A vs B). The 170-token / 30-15 configuration is included for transparency but does not isolate a single variable.
- Downstream fine-tuning uses single-seed point estimates.
- The tokenizer is reused without retraining; potential tokenization drift under corpus-specific vocabularies is not investigated.
- BPE tokenization propagates single-nucleotide-variant-induced token boundary shifts, which limits fidelity of variant-level representation. This artifact is a controlled baseline; single-nucleotide architectures (Caduceus, HyenaDNA, Evo2) are more appropriate substrates for individualized modeling.
- The GUE binary tasks characterize population-level regulatory representation. They do not directly test individualized regulatory prediction, which would require allele-specific expression, individual-level eQTL, or allele-specific chromatin accessibility tasks.

## Related/Future work
Extension of this pipeline to personal genomic sequences (continued pretraining on personalized haplotype-resolved corpora, evaluation on non-coding regulatory tasks) is ongoing under institutional data governance and is not part of this public release.
Downstream directions include: (i) continued pretraining on individualized human genome corpora to test whether variant-conditioned representations improve individualized non-coding regulatory prediction; (ii) comparison across tokenization inductive biases — BPE (DNABERT-2), 6-mer (Nucleotide Transformer v2), and single-nucleotide (Caduceus or Evo2) — for variant representation; (iii) evaluation on individualized regulatory tasks that directly test the personalization hypothesis, which the population-level GUE tasks used here do not.

## Acknowledgments
This work builds on the DNABERT-2 codebase (MAGICS-LAB, https://github.com/MAGICS-LAB/DNABERT_2) and released tokenizer (`zhihan1996/DNABERT-2-117M`), and on the MosaicML Examples BERT training pipeline. See `NOTICE` for full attribution.

## License
Apache-2.0, matching the upstream DNABERT-2 license.

## References
- Zhou et al., DNABERT-2 (2023). https://arxiv.org/abs/2306.15006
- Dao, FlashAttention-2 (2023). https://arxiv.org/abs/2307.08691
- MosaicML Composer. https://github.com/mosaicml/composer
