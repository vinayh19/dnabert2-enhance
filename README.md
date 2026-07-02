# DNABERT-2 Reproduction and Modernization for Precision Medicine

Reproduction and infrastructure modernization of the DNABERT-2 (Zhou et al., 2023) pretraining pipeline feeding onto a related work on Personalized Human Genome modeling, with this version restricted to the 1000 Genomes corpus provided by the DNABERT-2 authors, with a controlled ablation over MLM train/eval masking rates. Artifact from my work at Icahn School of Medicine at Mount Sinai. Weights and code released under Apache-2.0.

> **Scope.** This repository reproduces the DNABERT-2 pretraining *recipe* on a
> multi-species corpus. It is not a drop-in replacement for the released
> `zhihan1996/DNABERT-2-117M` checkpoint, which was trained on a 135-species,
> ~32.5 GB multi-species corpus. See "Corpus and expected performance gap" below.

## TL;DR
- Reproduced DNABERT-2 pretraining architecture (BPE tokenizer, 117M-param BERT variant).
- Modernized the stack: PyTorch [2.9], FlashAttention-2, [CUDA 12.8], (NVIDIA H100/A100/B200 GPU friendly) deterministic checkpointing with best-perplexity capture. 
- MosaicMl Composer framework code followed.
- Ran a controlled MLM masking-rate ablation: train/eval in {30/15, 30/30, 15/15}.
- Evaluated on the GUE benchmark (binary tasks: EMP, transcription factor prediction (human), splice site prediction Promoter core).

## Corpus and expected performance gap 
[explicit statement: human-only vs multi-species; Applicable only for further trained model]

## Pretraining
### Data
- Source: [https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view?usp=sharing]
- Corpus construction: Applicable only for further training [consensus sequences per individual]
- Sequence length: 1000 bp windows, 256 BPE tokens max.

### Model
- Architecture: BERT encoder, 117M parameters, matching DNABERT-2 configuration.
- Tokenizer: reused from `zhihan1996/DNABERT-2-117M` (BPE, 4096 vocab). Not retrained.
- Attention: FlashAttention-2 (replaces custom Triton dependencies and PyTorch SDPA in the original codebase).

### Objective
- Masked language modeling.
- Ablation cells: [Applicable only for further trained model]
  | Cell | Train mask | Eval mask | Motivation                   |
  |------|------------|-----------|------------------------------|
  | A    | 15%        | 15%       | DNABERT-2 default; baseline. |
  | B    | 30%        | 30%       | hypothesis; symmetric high mask. |
  | C    | 30%        | 15%       | Asymmetric; higher signal per step, standard eval. |

### Training
[optimizer, LR schedule, warmup, batch, grad accum, precision, hardware, wall-clock - TBA]

## Downstream evaluation [Public Checkpoint]
- Benchmark: GUE (Zhou et al., 2023).
- Tasks: EMP, TF prediction (human), splice site prediction — all binary.
- Fine-tuning protocol: LR=1e-4 over 50k batches
- Baselines: published `zhihan1996/DNABERT-2-117M` under identical fine-tuning protocol.

## Reproducing this work
[Phase 2 - TBA]

## Related/Future work
Extension of this pipeline to personal genomic sequences (continued pretraining on personalized haplotype-resolved corpora, evaluation on non-coding regulatory tasks) is ongoing under institutional data governance and is not part of this public release.

## Acknowledgments
This work builds directly on the DNABERT-2 codebase (MAGICS-LAB) and released tokenizer (zhihan1996/DNABERT-2-117M). See NOTICE for attribution.

## License
Apache-2.0, matching the upstream DNABERT-2 license.
