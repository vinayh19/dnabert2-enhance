# Contributions to the forked DNABERT-2 codebase
   
   This repository is a fork of MAGICS-LAB/DNABERT_2 (Apache-2.0). This file
   enumerates the modifications made in this fork.
   
   ## Infrastructure modernization
   - `bert_layers.py`: FlashAttention-2 backend replacing Triton implementation.
     Motivation: In order to make the training more efficient and decrease the run time 
     and make it compatible with the High performance computing clusters housing next-gen NVIDIA GPUs. 
     This change will help with reliably improving reproducability.
   
   ## MLM masking ablation
   - `configs/yaml`: One yaml configuration for 256/mlm15-15. Masking rates parameterized. 
     Make sure the end user change any parameter(s) according to their set up.
   - Note: the clean masking comparison is between the two 256 context models. See README §Limitations.
   
   ## Perplexity tracking
   - `mosaic_bert.py`: MLM perplexity computed over masked positions only. 
      Logged to WandB per eval step. The function is directly used from composer.utils
   
   ## Best-checkpoint capture
   - `configs/*.yaml`, `mosaic_bert.py`: Checkpoint saved at minimum eval cross entropy loss across training. Reduces need for post-hoc checkpoint selection.
  
