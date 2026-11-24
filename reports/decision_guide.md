# Decision guide

No decision is made in this file.

Ring Attention, DistFlashAttn, DeepSpeed Ulysses, and USP have not been timed on NVIDIA GPUs. There is no measured throughput, peak memory, or overlap to compare across sequence length, GPU count, or network. Choosing a method from the papers’ tables, or from the CPU correctness checks, would not answer the assignment.

The guide can be written after matched runs exist in `results/raw` and notebook 04 has loaded them. It must stay within one or two rendered pages, name the hardware and dtype, and mark every untested cell as untested.
