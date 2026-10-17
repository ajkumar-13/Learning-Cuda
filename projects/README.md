# Projects

Learning CUDA has no separately released projects yet. Every kernel the series teaches lives inside its post, beside the CPU model that reproduces it, under `posts/NN-slug/snippets/`. A candidate first project, with id cuda-p01, is a from-scratch attention kernel benchmarked against PyTorch's `scaled_dot_product_attention` across sequence lengths, judged by the roofline of [post 10](../posts/10-roofline/index.md); it is described but not committed. A project that is later released gets its own directory here, with its own README and release notes.
