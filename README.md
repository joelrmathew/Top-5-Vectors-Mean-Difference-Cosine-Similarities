This repo contains exploratory interpretability experiments on Gemma-2B using Sparse Autoencoders (SAEs).

We evaluate Gemma’s predictions on PIQA (Physical Interaction QA) and select 5 correct and 5 incorrect examples.

For each selected layer (5, 10, 15, 20), we compute the mean activation difference between correct vs. incorrect last-token activations.

These difference vectors are compared to SAE decoder directions to identify top-k most aligned features.

Feature explanations are fetched from Neuronpedia and merged with cosine similarity scores.

The repo includes both the chosen PIQA examples and the layer-wise feature importance + explanations, providing a dataset for analyzing how Gemma’s internal features correlate with right vs. wrong reasoning.
