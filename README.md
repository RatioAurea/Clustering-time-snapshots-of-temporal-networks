# Temporal Network Clustering — Code and Datasets

This repository contains the code and data accompanying our paper *Clustering Time Snapshots of Temporal Networks with Applications to Synthetic and Real World Systems* (https://arxiv.org/abs/2412.12187) on clustering time-snapshots of temporal networks. It includes implementations of the proposed methods, a synthetic benchmark generator, and all datasets used in our experiments.

## Repository Contents

- `lne.ipynb`: Implementation of the **LNE (Low-dimensional Node Embedding)** method proposed in the paper.
- `imc.ipynb`: Implementation of the **IMC (Invariant Measure Comparison)** method introduced as a simplified variant of LNE.
- `generator.ipynb`: Benchmark generator for synthetic temporal network datasets, as described in the paper.
- `datasets/`: Folder containing all datasets used in the study, including:
  - Synthetic datasets generated via `generator.ipynb`
  - Preprocessed real-world datasets used in the experiments

## Datasets

The `datasets/` folder contains both synthetic benchmarks and real-world temporal network datasets. References for the real-world datasets are listed below:

- **Cholera dataset**  
  Hsiao, A. et al. *Members of the human gut microbiota involved in recovery from vibrio cholera infection.* *Nature* 515 (2014). DOI: [https://doi.org/10.1038/nature13738](https://doi.org/10.1038/nature13738)
  and
  Melnyk, K. *graphKKE: Library for the analysis of time-evolving graphs* (2020). GitHub: [https://github.com/k-melnyk/graphKKE](https://github.com/k-melnyk/graphKKE) (Accessed: 2024-08-02)
  
- **Hunter-Gatherer dataset**  
  Zonker, J., Padilla-Iglesias, C. & Conrad, N. D. *Supplementary code and data for Royal Society Open Science Manuscript rsos.230495* (2023). DOI: [https://doi.org/10.12752/9254](https://doi.org/10.12752/9254)

- **Opinion Dynamics dataset**  
  Helfmann, L., Djurdjevac Conrad, N., Lorenz-Spreen, P. & Schütte, C. *Supplementary code for the paper modelling opinion dynamics under the impact of influencer and media strategies* (2023). DOI: [https://doi.org/10.12752/9267](https://doi.org/10.12752/9267)

- **Cell Division dataset**  
  Lucas, M. et al. *Inferring cell cycle phases from a partially temporal network of protein interactions.* *Cell Reports Methods* 3, 100397 (2023). DOI: [https://doi.org/10.1016/j.crmeth.2023.100397](https://doi.org/10.1016/j.crmeth.2023.100397)
  
- **Primary School Contacts dataset**  
  Stehlé, J. et al. *High-resolution measurements of face-to-face contact patterns in a primary school.* *PLOS ONE* 6, 1–13 (2011). DOI: [https://doi.org/10.1371/journal.pone.0023176](https://doi.org/10.1371/journal.pone.0023176)

Please refer to the paper and the corresponding notebooks for descriptions and usage of each dataset.
