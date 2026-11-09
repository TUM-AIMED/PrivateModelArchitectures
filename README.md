# Models for DP-SGD Training
DP-SGD introduces challenges for CNN training. We collect models that are already adapted to the prerequisites and proved to be robust in training.

## Models

| Model | Task | Paper |
|---|---|---|
| `ResNet9` | Classification | Klause et al., TPDP 2022 ([arXiv:2203.00324](https://arxiv.org/abs/2203.00324)) |
| `SmoothNet` | Classification | Remerscheid et al., TPDP 2022 ([arXiv:2205.04095](https://arxiv.org/abs/2205.04095)) |
| `LinkNet9` | Segmentation | Klause et al., TPDP 2022 ([arXiv:2203.00324](https://arxiv.org/abs/2203.00324)) |
| `UNet9` | Segmentation | Klause et al., TPDP 2022 ([arXiv:2203.00324](https://arxiv.org/abs/2203.00324)) |

## Citation

If you use these models in academic work, please consider citing the paper for each architecture you use:

```bibtex
@inproceedings{klause2022differentially,
  title     = {Differentially private training of residual networks with scale normalisation},
  author    = {Klause, Helena and Ziller, Alexander and Rueckert, Daniel and Hammernik, Kerstin and Kaissis, Georgios},
  booktitle = {Theory and Practice of Differential Privacy (TPDP) Workshop at ICML},
  year      = {2022},
  eprint    = {2203.00324},
  archivePrefix = {arXiv}
}

@inproceedings{remerscheid2022smoothnets,
  title     = {SmoothNets: Optimizing CNN architecture design for differentially private deep learning},
  author    = {Remerscheid, Nicolas W. and Ziller, Alexander and Rueckert, Daniel and Kaissis, Georgios},
  booktitle = {Theory and Practice of Differential Privacy (TPDP) Workshop at ICML},
  year      = {2022},
  eprint    = {2205.04095},
  archivePrefix = {arXiv}
}
```

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE), © 2023 AIMLAB, Technical University of Munich. You may use it commercially. Please keep the [NOTICE](NOTICE) file with any redistribution.

**Pretrained weights:** the optional RadImageNet-pretrained `ResNet9` weights (`use_pretrained_weights=True`) were trained on the [RadImageNet](https://www.radimagenet.com/) dataset. They are not covered by the Apache License and may be used for non-commercial research only, in line with the RadImageNet Research Use Agreement.
