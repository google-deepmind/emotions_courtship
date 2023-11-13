# Emotions and courtship

This repository contains the code accompanying the publication **Emotions and
courtship help bonded pairs cooperate, but emotional agents are vulnerable to
deceit**, published in the Proceedings of the National Academy of Sciences, 2023.

## Usage

Clone this repository with

```sh
git clone https://github.com/google-deepmind/emotions_courtship.git
```

You might need to install some dependencies:

```sh
pip3 install ml_collections absl-py numpy tree
```

You can run the model with default parameters with:

```python
python3 -m emotions_courtship.run_play
```

You can change the model parameters by editing the `config.py` file.

You can also load the dataset and the code used to produce the figures in the
research article using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/emotions_courtship/blob/main/notebooks/data_analysis.ipynb)

The full data used in the paper is at https://storage.googleapis.com/emotions_courtship/data.zip

## Citing this work

```bibtex
@article{sadedin2023emotions,
  title={Emotions and courtship help bonded pairs cooperate, but emotional agents are vulnerable to deceit},
  author={Sadedin, S and Du\'e\~nez-Guzm\'an, EA and Leibo, JZ},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={46},
  pages={e2308911120},
  year={2023},
  doi={10.1073/pnas.2308911120},
  URL={https://www.pnas.org/doi/abs/10.1073/pnas.2308911120},
  publisher={National Academy of Sciences}
}
```

## License and disclaimer

Copyright 2023, The Authors.

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
