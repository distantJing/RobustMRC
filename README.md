# Improving the robustness of machine reading comprehension model with hierarchical knowledge and auxiliary unanswerability prediction

This repository provides the official TensorFlow implementation of the research paper [Improving the robustness of machine reading comprehension model with hierarchical knowledge and auxiliary unanswerability prediction](https://www.sciencedirect.com/science/article/pii/S0950705120303567) (**Accepted by [Knowledge-Based Systems 2020]**). 

## Requirements
* tensorflow-gpu >= 1.5.0
* Glove
* Elmo


## Usage
```bash
python config.py --mode prepro
python config.py --mode train
python config.py --mode test
```

## Performance

||EM|F1|
|---|---|---|
|Dev 1.1|73.9|82.4|
|Test 1.1|75.4|83.4|
|AddSent|58.3|65.5|
|AddOneSent|65.2|72.5|


## How to cite
If you extend or use this work, please cite the relevant papers:
```bibtex
@article{DBLP:journals/kbs/WuX20,
  author       = {Zhijing Wu and
                  Hua Xu},
  title        = {Improving the robustness of machine reading comprehension model with
                  hierarchical knowledge and auxiliary unanswerability prediction},
  journal      = {Knowledge-Based Systems},
  volume       = {203},
  pages        = {106075},
  year         = {2020},
}
@inproceedings{DBLP:conf/aaai/WuX20,
  author       = {Zhijing Wu and
                  Hua Xu},
  title        = {A Multi-Task Learning Machine Reading Comprehension Model for Noisy
                  Document (Student Abstract)},
  booktitle    = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence 2020},
  pages        = {13963--13964},
  year         = {2020},
}
```
