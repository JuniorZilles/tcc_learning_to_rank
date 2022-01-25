<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://miro.medium.com/max/1200/1*d9drCvBUwmvTlAasuErLUA.jpeg" alt="Learning to Rank"></a>
</p>

<h3 align="center">Implementações TCC sobre Learning to Rank</h3>

<div align="center">

[![Python](https://img.shields.io/badge/language-python-blue.svg)]()
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub issues](https://img.shields.io/github/issues/JuniorZilles/tcc_learning_to_rank.svg)](https://GitHub.com/JuniorZilles/tcc_learning_to_rank/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/JuniorZilles/tcc_learning_to_rank.svg)](https://GitHub.com/JuniorZilles/tcc_learning_to_rank/pull/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Esse repositório tem como finalidade comportar os códigos implementados para o desenvolvimento do trabalho de conclusão do curso.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)

## 🧐 About <a name = "about"></a>


Write about 1-2 paragraphs describing the purpose of your project.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

- [Python](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Graphviz](https://graphviz.gitlab.io/download/)

### Installing

```
pip -u install requirements.txt
```

## 🔧 Running the tests <a name = "tests"></a>

Explain how to run the automated tests for this system.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## 🎈 Usage <a name="usage"></a>

Add notes about how to use the system.


## train SVM

svm_rank_learn -c 3 data/MSLR30K/train.dat models/rankSVM.MSLR30K.model 1> train_logs/train.rankSVM.MSLR30K.log 2>&1 

## evaluate datasets from letor 4.0

perl Eval-Score-4.0.pl data/TD2004/test.txt predicted/xgboost.regression.td2004.txt evaluation_letor\xgboost.regression.td2004.txt 0

## evaluate datasets MSLR30K and MSLR10K
perl Eval-Score-MSLR.pl data/MSLR10K/test.txt predicted/xgboost.regression.mslr10k.txt evaluation_letor\xgboost.regression.mslr10k.txt 0