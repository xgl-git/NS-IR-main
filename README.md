

### Dependencies 

We use anaconda to create python environment:
```
conda create --name ns-ir python=3.9
```
Install all required libraries:
```
pip install -r requirements.txt
```
### Model Preparation
bge-large https://huggingface.co/BAAI/bge-large-en-v1.5


### Data Preparation
NegConstraint:  
NegConstraint.zip

BEIR benchmark:  
https://github.com/beir-cellar/beir/tree/main

TREC benchmark:  
DL'19 https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019  
DL'20 https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020

### Run Program
```
bash run.sh
```