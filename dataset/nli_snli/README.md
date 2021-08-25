# Dataset for CoNLL-2003


## Description
The SNLI dataset (Stanford Natural Language Inference) aims to judge the relation between sentences given that they describe the same event. The relation can be “entailment”, “neutral”, “contradiction” or “-”, where “-” indicates that an agreement could not be reached.


## Meta Data
* Official Homepage: https://nlp.stanford.edu/projects/snli/
* Download link: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
* Paper: [A large annotated corpus for learning natural language inference](https://arxiv.org/pdf/1508.05326v1.pdf)
* Github Repo: None
* Corresponding Author: [Christopher D. Manning](https://nlp.stanford.edu/manning/)
* Supported Task: Natural Language Inference
* Language: English 



## Data Structure
### Example

```
{
sentence A: "This church choir sings to the masses as they sing joyous songs from the book at a church.",
sentence B: "The church has cracks in the ceiling.",
label: "neutral",
}
```


### Format
NOT CLEAR~~~~~


### Split
SNLI dataset offer the official training, development, and testing set.


## Reference
 ```
 @article{sang2003introduction,
  title={Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition},
  author={Sang, Erik F and De Meulder, Fien},
  journal={arXiv preprint cs/0306050},
  year={2003}
}
```
