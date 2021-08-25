# Dataset for CoNLL-2003


## Description
CoNLL-2003 is a popular NER dataset, which contains four named entity types: person, location, organization, and miscellaneous name.

## Meta Data
* Official Homepage: https://www.clips.uantwerpen.be/conll2003/ner/ 
* Download link: https://www.clips.uantwerpen.be/conll2003/eng.raw.tar
* Paper: [Introduction to the CoNLL-2003 Shared Task:
Language-Independent Named Entity Recognition](https://aclanthology.org/W03-0419.pdf)
* Github Repo: None
* Corresponding Author: [Erik Tjong Kim Sang](https://ifarm.nl/erikt/)
* Supported Task: Named Entity Recognition 
* Language: English 



## Data Structure
### Example

```
"token"	"true-tag"	"prediction-tag"
SOCCER	O	O
-	O	O
JAPAN	B-LOC	B-MISC
GET	O	O
LUCKY	O	O
WIN	O	O
,	O	O
CHINA	B-PER	B-ORG
IN	O	O
SURPRISE	O	O
DEFEAT	O	O
.	O	O

Nadim	B-PER	B-PER
Ladki	I-PER	I-PER

AL-AIN	B-LOC	B-LOC
,	O	O
United	B-LOC	B-LOC
Arab	I-LOC	I-LOC
Emirates	I-LOC	I-LOC
1996-12-06	O	O
```


### Format
The ``.csv`` is the most common format for NER datasets, which always keep the dataset in three columns: words, true-tags, and predicted-tags, separated by space.


### Split
CoNLL-2003 offer the official training, dev, and testing set.


## Reference
 ```
 @article{sang2003introduction,
  title={Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition},
  author={Sang, Erik F and De Meulder, Fien},
  journal={arXiv preprint cs/0306050},
  year={2003}
}
```
