# Dataset for CoNLL-2000


## Description
CoNLL-2000 is a popular Chunking dataset, which was collected from Wall Street Journal corpus (WSJ).

## Meta Data
* Official Homepage: https://www.clips.uantwerpen.be/conll2000/chunking/
* Download link: https://www.clips.uantwerpen.be/conll2000/chunking/
* Paper: [Introduction to the CoNLL-2000 Shared Task Chunking](https://aclanthology.org/W00-0726.pdf)
* Github Repo: None
* Corresponding Author: [Erik Tjong Kim Sang](https://ifarm.nl/erikt/)
* Supported Task: Text Chunking 
* Language: English 



## Data Structure
### Example

```
"token"	"true-tag"	"prediction-tag"
In	B-PP	B-PP
January	B-NP	B-NP
,	O	O
he	B-NP	B-NP
accepted	B-VP	B-VP
the	B-NP	B-NP
position	I-NP	I-NP
of	B-PP	B-PP
vice	B-NP	B-NP
chairman	I-NP	I-NP
of	B-PP	B-PP
Carlyle	B-NP	B-NP
Group	I-NP	I-NP
,	O	O
a	B-NP	B-NP
merchant	I-NP	I-NP
banking	I-NP	I-NP
concern	I-NP	I-NP
.	O	O

SHEARSON	B-NP	B-NP
LEHMAN	I-NP	I-NP
HUTTON	I-NP	I-NP
Inc	I-NP	I-NP
.	O	O
```


### Format
The ``.csv`` is the most common format for Chunk datasets, which always keep the dataset in three columns: token, true-tag, and predicted-tag, separated by space.


### Split
CoNLL-2000 provides an official split of training and testing set. When train a CHUNK model, we randomly select 20% of the training samples from the training set as the development set. 




## Reference
 ```
@article{sang2000introduction,
  title={Introduction to the CoNLL-2000 shared task: Chunking},
  author={Sang, Erik F and Buchholz, Sabine},
  journal={arXiv preprint cs/0009008},
  year={2000}
}
```
