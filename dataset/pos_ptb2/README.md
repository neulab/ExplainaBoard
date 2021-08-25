# Dataset for PTB2


## Description
Penn Treebank 2 (PTB2) is one of the most known English POS-tagging datasets. The task consists of annotating each word with its Part-of-Speech tag, containing 45 different POS tags. The pos-tagging models are evaluated based on accuracy.

## Meta Data
* Official Homepage: https://catalog.ldc.upenn.edu/LDC99T42 
* Download link: https://catalog.ldc.upenn.edu/LDC99T42
* Paper: [Building a Large Annotated Corpus of English: The Penn Treebank](https://dl.acm.org/doi/10.5555/972470.972475)
* Github Repo: None
* Corresponding Author: Not clear
* Supported Task: POS-tagging  
* Language: English 



## Data Structure
### Example

```
"token"	"true-tag"	"prediction-tag"
The	DT	DT
complicated	VBN	JJ
language	NN	NN
in	IN	IN
the	DT	DT
huge	JJ	JJ
new	JJ	JJ
law	NN	NN
has	VBZ	VBZ
muddied	VBN	VBN
the	DT	DT
fight	NN	NN
.	.	.

The	DT	DT
law	NN	NN
does	VBZ	VBZ
allow	VB	VB
the	DT	DT
RTC	NNP	NNP
to	TO	TO
borrow	VB	VB
from	IN	IN
the	DT	DT
Treasury	NNP	NNP
up	IN	IN
to	TO	TO
$	$	$
5	CD	CD
billion	CD	CD
at	IN	IN
any	DT	DT
time	NN	NN
.	.	.
```


### Format
The ``.csv`` is the most common format for POS-tagging datasets, which always keep the dataset in three columns: token, true-tag, and predicted-tag, separated by space.


### Split
In most of the works, sections from 0 to 18 are used for training, sections from 19 to 21 are used for validation, and sections from 22 to 24 are used for testing.

## Reference
 ```
@article{marcus1993building,
  title={Building a large annotated corpus of English: The Penn Treebank},
  author={Marcus, Mitchell and Santorini, Beatrice and Marcinkiewicz, Mary Ann},
  year={1993}
}
```
