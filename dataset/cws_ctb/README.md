# Dataset for CTB


## Description
CTB (Chinese Treebank dataset) is a popular Chinese Word Segmentation (CWS) dataset. 

## Meta Data
* Official Homepage: https://www.aclweb.org/mirror/ijcnlp08/sighan6/callforpapers.htm
* Download link: https://www.aclweb.org/mirror/ijcnlp08/sighan6/callforpapers.htm
* Paper: [Sixth SIGHAN Workshop on Chinese Language Processing](https://aclanthology.org/I08-4.pdf)
* Github Repo: None
* Corresponding Author: Not clear
* Supported Task: Chinese Word Segmentation
* Language: Chinese 



## Data Structure
### Example

```
"token"	"true-tag"	"prediction-tag"
中	B	B
国	E	E
最	B	B
大	E	E
氨	B	B
纶	M	M
丝	E	E
生	B	B
产	E	E
基	B	B
地	E	E
在	S	S
连	B	B
云	M	M
港	E	E
建	B	B
成	E	E
```


### Format
The ``.csv`` is the most common format for CWS datasets, which always keep the dataset in three columns: tokens, true-tags, and predicted-tags, separated by space. Different sentences are separated with a blank line.


### Split
CTB provides an official split of training, development, and testing set.


## Reference
 ```
 @inproceedings{jin2008fourth,
  title={The fourth international chinese language processing bakeoff: Chinese word segmentation, named entity recognition and chinese pos tagging},
  author={Jin, Guangjin and Chen, Xiao},
  booktitle={Proceedings of the sixth SIGHAN workshop on Chinese language processing},
  year={2008}
}
```
