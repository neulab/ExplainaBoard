# Dataset for absa


## Description
Here include three widely used datasets for Aspect-Based Sentiment Analysis. 
## Meta Data
* Official Homepage: https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools 
* Download link:  https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
* Paper:  
  * [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://doi.org/10.3115/v1/s14-2004)
  * [Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](https://doi.org/10.3115/v1/p14-2009)
* Github Repo:   
  * https://github.com/ROGERDJQ/RoBERTaABSA.git
* Language:  English



## Data Structure
### Example
```tsv
Boot time 	"[[ Boot time ]] is super fast , around anywhere from 35 seconds to 1 minute ."	positive	positive	0.996909559
```

### Format
Each line contains five elements separated by a "tab", which are aspect term, sentence, and ground truth label, prediction label, prediction probability.




## Reference
 ```tex
 @inproceedings{DBLP:conf/semeval/PontikiGPPAM14,
  author    = {Maria Pontiki and
               Dimitris Galanis and
               John Pavlopoulos and
               Harris Papageorgiou and
               Ion Androutsopoulos and
               Suresh Manandhar},
  title     = {SemEval-2014 Task 4: Aspect Based Sentiment Analysis},
  booktitle = {Proceedings of the 8th International Workshop on Semantic Evaluation,
               SemEval@COLING 2014, Dublin, Ireland, August 23-24, 2014},
  pages     = {27--35},
  publisher = {The Association for Computer Linguistics},
  year      = {2014},
  url       = {https://doi.org/10.3115/v1/s14-2004},
  doi       = {10.3115/v1/s14-2004},
}


@inproceedings{DBLP:conf/acl/DongWTTZX14,
  author    = {Li Dong and
               Furu Wei and
               Chuanqi Tan and
               Duyu Tang and
               Ming Zhou and
               Ke Xu},
  title     = {Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment
               Classification},
  booktitle = {Proceedings of the 52nd Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2014, June 22-27, 2014, Baltimore, MD, USA, Volume
               2: Short Papers},
  pages     = {49--54},
  publisher = {The Association for Computer Linguistics},
  year      = {2014},
  url       = {https://doi.org/10.3115/v1/p14-2009},
  doi       = {10.3115/v1/p14-2009},
}


@inproceedings{DBLP:conf/naacl/DaiYSLQ21,
  author    = {Junqi Dai and
               Hang Yan and
               Tianxiang Sun and
               Pengfei Liu and
               Xipeng Qiu},

  title     = {Does syntax matter? {A} strong baseline for Aspect-based Sentiment
               Analysis with RoBERTa},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {1816--1829},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.naacl-main.146},
  doi       = {10.18653/v1/2021.naacl-main.146},
}
```
