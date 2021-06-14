# System output of PEGASUS on Newsroom


## Model Description
PEGASUS uses pre-training objectives tailored for abstractive text summarization. Specifically, in PEGASUS, important sentences are removed/masked from an input document and are generated together as one output sequence from the remaining sentences, similar to an extractive summary. 

## Meta Data
* Github Repo: [google-research/pegasus](https://github.com/google-research/pegasus)
* Paper: [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf)
* Corresponding Author: Peter J. Liu

## Results
The ROUGE scores achieved by PEGASUS on **Newsroom** dataset is shown below.

| Model variant | ROUGE-1|ROUGE-2| ROUGE-L|
|:--- |:--- |:--- |:--- |
|PEGASUS-BASE |42.38 | 30.06 | 38.52 |
|PEGASUS-LARGE (C4)| 45.07 | 33.39 | 41.28 |
|PEGASUS-LARGE (HugeNews) | 45.15 |33.51|41.33|

## Reference
```
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```



