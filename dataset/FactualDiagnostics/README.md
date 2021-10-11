# Dataset for FactualDiagnostics


## Description
This is the factual diagnostic test sets proposed in paper: Factuality Checker is not Faithful: Adversarial Meta-evaluation of Factuality in Summarization.

## Meta Data
* Download link:  
* Paper:  Factuality Checker is not Faithful: Adversarial Meta-evaluation of Factuality in Summarization
* Github Repo: https://github.com/zide05/AdvFact
* Corresponding Author:  Yiran Chen, Pengfei Liu, Xipeng Qiu
* Supported Task:  Factual error detection in summarization
* Language:  English



## Data Structure
### Example
```
{"id": "13599161", "text": "Cuadrilla , the firm behind the tests , said .....", "claim": "shale gas drilling in lancashire has been suspended after a magnitude-7. 5 earthquake struck.", "label": "INCORRECT"}
{"id": "39263182", "text": "Neil Aspin 's promotion-chasing hosts have not lost in nine National League matches while ...", "claim": "gateshead remain unbeaten in the national league after being held to a draw by guiseley.", "label": "CORRECT"}
```

### Format
The file is in the format of jsonl: every line is a json dict. The most important key for one sample is the text, claim and label. Text is the input document and the claim is the sentence from summary, the label represents whether the claim contains factuality error compared with the text.

### Split
There are six base test sets (orig) and the corresponding 24 fine-grained test sets (AntoSub, EntRep, NumEdit and SynPrun test sets for every base test set), the detail construct process can refer to the original paper.


## Reference
 
