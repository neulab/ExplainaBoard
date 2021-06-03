# Output Format for API Use

### Text Classification
```bash
sentence \t true_label \t predicted_label
```
Although the above example file of text classification show multiple columns, but if not aiming to print the ``calibration`` feature, the above three columns are sufficient.


### Tagging
```bash
token \t true_label \t predicted_label
```
Specific tasks including Named Entity Recognition (NER), Chinese Word Segmentation (CWS), Text Chunking (chunk), Part-of-Speech (POS).



### Natural Language Inference
```bash
sentence1 \t sentence2 \t true_label \t predicted_label
```
