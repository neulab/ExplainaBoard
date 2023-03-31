# Supported Metrics

| Metric                  | Description                                                                                                                                                                                                                                                                       | Used Tasks                                                                                                                       | Ref URL                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| rouge1                  | ROUGE-1 refers to the overlap of unigram (each word) between the system and reference summaries.                                                                                                                                                                                  | Summarization, Conditional generation                                                                                            | [link](https://aclanthology.org/W04-1013.pdf)                                                                    |
| rouge2                  | ROUGE-2 refers to the overlap of bigrams between the system and reference summaries.                                                                                                                                                                                              | Summarization, Conditional generation                                                                                            | [link](https://aclanthology.org/W04-1013.pdf)                                                                    |
| rougeL                  | ROUGE-L refers to the longest common subsequence between the system and reference summaries.                                                                                                                                                                                      | Summarization, Conditional generation                                                                                            | [link](https://aclanthology.org/W04-1013.pdf)                                                                    |
| prism_qe                | PRISM for quality estimation. It calculates Score(hypothesis\| source)                                                                                                                                                                                                            | Machine translation, Conditional generation                                                                                      | [link](https://arxiv.org/abs/2004.14564)                                                                         |
| prism                   | PRISM is a sequence to sequence framework trained from scratch. prism calculates the average generation score of Score(hypothesis\|reference) and Score(reference\|hypothesis).                                                                                                   | Machine translation, Conditional generation                                                                                      | [link](https://arxiv.org/abs/2004.14564)                                                                         |
| mover_score             | MoverScore is a metric similar to BERTScore. Different from BERTScore, it uses the Earth Mover’s Distance instead of the Euclidean Distance.                                                                                                                                      | Summarization, Conditional generation                                                                                            | [link](https://aclanthology.org/D19-1053.pdf)                                                                    |
| comet_qe                | COMET for quality estimation. comet_qe uses the wmt20-comet-qe-da checkpoint which utilizes only source and hypothesis.                                                                                                                                                           | Machine translation, Conditional generation                                                                                      | [link](https://aclanthology.org/2020.emnlp-main.213.pdf)                                                         |
| comet                   | COMET is a neural framework for training multilingual machine translation evaluation models. comet uses the wmt20-comet-da checkpoint which utilizes source, hypothesis and reference.                                                                                            | Machine translation, Conditional generation                                                                                      | [link](https://aclanthology.org/2020.emnlp-main.213.pdf)                                                         |
| chrf                    | CHRF measures the character-level ngram matches between hypothesis and reference.                                                                                                                                                                                                 | Machine translation, Conditional generation                                                                                      | [link](https://aclanthology.org/W15-3049.pdf)                                                                    |
| bleu                    | BLEU measures modified ngram matches between each candidate translation and the reference translations.                                                                                                                                                                           | Machine translation, Summarization, Conditional generation                                                                       | [link](https://aclanthology.org/P02-1040.pdf)                                                                    |
| bert_score_f            | BERTScore f score.                                                                                                                                                                                                                                                                | Machine translation, Summarization, Conditional generation                                                                       | [link](https://arxiv.org/pdf/1904.09675.pdf)                                                                     |
| bert_score_r            | BERTScore recall.                                                                                                                                                                                                                                                                 | Machine translation, Summarization, Conditional generation                                                                       | [link](https://arxiv.org/pdf/1904.09675.pdf)                                                                     |
| bert_score_p            | BERTScore is a metric designed for evaluating translated text using BERT-based matching framework. bert_score_p calculates the BERTScore precision.                                                                                                                               | Machine translation, Summarization, Conditional generation                                                                       | [link](https://arxiv.org/pdf/1904.09675.pdf)                                                                     |
| bart_score_en_src       | BARTScore using the CNNDM finetuned BART. It calculates Score(hypothesis\|source).                                                                                                                                                                                                | Machine translation, Summarization, Conditional generation                                                                       | [link](https://arxiv.org/abs/2106.11520)                                                                         |
| bart_score_en_ref       | BARTScore is a sequence to sequence framework based on pre-trained language model BART.                                                                                                                                                                                           | Machine translation, Summarization, Conditional generation                                                                       | [link](https://arxiv.org/abs/2106.11520)                                                                         |
| bart_score_cnn_hypo_ref | BARTScore using the CNNDM finetuned BART. It calculates the average generation score of Score(hypothesis\|reference) and Score(reference\|hypothesis).                                                                                                                            | Machine translation, Summarization, Conditional generation                                                                       | [link](https://arxiv.org/abs/2106.11520)                                                                         |
| bart_score_summ         | BARTScore is a sequence to sequence framework based on pre-trained language model BART. For this metric, BART is finetuned on the summarization data: CNN-Dailymail. It calculates the average generation score of Score(hypothesis\|reference) and Score(reference\|hypothesis). | Summarization, Conditional generation                                                                                            |                                                                                                                  |
| bart_score_mt           | BARTScore is a sequence to sequence framework based on pre-trained language model BART. For this metric, BART is finetuned on the paraphrase data: ParaBank. It calculates the average generation score of Score(hypothesis\|reference) and Score(reference\|hypothesis).         | Machine translation, Conditional generation                                                                                      |                                                                                                                  |
| length                  | The length of generated text.                                                                                                                                                                                                                                                     | Machine translation, Summarization, Conditional generation                                                                       | [link](https://github.com/mjpost/sacrebleu)                                                                      |
| length_ratio            | The ratio between the length of generated text and gold reference                                                                                                                                                                                                                 | Machine translation, Summarization, Conditional generation                                                                       | [link](https://github.com/mjpost/sacrebleu)                                                                      |
| Accuracy                | Percentage of all correctly classified samples                                                                                                                                                                                                                                    | Text classification, QA multiple choice, Aspect-based sentiment classification, Text pair classification, Tabular classification | [link](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)  |
| F1                      | Harmonic mean of precision and recall                                                                                                                                                                                                                                             | Named entity recognition, Word segmentation, Chunking, Cloze multiple choice, QA extractive, QA open domain                      | [link](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)  |
| CorrectCount            | The number of correctly predicted (e.g., classified) samples                                                                                                                                                                                                                      | Cloze multiple choice, Cloze generative, QA multiple choice                                                                      | [link](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)/ |
| Hits1                   | It is the count of how many positive samples are ranked in the top-1 positions against a bunch of negatives.                                                                                                                                                                      | KG link prediction                                                                                                               | [link](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)           |
| Hits2                   | It is the count of how many positive samples are ranked in the top-2 positions against a bunch of negatives.                                                                                                                                                                      | KG link prediction                                                                                                               | [link](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)           |
| Hits3                   | It is the count of how many positive samples are ranked in the top-3 positions against a bunch of negatives.                                                                                                                                                                      | KG link prediction                                                                                                               | [link](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)           |
| Hits4                   | It is the count of how many positive samples are ranked in the top-4 positions against a bunch of negatives.                                                                                                                                                                      | KG link prediction                                                                                                               | [link](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)           |
| Hits5                   | It is the count of how many positive samples are ranked in the top-5 positions against a bunch of negatives.                                                                                                                                                                      | KG link prediction                                                                                                               | [link](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)           |
| Hits10                  | It is the count of how many positive samples are ranked in the top-10 positions against a bunch of negatives.                                                                                                                                                                     | KG link prediction                                                                                                               | [link](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)           |
| MRR                     | Mean Reciprocal Rank is a measure to evaluate systems that return a ranked list of answers to queries.                                                                                                                                                                            | KG link prediction                                                                                                               | [link](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)                                                       |
| MR                      | The mean rank of the true output in a predicted n-best list                                                                                                                                                                                                                       | KG link prediction                                                                                                               |                                                                                                                  |
| Perplexity              | Perplexity is a measurement of how well a probability distribution or probability model predicts a sample.                                                                                                                                                                        | Language modeling                                                                                                                | [link](https://en.wikipedia.org/wiki/Perplexity)                                                                 |
| LogProb                 | A logarithm of a probability.                                                                                                                                                                                                                                                     | Language modeling                                                                                                                | [link](https://en.wikipedia.org/wiki/Log_probability)                                                            |
| ExactMatch              | The Exact Match metric measures the percentage of predictions that match any one of the ground truth answers exactly                                                                                                                                                              | QA extractive, QA open domain                                                                                                    | [link](https://rajpurkar.github.io/mlx/qa-and-squad/)                                                            |
| LikertScore_fluency     | Human evaluation metric for the fluency of texts with likert style                                                                                                                                                                                                                | Machine translation, Summarization, Conditional generation                                                                       |                                                                                                                  |
| LikertScore_coherence   | Human evaluation metric for the coherence of texts with likert style                                                                                                                                                                                                              | Machine translation, Summarization, Conditional generation                                                                       |                                                                                                                  |
| LikertScore_factuality  | Human evaluation metric for the factuality of texts with likert style                                                                                                                                                                                                             | Machine translation, Summarization, Conditional generation                                                                       |                                                                                                                  |
| SeqCorrectCount         | The number of correctly predicted spans in a sequence.                                                                                                                                                                                                                            | Grammatical error correction                                                                                                     |                                                                                                                  |
| RMSE                    | Root mean square error (RMSE) measures the the difference between values predicted by the model and the values observed.                                                                                                                                                          | Tabular regression                                                                                                               | [link](https://en.wikipedia.org/wiki/Root-mean-square_deviation)                                                 |
| Absolute Error          | Absolute error is the absolute discrepancy between the prediction and true value.                                                                                                                                                                                                 | Tabular regression                                                                                                               | [link](https://en.wikipedia.org/wiki/Approximation_error)                                                        |