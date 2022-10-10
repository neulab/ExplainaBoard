# File Formats Supported by Different Tasks

Here is the list of the file formats supported by different tasks.

For dataset files, if your datasets have been supported
by [datalab](https://github.com/ExpressAI/DataLab/tree/main/datasets),
you fortunately don't need to prepare the dataset.
Otherwise, you can upload your custom datasets in the supported formats.

You may refer to the example files for more specific formats,
    or click on the task link for more explanation.

To upload custom features and analysis, please refer to this [instruction](./add_custom_features.md).

| Task                | File Type   | File Format | Example File |
|---------------------|-------------|-------------|--------------|
| [conditional generation (machine translation/summarization)](./task_conditional_generation.md) | dataset | TSV | [cnndm_mini-dataset.tsv](../data/system_outputs/cnndm/cnndm_mini-dataset.tsv) |
|                     |        | JSON | [conala-dataset.json](../data/system_outputs/conala/conala-dataset.json) |
|                     | output | JSON | [conala-baseline-output.json](../data/system_outputs/conala/conala-baseline-output.json) |
|                     |        | TXT | [cnndm_mini-bart-output.txt](../data/system_outputs/cnndm/cnndm_mini-bart-output.txt) |
| [text classification](./task_text_classification.md) | dataset | TSV | [sst2-dataset.tsv](../data/system_outputs/sst2/sst2-dataset.tsv) |
|                     |        | JSON | [text-classification-dataset.json](../integration_tests/artifacts/text_classification/dataset.json) |
|                     | output | JSON | [text-classification-output.json](../integration_tests/artifacts/text_classification/output_user_metadata.json) |
|                     |        | TXT | [sst2-lstm-output.txt](../data/system_outputs/sst2/sst2-lstm-output.txt) |
| sequence labeling (NER/word segmentation/chunking) | dataset | CoNLL | [conll2003-dataset.conll](../data/system_outputs/conll2003/conll2003-dataset.conll) |
|                     | output | CoNLL | [conll2003-elmo-output.conll](../data/system_outputs/conll2003/conll2003-elmo-output.conll) |
|                     |        | JSON | |
| cloze multiple choice | dataset | JSON |  |
|                     | output | JSON |  |
| cloze generative | dataset | JSON |  |
|                     | output | JSON |  |
| [QA (extractive)](./task_extractive_qa.md) | dataset | JSON | [squad_mini-dataset.json](../data/system_outputs/squad/squad_mini-dataset.json) |
|                     | output | JSON | [squad_mini-example-output.json](../data/system_outputs/squad/squad_mini-example-output.json) |
| [QA (MCQ)](./task_qa_multiple_choice.md) | dataset | JSON | [fig_qa-dataset.json](../data/system_outputs/fig_qa/fig_qa-dataset.json) |
|                     | output | JSON | [fig_qa-bert-output.json](../data/system_outputs/fig_qa/fig_qa-bert-output.json) |
| [QA (open domain)](./task_qa_open_domain.md) | dataset | JSON | |
|                     | output | TXT | [test.dpr.nq.txt](../data/system_outputs/qa_open_domain/test.dpr.nq.txt) |
| [aspect-based sentiment analysis](./task_aspect_based_sentiment_classification.md) | dataset | TSV | [absa-dataset.tsv](../data/system_outputs/absa/absa-dataset.tsv) |
|                     |        | JSON | |
|                     | output | JSON | [absa-example-output-confidence.json](../data/system_outputs/absa/absa-example-output-confidence.json) |
|                     |        | TXT | [absa-example-output.txt](../data/system_outputs/absa/absa-example-output.txt) |
| [grammatical error correction](./task_grammatical_error_correction.md) | dataset | JSON |  |
|                     | output | JSON |  |
| [text pair classification](./task_text_pair_classification.md) | dataset | TSV | [snli-dataset.tsv](../data/system_outputs/snli/snli-dataset.tsv) |
|                     |        | JSON |  |
|                     | output | JSON |  |
|                     |        | TXT | [snli-roberta-output.txt](../data/system_outputs/snli/snli-roberta-output.txt) |
| [knowledge graph link tail prediction](./task_kg_link_tail_prediction.md) | dataset | JSON |  |
|                     | output | JSON |  |
| language modeling | dataset | JSON |  |
|                     |        | TXT |  |
|                     | output | JSON |  |
|                     |        | TXT |  |
| tabular classification | dataset | JSON | [sst2-tabclass-dataset.json](../data/system_outputs/sst2_tabclass/sst2-tabclass-dataset.json) |
|                     | output | JSON |  |
|                     |        | TXT |  |
| tabular regression | dataset | JSON | [sst2-tabreg-dataset.json](../data/system_outputs/sst2_tabreg/sst2-tabreg-dataset.json) |
|                     | output | JSON |  |
|                     |        | TXT | [sst2-tabreg-lstm-output.txt](../data/system_outputs/sst2_tabreg/sst2-tabreg-lstm-output.txt) |
