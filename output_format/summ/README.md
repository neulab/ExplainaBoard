# Submission Format of System's Outputs

The submission format of system's output for the Summarization task should include three columns separated by a tab.

- `1 column`: the source document;
- `2 column`: the reference summary;
- `3 column`: the system-generated summary;

The ROUGE scores of the submitted outputs will be automatically computed. 

For CNN/DM and XSum datasets, the re-ranking scores of the outputs from the re-ranking model will also be computed.



# Downloading Format of System's Outputs

Users can also download system outputs that we have provided (with rich post-processing information).

- `1 column`: the source document;
- `1 column`: the reference summary;
- `3 column`: the system-generated summary;
- `4 column`: sample-level ROUGE-1;
- `5 column`: sample-level ROUGE-2;
- `6 column`: sample-level ROUGE-L;
- `7 column`: corpus-level ROUGE-1;
- `8 column`: corpus-level ROUGE-2;
- `9 column`: corpus-level ROUGE-L;
- `10 column`: refactoring score of this sample based on the [work](https://arxiv.org/pdf/2104.07210.pdf);
