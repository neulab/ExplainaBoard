# Submission Format of System's Output

For each task, we give an example file to show how output files should be organized and then submitted by [google form](https://docs.google.com/forms/u/1/d/e/1FAIpQLSdb_3PPRTXXjkl9MWUeVLc8Igw0eI-EtOrU93i6B61X9FRJKg/viewform).



## Samples 

9 tasks and its sample name.

Each task has several columns separated by a tab, and the column description is shown in the following table.
For example, The system's output file of the NER task should contain three columns separated by tab: token, true tag, and predict tag.

`Note: `

- The system's output file for the sequence labeling tasks (NER, POS-tagging, Chunking, CWS) should have the sentence boundary like the original testing set.

| Task             				  | Sample           | Description                  |
|---------------------------------|------------------|------------------------------|  
| Named Entity Recognition   	  | sample-ner.tsv   | token, true tag, predict tag |
| Part-of-Speech Tagging          | sample-pos.tsv   | token, true tag, predict tag |
| Chinese Word Segmentation 	  | sample-cws.tsv   | token, true tag, predict tag |
| Text Chunking 				  | sample-chunk.tsv | token, true tag, predict tag |
| Text Classification       	  | sample-tc.tsv    | sentence, true tag, predict tag, prediction probability | 
| Aspect Sentiment Classification | sample-absa.tsv  | sentence 1, sentence 2, true tag, predict tag |
| Natural Language Inference      | sample-nli.tsv   | sentence 1, sentence 2, true tag, predict tag, prediction probability | 
| Semantic Parsing                | sample-semp.csv  | db_id, question, true query, predict query |
| Text Summarization              | sample-summ.tsv  | 
