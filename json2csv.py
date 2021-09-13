#conding=utf8
# /usr2/home/pliu3/data/InterpretEval/task-panx/genPreComputed
import json

import numpy as np
import os
import argparse


dict_model2url = {
    "ABS-ConvS2S": "https://drive.google.com/drive/folders/1AfPPyKer83x5gcSHnVlmVDoj_kx9jVSZ?usp=sharing",
    "tConvS2S": "https://drive.google.com/drive/folders/1svxHYvtV_wCOZpsoV7V-NHFTme3LHzJa?usp=sharing",
    "Ext-Oracle": "https://drive.google.com/drive/folders/1uhV9eyBLJ0dPx1grRw7zAG4Mdo6eQgN0?usp=sharing",
    "Ext-Lead": "https://drive.google.com/drive/folders/1ga9La5dz36sR-XVoTwHyrS34IiZqU7A3?usp=sharing",
    "ABS-PreSumm-Mix": "https://drive.google.com/drive/folders/1w1JFDUObvLLhCDvWC-JY5mooCiUySD6g?usp=sharing",
    "ABS-SemSim": "https://drive.google.com/drive/folders/1LDtaQcfKyc6ERDtdW1wjnYy2Mg0IRGZO?usp=sharing",
    "ABS-T5-11B": "https://drive.google.com/drive/folders/1IpCEkBJqkBu5hkyVbjj1IhqiK28ZaXII?usp=sharing",
    "ABS-UniLM-v2": "https://drive.google.com/drive/folders/1E_yBKCWH9G3BpPRi7xdiEps9Em26zdAa?usp=sharing",
    "EXT-MatchSumm": "https://drive.google.com/drive/folders/1lQAqekRFhlbhlBRc-S0cgfvbt3hbAx3O?usp=sharing",
    "ABS-BART": "https://drive.google.com/drive/folders/1k1eROTpSe9cvoKLT1v3wXd_BRKME9ROn?usp=sharing",
    "ABS-T5-3B": "https://drive.google.com/drive/folders/1rmXshfvsGwIlFgKS39zxh7_B_gJ5XGbc?usp=sharing",
    "ABS-T5-Large": "https://drive.google.com/drive/folders/1WfHPgx6o4jGF3riwYTUsejmScZjo888p?usp=sharing",
    "ABS-T5-Base": "https://drive.google.com/drive/folders/1tYwODVg04lkaGSBPUkIuZL13Ndn00Df4?usp=sharing",
    "ABS-UniLM-v1": "https://drive.google.com/drive/folders/1x7q0kl8BwpeS3Fa-uFapY_CfJDZhelTT?usp=sharing",
    "EXT-PNBERT-RL": "https://drive.google.com/drive/folders/1MPaTyn8m32hhaFqWXfT9g1aVNa2R0twa?usp=sharing",
    "ABS-T5-Small": "https://drive.google.com/drive/folders/1IG19raVnp8XDcXWlptpRYg5U3gefNxgW?usp=sharing",
    "EXT-PNBERT": "https://drive.google.com/drive/folders/1d9niDlWk8H9E0dml895B4vpmgCRoFuAv?usp=sharing",
    "EXT-HeteGraph": "https://drive.google.com/drive/folders/1nYhOZy5buTvCYclZLvWiWSeRLHnfi9g3?usp=sharing",
    "ABS-PreSumm": "https://drive.google.com/drive/folders/1Y-mBljZuGvkZTGdcdBfHYZiKdoZX-vCu?usp=sharing",
    "ABS-TwoStage-RL": "https://drive.google.com/drive/folders/1_qWbvpf1ejw2cyU36O_9-jz99JPACd1_?usp=sharing",
    "EXT-BanditSumm": "https://drive.google.com/drive/folders/12Fjh0Fz6HmLbPDxfBDvX2bS8rS9kXk_I?usp=sharing",
    "ABS-BottomUp": "https://drive.google.com/drive/folders/1vgNd-uEceL2ISEsIHOZlOR911FJNuAf7?usp=sharing",
    "ABS-Neusumm": "https://drive.google.com/drive/folders/1Rq3zMxhOfcs4TCL-btU7M4_CpDYJBs3y?usp=sharing",
    "ABS-FastAbs-RL": "https://drive.google.com/drive/folders/12SZS6rtgyB3UCJcmPhS-TNAlUBgHRDVI?usp=sharing",
    "ABS-PreSumm-Trans": "https://drive.google.com/drive/folders/1PNSbjCn0bYOi2qL8bk2ZwLQOTvIDolZA?usp=sharing",
    "ABS-FastAbs-RL-Rerank": "https://drive.google.com/drive/folders/1xTM-DA1ELiPN-PL9YwOlPNpUPWrmx5it?usp=sharing",
    "EXT-Refresh": "https://drive.google.com/drive/folders/1HYezLmyRA3K4E0JzBfgo8R85Ye5iKCHH?usp=sharing",
    "ABS-Ptr-Generator-Gen-Cov": "https://drive.google.com/drive/folders/1GZTHTjg7aaUvauEz3CduliemM8a9lpRe?usp=sharing",
    "ABS-Ptr-Generator-Gen": "https://drive.google.com/drive/folders/1CmLyjx34haGxRz7QDwVQq-XhSCoOMn3j?usp=sharing",
    "ABS-Ptr-Generator-Baseline": "https://drive.google.com/drive/folders/1AJTAMRCFoZtVzJ5TFdZuT06EziChNWlU?usp=sharing",
    "ABS-GSum": "https://drive.google.com/drive/folders/1lfGRNkP0dxb9oRlIi0hO4zvYG_hUtcX8?usp=sharing",
    "ABS-BART-Rerank": "https://drive.google.com/drive/folders/1wEONGoV5a2_5HPo5-Cw6vWh7GxzENjaS?usp=sharing",
    "ABS-Refactor":"https://drive.google.com/drive/folders/1Qgkphp1UEjlLfMPZ85h7LwpeVwfLBN0-?usp=sharing",
    "ABS-PEGASUS":"https://drive.google.com/drive/folders/1tpsI1aBMHaj8h_eOLWqvXDaCkmqmqEaF?usp=sharing",

}

dict_task2metric = {
  'ner': "F1",
  'pos': "Accuracy",
  'tc': "Accuracy",
  'nli': "Accuracy",
  'summ': "ROUGE",
  'chunk': "F1",
  'cws': "F1",
  'absa': "Accuracy",
  'mt':'BLEU',
}







dict_map_task = {"ner":"Named Entity Recognition",
    "chunk":"Chunking",
    "cws":"Chinese Word Segmentation",
    "pos":"Part-of-Speech",
    "tc":"Text Classification",
    "summ":"Summarization",
    "nli":"Natural Language Inference",
    "absa":"Aspect-based Sentiment Analysis",
    'mt': 'Machine Translation',

}

dict_map_model = {
    "CbertWnon_snonMlp_larger":"CbertWnon_snonMlp_larger",
    "CcnnWglove_lstmCrf_larger":"CcnnWglove_lstmCrf_larger",
    "CbertWnon_snonMlp":"CbertWnon_snonMlp",
    "CcnnWglove_lstmMlp":"CcnnWglove_lstmMlp",
    "CelmoWglove_lstmCrf":"ELMo+GloVe",
    "CflairWnone_lstmCrf":"CflairWnone_lstmCrf",
    "luke":"LUKE",
    "CcnnWglove_cnnCrf":"CcnnWglove_cnnCrf",
    "CcnnWnone_lstmCrf":"CcnnWnone_lstmCrf",
    "CelmoWnone_lstmCrf":"ELMo",
    "CnoneWrand_lstmCrf":"CnoneWrand_lstmCrf",
    "roberta_context":"FLERT-RoBERTa",
    "CcnnWglove_lstmCrf":"CcnnWglove_lstmCrf",
    "CcnnWrand_lstmCrf":"CcnnWrand_lstmCrf",
    "CflairWglove_lstmCrf":"FLAIR+GLoVe",
    "crfpp_results":"crfpp_results",
    "xlmr_context":"FLERT-XLMR",
    "rnn": "RNN",
    "lstm": "LSTM",
    "cnn": "CNN",
    "banditsumm": "Bandit",
    "bart": "BART",
    "bertabs": "BERTAbs",
    "t5": "T5",
    "unilmv1": "UniLM-v1",
    "unilmv2": "UniLM-v2",
    "gcn": "Graph Neural Networks",
    "roberta": "RoBERTa",
    "CbertBnonLstmMlp":"CbertBnonLstmMlp",
    "CelmBnonLstmMlp":"CelmBnonLstmMlp",
    "Cw2vBavgCnnCrf":"Cw2vBavgCnnCrf",
    "Cw2vBavgLstmMlp":"Cw2vBavgLstmMlp",
    "CbertBw2vLstmMlp":"CbertBw2vLstmMlp",
    "CrandBavgLstmCrf":"CrandBavgLstmCrf",
    "Cw2vBavgLstmCrf":"Cw2vBavgLstmCrf",
    "Cw2vBw2vLstmCrf":"Cw2vBw2vLstmCrf",
    "bert":"BERT",
    "CbertWnon_lstmCrf":"BERT",
    "CbertWglove_lstmCrf": "BERT+GLoVe",
    "CflairWglove_lstmCrf":"FLAIR+GLoVe",
    "CflairWnon_lstmCrf": "FLAIR",
    "CbertWnon_snonMlp_larger": "BERT+LargerContext",
    "CcnnWglove_lstmCrf_larger": "GLoVe+LargerContext",
    "dpcnn": "DPCNN",
    "lstm-selfattention": "LSTM-SelfAttention",
    "": "",
    "abs-semsim":"ABS-SemSim",
    "abs-t5_11B":"ABS-T5-11B",
    "abs-unilm_v2":"ABS-UniLM-v2",
    "ext-matchsumm":"EXT-MatchSum",
    "abs-bart":"ABS-BART",
    "abs-t5_3B":"ABS-T5-3B",
    "abs-t5_large":"ABS-T5-Large",
    "abs-t5_base":"ABS-T5-Base",
    "abs-unilm_v1":"ABS-UniLM-v1",
    "ext-pnbert_pn_rl":"EXT-PNBERT-RL",
    "abs-t5_small":"ABS-T5-Small",
    "ext-pnbert_pn":"EXT-PNBERT",
    "ext-heter_graph":"EXT-HeteGraph",
    "abs-presumm_abs":"ABS-PreSumm",
    "abs-presumm_ext_abs":"ABS-PreSumm-Mix",
    "abs-two_stage_rl":"ABS-TwoStage_RL",
    "ext-banditsumm":"EXT-BanditSumm",
    "abs-bottom_up":"ABS-BottomUp",
    "abs-neusumm":"ABS-NeuSumm",
    "abs-fast_abs_rl":"ABS-FastAbs-RL",
    "abs-presumm_trans_abs":"ABS-PreSumm-Trans",
    "abs-fast_abs_rl_rerank":"ABS-FastAbs-RL-Rerank",
    "ext-refresh":"EXT-Refresh",
    "abs-ptr_generator_gen_cov":"ABS-Ptr-Generator-Gen-Cov",
    "abs-ptr_generator_gen":"ABS-Ptr-Generator-Gen",
    "abs-ptr_generator_baseline":"ABS-Ptr-Generator-Baseline",
    "abs-gsum":"ABS-GSum",
    "abs-bart-reranked":"ABS-BART-Rerank",
    "abs-convs2s":"ABS-ConvS2S",
    "abs-ext_oracle":"Ext-Oracle",
    "abs-lead":"Ext-Lead",
    "abs-pgn":"ABS-Ptr-Generator-Baseline",
    "abs-tconvs2s":"tConvS2S",
    "abs-gsum-reranked":"ABS-Refactor",
    "abs-pegasus":"ABS-PEGASUS",
    "transformer":"Transformer",
}

dict_map_dataset={
    "conll00":"CoNLL-2000",
    "as":"AS",
    "cityu":"CityU",
    "ckip":"CKIP",
    "ctb":"CTB",
    "msr":"MSR",
    "ncc":"NCC",
    "pku":"PKU",
    "sxu":"SXU",
    "conll03":"CoNLL-2003",
    "notebc":"Ontonotes5-BC",
    "notebn":"Ontonotes5-BN",
    "notemz":"Ontonotes5-MZ",
    "notewb":"Ontonotes5-WB",
    "wnut16":"WNUT-2016",
    "ptb2":"PTB2",
    "cnndm":"CNNDM",
    "xsum":"XSum",
    "nyt":"NYT",
    "ag_news":"AG News",
    "atis":"ATIS",
    "imdb":"IMDB",
    "yahoo_answers_topics":"Yahoo answer",
    "amazon_polarity":"Amazon",
    "dbpedia_14":"Dbpedia-14",
    "sogou_news":"Sogou",
    "cr": "CR",
    "mr": "MR",
    "sem_eval_2014_task_1":"SICK",
    "snli":"SNLI",
    "twitter": "Twitter",
    "laptop": "Laptop",
    "rest14": "Restaurant-2014",
    "rest16": "Restaurant-2016",
    "sms_spam":"SMS_Spam",
    "ade":"ADE",
    "fdu":"FDU-MTL16",
    "rotten_tomatoes":"Rotten_Tomato",
    "sst2":"SST2",
    "sem_eval_2014-task_1":"SICK",
    "fdu-mtl":"FDU-MTL16",
    "geo":"GEO",
    "wikisql":"WikiSQL",
    "spider":"Spider",
    "overnight":"Overnight",
}


dict_model2title = {
    "BERT+LargerContext":"Larger-Context Tagging: When and Why Does It Work?",
    "GLoVe+LargerContext":"Larger-Context Tagging: When and Why Does It Work?",
    "DPCNN":"Deep Pyramid Convolutional Neural Networks for Text Categorization",
    "LSTM-SelfAttention":"A structured self-attentive sentence embedding",
    "CNTN":"Siamese Convolutional Networks for Cognate Identification",
    "ABS-ConvS2S":"Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    "tConvS2S":"Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    "Ext-Oracle":"Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    "Ext-Lead":"Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    "ABS-PreSumm-Mix":"Text Summarization with Pretrained Encoders",
    "":"",
	"BERT": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "RoBERTa": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "Graph Neural Networks":"Semi-Supervised Classification with Graph Convolutional Networks",
    "LSTM":"Long short-term memory",
    "Bridge":"Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing",
    "Editsql":"Editing-Based SQL Query Generation for Cross-Domain Context-Dependent Questions",
    "TranX-Execution-Guided-Decoding":"TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation",
    "TranX":"TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation",
    "Coarse2fine":"Coarse-to-Fine Decoding for Neural Semantic Parsing",
    "Seq2seq-attention":"Semantic Parsing with Dual Learning",
    "lstm-selfattention":"A structured self-attentive sentence embedding",
    "CNN":"Convolutional Neural Networks for Sentence Classification",
    "FLERT-RoBERTa":"FLERT: Document-Level Features for Named Entity Recognition",
    "FLERT-XLMR":"FLERT: Document-Level Features for Named Entity Recognition",
    "crfpp_results":"Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data",
    "LUKE":"LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention",
    "FLAIR+GLoVe":"Pooled Contextualized Embeddings for Named Entity Recognition",
    "FLAIR": "Pooled Contextualized Embeddings for Named Entity Recognition",
    "ELMo":"Deep contextualized word representations",
    "CbertWnon_snonMlp": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "CflairWnone_lstmCrf": "Pooled Contextualized Embeddings for Named Entity Recognition",
    "BERT": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "CbertWnon_lstmCrf": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "CbertWnon_snonMlp_larger": "Larger-Context Tagging: When and Why Does It Work?",
    "CcnnWglove_lstmCrf_larger": "Larger-Context Tagging: When and Why Does It Work?",
    "ESIM":"Enhanced LSTM for Natural Language Inference",
    "CNTM":"Convolutional Neural Tensor Network Architecture for Community-based Question Answering",
    "ABS-SemSim":"Learning by semantic similarity makes abstractive summarization better",
    "ABS-T5-11B":"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "ABS-UniLM-v2":"Pseudo-Masked Language Models for Unified Language Model Pre-Training",
    "EXT-MatchSumm":"Extractive Summarization as Text Matching",
    "ABS-BART": "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
    "ABS-T5-3B": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "ABS-T5-Large": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "ABS-T5-Base": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "ABS-UniLM-v1": "Unified Language Model Pre-training for Natural Language Understanding and Generation",
    "EXT-PNBERT-RL": "Searching for Effective Neural Extractive Summarization: What Works and What’s Next",
    "ABS-T5-Small": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "EXT-PNBERT": "Searching for Effective Neural Extractive Summarization: What Works and What's Next",
    "EXT-HeteGraph": "Heterogeneous Graph Neural Networks for Extractive Document Summarization",
    "ABS-PreSumm": "Text Summarization with Pretrained Encoders",
    "ABS-TwoStage-RL": "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting",
    "EXT-BanditSumm": "BanditSum: Extractive Summarization as a Contextual Bandit",
    "ABS-BottomUp": "Bottom-Up Abstractive Summarization",
    "ABS-Neusumm": "Neural Document Summarization by Jointly Learning to Score and Select Sentences",
    "ABS-FastAbs-RL": "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting",
    "ABS-PreSumm-Trans": "Text Summarization with Pretrained Encoders",
    "ABS-FastAbs-RL-Rerank": "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting",
    "EXT-Refresh": "Ranking Sentences for Extractive Summarization with Reinforcement Learning",
    "ABS-Ptr-Generator-Gen-Cov": "Get To The Point: Summarization with Pointer-Generator Networks",
    "ABS-Ptr-Generator-Gen": "Get To The Point: Summarization with Pointer-Generator Networks",
    "ABS-Ptr-Generator-Baseline": "Get To The Point: Summarization with Pointer-Generator Networks",
    "ABS-GSum":"GSum: A General Framework for Guided Neural Abstractive Summarization",
    "ABS-BART-Rerank":"RefSum: Refactoring Neural Summarization",
    "ABS-Refactor":"RefSum: Refactoring Neural Summarization",
    "ABS-PEGASUS":"PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization",
    "Transformer":"Attention is All you Need",
}


def format4(p):
    return format(float(p) , '.4g')

# python json2csv-xlsum.py --path /usr2/home/pliu3/data/ExplainaBoard/task-xlsum/output
# python json2csv.py  --path_read /home/ubuntu/ExplainaBoard/report/ner/model1.json --path_write /home/ubuntu/ExplainaBoard_frontend/database/ner/ner.csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Json to CSV')
    parser.add_argument('--path_read', type=str, required=False,help="the type of the task")
    parser.add_argument('--path_write', type=str, required=False,help="the type of the task")
    args = parser.parse_args()
    path_read = args.path_read
    path_write = args.path_write


    # print("path_read","path_write")

    fout_csv = open(path_write, "w")


    dataset_list = os.listdir(path_read)


    idx = 0
    string_metric_names = ""

    for dataset_fullname in dataset_list:
        path_base_output_model_dataset = path_read + "/" + dataset_fullname
        fjson = open(path_base_output_model_dataset, "r")
        json_cont = json.load(fjson)

        task = json_cont["task"]
        dataset = json_cont["data"]["name"]
        language = json_cont["data"]["language"]
        model = json_cont["model"]["name"]
        f1 = json_cont["model"]["results"]["overall"]["performance"]
        databias = json_cont["data"]
        single = json_cont["model"]

        idx += 1
        year = 2020
        conf = 'EMNLP'
        metric = "F1"
        check = ''
        file = 3

        sota = "0"
        if single == '':
            single = 'Null'


        title = "Interpretable Multi-dataset Evaluation for Named Entity Recognition"

        authors = "null"
        url = "https://www.aclweb.org/anthology/2020.emnlp-main.457.pdf"
        bib = "https://www.aclweb.org/anthology/2020.emnlp-main.457.bib"
        f1 = format(float(f1), '.4g')

        fout_csv.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
            idx, year, conf, task, dataset, metric, model, f1, title, authors, url, bib,
            json.dumps(single, ensure_ascii=False),
            json.dumps(databias), file, sota, check))



    fout_csv.close()


