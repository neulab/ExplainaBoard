# System output


## Model Description

We fine-tuned a pretrained [mT5](https://www.aclweb.org/anthology/2021.naacl-main.41/) with XL-Sum with 512 token inputs and 84 token outputs for 50k steps. We used a batch size of 256 samples and 0.5 upscaling factor with inverse square root learning rate schedule. Details of the training hyperparameters can be found [here](https://github.com/csebuetnlp/xl-sum/blob/master/seq2seq/README.md).

## Meta Data
* Download link for system output: [This link](https://docs.google.com/uc?export=download&id=16OaTryW-QCaYevQ9Gtbr7aKhCJKBDEAr)
* Github Repo: https://github.com/csebuetnlp/xl-sum
* Paper: [**"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"**](http://arxiv.org/abs/2106.13822)
* Corresponding Author: [Rifat Shahriyar](http://rifatshahriyar.github.io/)

## Results

Language | ROUGE-1 / ROUGE-2 / ROUGE-L
---------|----------------------------
Amharic | 20.0485 / 7.4111 / 18.0753
Arabic | 34.9107 / 14.7937 / 29.1623
Azerbaijani | 21.4227 / 9.5214 / 19.3331
Bengali | 29.5653 / 12.1095 / 25.1315
Burmese | 15.9626 / 5.1477 / 14.1819
Chinese (Simplified) | 39.4071 / 17.7913 / 33.406
Chinese (Traditional) | 37.1866 / 17.1432 / 31.6184
English | 37.601 / 15.1536 / 29.8817
French | 35.3398 / 16.1739 / 28.2041
Gujarati | 21.9619 / 7.7417 / 19.86
Hausa | 39.4375 / 17.6786 / 31.6667
Hindi | 38.5882 / 16.8802 / 32.0132
Igbo | 31.6148 / 10.1605 / 24.5309
Indonesian | 37.0049 / 17.0181 / 30.7561
Japanese | 48.1544 / 23.8482 / 37.3636
Kirundi | 31.9907 / 14.3685 / 25.8305
Korean | 23.6745 / 11.4478 / 22.3619
Kyrgyz | 18.3751 / 7.9608 / 16.5033
Marathi | 22.0141 / 9.5439 / 19.9208
Nepali | 26.6547 / 10.2479 / 24.2847
Oromo | 18.7025 / 6.1694 / 16.1862
Pashto | 38.4743 / 15.5475 / 31.9065
Persian | 36.9425 / 16.1934 / 30.0701
Pidgin | 37.9574 / 15.1234 / 29.872
Portuguese | 37.1676 / 15.9022 / 28.5586
Punjabi | 30.6973 / 12.2058 / 25.515
Russian | 32.2164 / 13.6386 / 26.1689
Scottish Gaelic | 29.0231 / 10.9893 / 22.8814
Serbian (Cyrillic) | 23.7841 / 7.9816 / 20.1379
Serbian (Latin) | 21.6443 / 6.6573 / 18.2336
Sinhala | 27.2901 / 13.3815 / 23.4699
Somali | 31.5563 / 11.5818 / 24.2232
Spanish | 31.5071 / 11.8767 / 24.0746
Swahili | 37.6673 / 17.8534 / 30.9146
Tamil | 24.3326 / 11.0553 / 22.0741
Telugu | 19.8571 / 7.0337 / 17.6101
Thai | 37.3951 / 17.275 / 28.8796
Tigrinya | 25.321 / 8.0157 / 21.1729
Turkish | 32.9304 / 15.5709 / 29.2622
Ukrainian | 23.9908 / 10.1431 / 20.9199
Urdu | 39.5579 / 18.3733 / 32.8442
Uzbek | 16.8281 / 6.3406 / 15.4055
Vietnamese | 32.8826 / 16.2247 / 26.0844
Welsh | 32.6599 / 11.596 / 26.1164
Yoruba | 31.6595 / 11.6599 / 25.0898

## Reference

```
@inproceedings{hasan-etal-2021-xlsum,
    title = "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md Saiful and
      Samin, Kazi  and
      Li, Yuan-Fang and
      Kang, Yong-Bin and 
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    month = "August",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/2106.13822"
}
```

