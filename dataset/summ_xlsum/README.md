# Dataset for XL-Sum


## Description

XL-Sum contains a total of 1.35 million article-summary pairs covering 45 languages, making it the largest abstractive text summarization dataset publicly available.


## Meta Data

* Official Homepage: https://github.com/csebuetnlp/xl-sum
* Download link: [This link](https://docs.google.com/uc?export=download&id=1fKxf9jAj0KptzlxUsI3jDbp4XLv_piiD)
* Paper: [**"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"**](http://arxiv.org/abs/2106.13822)
* Github Repo: https://github.com/csebuetnlp/xl-sum
* Corresponding Author: [Rifat Shahriyar](http://rifatshahriyar.github.io/)
* Supported Tasks: Multilingual texts summarizarion, monolingual text summarization
* Languages: 

Language      | ISO 639-1 Code |
--------------|----------------|
Amharic | am |
Arabic | ar |
Azerbaijani | az |
Bengali | bn |
Burmese | my |
Chinese (Simplified) | zh-CN |
Chinese (Traditional) | zh-TW |
English | en |
French | fr |
Gujarati | gu |
Hausa | ha |
Hindi | hi |
Igbo | ig |
Indonesian | id |
Japanese | ja |
Kirundi | rn |
Korean | ko |
Kyrgyz | ky |
Marathi | mr |
Nepali | np |
Oromo | om |
Pashto | ps |
Persian | fa |
Pidgin`*` | n/a |
Portuguese | pt |
Punjabi | pa |
Russian | ru |
Scottish Gaelic | gd |
Serbian (Cyrillic) | sr |
Serbian (Latin) | sr |
Sinhala | si |
Somali | so |
Spanish | es |
Swahili | sw |
Tamil | ta |
Telugu | te |
Thai | th |
Tigrinya | ti |
Turkish | tr |
Ukrainian | uk |
Urdu | ur |
Uzbek | uz |
Vietnamese | vi |
Welsh | cy |
Yoruba | yo |

`*` West African Pidgin English


## Data Structure

### Example

```
  {
    "id": "technology-17657859",
    "url": "https://www.bbc.com/news/technology-17657859",
    "title": "Yahoo files e-book advert system patent applications",
    "summary": "Yahoo has signalled it is investigating e-book adverts as a way to stimulate its earnings.",
    "text": "Yahoo's patents suggest users could weigh the type of ads against the sizes of discount before purchase. It says in two US patent applications that ads for digital book readers have been \"less than optimal\" to date. The filings suggest that users could be offered titles at a variety of prices depending on the ads' prominence They add that the products shown could be determined by the type of book being read, or even the contents of a specific chapter, phrase or word. The paperwork was published by the US Patent and Trademark Office late last week and relates to work carried out at the firm's headquarters in Sunnyvale, California. \"Greater levels of advertising, which may be more valuable to an advertiser and potentially more distracting to an e-book reader, may warrant higher discounts,\" it states. Free books It suggests users could be offered ads as hyperlinks based within the book's text, in-laid text or even \"dynamic content\" such as video. Another idea suggests boxes at the bottom of a page could trail later chapters or quotes saying \"brought to you by Company A\". It adds that the more willing the customer is to see the ads, the greater the potential discount. \"Higher frequencies... may even be great enough to allow the e-book to be obtained for free,\" it states. The authors write that the type of ad could influence the value of the discount, with \"lower class advertising... such as teeth whitener advertisements\" offering a cheaper price than \"high\" or \"middle class\" adverts, for things like pizza. The inventors also suggest that ads could be linked to the mood or emotional state the reader is in as a they progress through a title. For example, they say if characters fall in love or show affection during a chapter, then ads for flowers or entertainment could be triggered. The patents also suggest this could applied to children's books - giving the Tom Hanks animated film Polar Express as an example. It says a scene showing a waiter giving the protagonists hot drinks \"may be an excellent opportunity to show an advertisement for hot cocoa, or a branded chocolate bar\". Another example states: \"If the setting includes young characters, a Coke advertisement could be provided, inviting the reader to enjoy a glass of Coke with his book, and providing a graphic of a cool glass.\" It adds that such targeting could be further enhanced by taking account of previous titles the owner has bought. 'Advertising-free zone' At present, several Amazon and Kobo e-book readers offer full-screen adverts when the device is switched off and show smaller ads on their menu screens, but the main text of the titles remains free of marketing. Yahoo does not currently provide ads to these devices, and a move into the area could boost its shrinking revenues. However, Philip Jones, deputy editor of the Bookseller magazine, said that the internet firm might struggle to get some of its ideas adopted. \"This has been mooted before and was fairly well decried,\" he said. \"Perhaps in a limited context it could work if the merchandise was strongly related to the title and was kept away from the text. \"But readers - particularly parents - like the fact that reading is an advertising-free zone. Authors would also want something to say about ads interrupting their narrative flow.\""
}
  ```

### Format

All dataset files are in `.jsonl` format i.e. one JSON per line. One example from the English dataset is given above in JSON format. There are five fields: `id`, `url`, `title`, `summary`, and `text`.

### Split

We used a 80%-10%-10% split for all languages with a few exceptions. `English` was split 93%-3.5%-3.5% for the evaluation set size to resemble that of `CNN/DM` and `XSum`; `Scottish Gaelic`, `Kyrgyz` and `Sinhala` had relatively fewer samples, their evaluation sets were increased to 500 samples for more reliable evaluation. Same articles were used for evaluation in the two variants of Chinese and Serbian to prevent data leakage in multilingual training. Exact spilts can be found [here](https://github.com/csebuetnlp/xl-sum/blob/master/README.md). 


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