# Dataset for Newsroom


## Description
**Newsroom** is a large dataset for training and evaluating summarization systems. It contains 1.3 million articles and summaries written by authors and editors in the newsrooms of 38 major publications. The summaries are obtained from search and social metadata between 1998 and 2017 and use a variety of summarization strategies combining extraction and abstraction.

## Meta Data
* Official Homepage: https://lil.nlp.cornell.edu/newsroom/index.html
* Download link: https://lil.nlp.cornell.edu/newsroom/download/index.html
* Paper: [NEWSROOM: A Dataset of 1.3 Million Summaries
with Diverse Extractive Strategies](https://www.aclweb.org/anthology/N18-1065.pdf)
* Github Repo: [lil-lab/newsroom](https://github.com/lil-lab/newsroom)
* Corresponding Author: Yoav Artzi
* Supported Task: Summarization
* Language: English



## Data Structure
A data sample is shown in the following json format.
### Example
```json
{
    "url": "http:\/\/www.nydailynews.com\/archives\/news\/1995\/10\/14\/1995-10-14_selena_s_last_cries___shot_s.html",
    "archive": "http:\/\/web.archive.org\/web\/20090428161725id_\/http:\/\/www.nydailynews.com:80\/archives\/news\/1995\/10\/14\/1995-10-14_selena_s_last_cries___shot_s.html",
    "title": "SELENA'S LAST CRIES SHOT SINGER BEGGED HELP, NAMED SUSPECT",
    "date": "20090428161725",
    "text": "By MATT SCHWARTZ in Houston and WENDELL JAMIESON in New York Daily News Writers\n\nSaturday, October 14th 1995, 4:22AM\n\nBleeding from a massive chest wound, Tejano star Selena cried, \"Help me! Help me! I've been shot!\" and then named her killer with her dying breath.\n\nShaken witnesses yesterday told a spellbound Houston courtroom how the blood-covered, mortally wounded 23-year-old Hispanic singing sensation burst into the lobby of the Corpus Christi Days Inn last March 31.\n\nGasping for breath, Selena told motel workers that Yolanda Saldivar the president of her fan club shot her once in the back. She begged, \"Close the door or she will shoot me again,\" the witnesses said.\n\nThe testimony came on the third day of Saldivar's trial on charges she murdered Selena with a shot from a .38-caliber revolver when the star tried to fire her for embezzling $30,000 from two boutiques she managed for the singer.\n\nAs a paramedic and motel workers recounted Selena's last desperate moments, her mother, father and brother sobbed quietly. Saldivar, as she has throughout the trial, stared blankly.\n\nRuben Deleon, the motel's sales director, said he knelt over the dying star and asked who shot her.\n\n\"She said 'Yolanda Saldivar in room 158,' \" Deleon said.\n\n\"She was yelling, 'Help me! Help me! I've been shot,' \" said Rosalinda Gonzalez, an assistant manager. \"I asked who shot her. She said the lady in room 158. She moaned. Her eyes rolled up.\"\n\nFront desk clerk Shawna Vela said she dialed 911 and took the phone with her as she kneeled over the fallen singer, asking her what happened.\n\n\"She said 'Yolanda,' she said 'In room 158,' \" Vela testified.\n\nThe first paramedic on the scene, Richard Fredrickson, testified that he arrived just two minutes after the call but it was already too late.\n\n\"The girl was covered with blood,\" he remembered. \"Blood was thick from her neck to her knees, all the way around both sides.\"\n\nFredrickson couldn't even see the mortal wound until he cut off Selena's sweatshirt. He felt for a pulse in her neck but could feel only twitching muscles, he said.\n\nMinutes later, as he rode in an ambulance with the now unconscious Selena, he unclenched the dying woman's fist and made an ironic discovery.\n\n\"When I opened it, a ring fell out,\" he said. \"It was covered with blood.\"\n\nThe 14-karat gold and diamond ring, topped with a white-gold egg, was a gift from the Grammy winner's boutique employes and Saldivar. Police have said Saldivar demanded the ring back. But before Selena could hand it over, she was shot.\n\nThe singer, whose full name was Selena Quintanilla Perez, was born around Easter and collected decorative eggs.\n\nThe defense claims Saldivar, 35, was hysterical and shot Selena by accident. Prosecutors say it was deliberate.",
    "summary": "Bleeding from a massive chest wound, Tejano star Selena cried, \"Help me! Help me! I've been shot!\"and then named her killer with her dying breath. Shaken witnesses yesterday told a spellbound Houston courtroom how the blood-covered, mortally wounded 23-year-old Hispanic singing sensation burst into the lobby of the Corpus Christi Days Inn last March 31. Gasping for breath, Selena told motel workers that Yolanda Saldivar the president of her fan club shot",
    "compression": 6.9651162791,
    "coverage": 0.988372093,
    "density": 25.4069767442,
    "compression_bin": "low",
    "coverage_bin": "high",
    "density_bin": "extractive"
}
```

### Format
Each data sample contains the following fields:
* url: Original URL.
* archive: URL in the Internet Archive ([Archive.org](https://archive.org/)).
* title: Article title.
* date: Article date.
* text: Article text.
* summary: Summary written by newsroom editors and journalists.
* compression: How succinct a summary is. Compression(A, S) is defined as the word ratio between the article and summary:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![\Large Compression(A, S) = \frac{|A|}{|S|}](https://latex.codecogs.com/svg.image?Compression(A,%20S)%20=%20%5Cfrac%7B%7CA%7C%7D%7B%7CS%7C%7D)
* coverage: The extent to which a summary is derivative of a text. Coverage(A, S) measures the percentage of words in the summary that are part of an extractive fragment with the article:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![\Large Coverage(A, S) = \frac{1}{|S|}\sum_{f \in \mathcal{F}(A, S)}|f|](https://latex.codecogs.com/svg.image?Coverage(A,&space;S)&space;=&space;\frac{1}{|S|}\sum_{f&space;\in&space;\mathcal{F}(A,&space;S)}|f|)
* density: How well the word sequence of a summary can be described as a series of extractions. Density(A, S) is defined as the average length of the extractive fragment to which each word in the summary belongs:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![\Large Density(A, S) = \frac{1}{|S|}\sum_{f \in \mathcal{F}(A, S)}|f|^2](https://latex.codecogs.com/svg.image?Density(A,%20S)%20=%20%5Cfrac%7B1%7D%7B%7CS%7C%7D%5Csum_%7Bf%20%5Cin%20%5Cmathcal%7BF%7D(A,%20S)%7D%7Cf%7C%5E2)
* compression_bin: The degree of compression.
* coverage_bin: The degree of coverage.
* density_bin: The degree of density.

### Split
The data are divided into *training* (76%), *development* (8%), *test* (8%), and unreleased test (8%) datasets using a hash function of the article URL.

| Dataset Split | Number of samples |
| :-----| :----: |
| Training | 995,041 | 
| Development | 108,837 | 
| Test | 108,862 |
| Unreleased test| 109,255 |



## Reference
```
@inproceedings{N18-1065,
  author    = {Grusky, Max and Naaman, Mor and Artzi, Yoav},
  title     = {NEWSROOM: A Dataset of 1.3 Million Summaries
               with Diverse Extractive Strategies},
  booktitle = {Proceedings of the 2018 Conference of the
               North American Chapter of the Association for
               Computational Linguistics: Human Language Technologies},
  year      = {2018},
}
```
