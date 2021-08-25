# Dataset for WikiAsp


## Description

WikiAsp is an aspect-based summarization dataset constructed from Wikipedia. The inputs are the cited references, and the outputs are the actual Wikipedia section texts. It contains 20 domains with 10 pre-defined aspects for each domain.

## Meta Data
* Official Homepage: https://github.com/neulab/wikiasp
* Download link: [Release](https://github.com/neulab/wikiasp/releases/tag/v1.0)
* Paper: [WikiAsp: A Dataset for Multi-domain Aspect-based Summarization.](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00362/98088/WikiAsp-A-Dataset-for-Multi-domain-Aspect-based)
* Github Repo: [neulab/wikiasp](https://github.com/neulab/wikiasp)
* Corresponding Author: Hiroaki Hayashi
* Supported Task: Summarization
* Language: English

## Data Structure
### Example

```json
{    "exid": "train-78-3351",
    "inputs": [
        "< EOT > the family scorpaenidae contains around 45 genera and 380 species .",
        "scorpionfishes have large , heavily ridged and spined heads .",
        "venomous spines on their back and fins with a groove and venom sack .",
        "well camouflaged with tassels , warts and colored specks .", ...
    ],
    "targets": [
        [
            "behavior",
            "  i  .  didactylus is a piscivorous ambush predator  .  it is nocturnal and typically lies partially buried on the sea floor or on a coral head during the day  ,  covering itself with sand and other debris to further camouflage itself  .  it has no known natural predators  .  when disturbed by a scuba diver or a potential predator  ,  it fans out its brilliantly colored pectoral and caudal fins as a warning  .  once dug in  ,  it isvery reluctant to leave its hiding place  .  when it does move  ,  it displays an unusual mechanism of subcarangiform locomotion  \u2014  it crawls slowly along the seabed  ,  employing the four lower rays  (  two on each side  )  of its pectoral fins as legs  .  the bearded ghoul has poisonous dorsal fish spines that can cause a painful wound  ."
        ]
    ]
}
```

### Format
Each data sample contains the following fields:
* exid: Example identifier.
* inputs: Sentence-tokenized cited references.
* targets: List of aspect-based summaries, each of which is a pair of an aspect and a summary.

### Split
The data consists of 20 domains, each of which has its own splits of *train*, *development* and *test* sets.


| Domain | Train | Dev | Test |
| :----: | :----: |:----: |:----: |
| Album | 24434 | 3104 | 3038 |
| Animal | 16540 | 2005 | 2007 |
| Artist | 26754 | 3194 | 3329 |
| Building | 20449 | 2607 | 2482 |
| Company | 24353 | 2946 | 3029 |
| EducationalInstitution | 17634 | 2141 | 2267 |
| Event | 6475 | 807 | 828 |
| Film | 32129 | 4014 | 3981 |
| Group | 11966 | 1462 | 1444 |
| HistoricPlace | 4919 | 601 | 600 |
| Infrastructure | 17226 | 1984 | 2091 |
| MeanOfTransportation | 9277 | 1215 | 1170 |
| OfficeHolder | 18177 | 2218 | 2333 |
| Plant | 6107 | 786 | 774 |
| Single | 14217 | 1734 | 1712 |
| SoccerPlayer | 17599 | 2150 | 2280 |
| Software | 13516 | 1637 | 1638 |
| TelevisionShow | 8717 | 1128 | 1072 |
| Town | 14818 | 1911 | 1831 |
| WrittenWork | 15065 | 1843 | 1931 |

## Reference
```
@article{hayashi2021wikiasp,
  title={WikiAsp: A Dataset for Multi-domain Aspect-based Summarization},
  author={Hayashi, Hiroaki and Budania, Prashant and Wang, Peng and Ackerson, Chris and Neervannan, Raj and Neubig, Graham},
  journal={Transactions of the Association for Computational Linguistics},
  volume={9},
  pages={211--225},
  year={2021}
}
```
