# Interpretable Evaluation for Text Generation



## Goal

## Input: System output
The specific format (jsonl) follows:
```json
{
	'id': an unique id
	'src': 'This is the source text.',
	'refs': ['This is the reference text one.',
					 'This is the reference text two.',
					 'This is the reference text theree.'],
	'hypo': 'This is the hypothesis text.',
	'scores': {
		'bleu': '0.321',
		'comet': '1.211',
		'bert_score': '0.877',
		...
	}
}
```

#### Example
Here is an [example]()


## Output: Analysis of system output

#### Example
Here is an [example](https://github.com/neulab/ExplainaBoard/blob/NLG/developing/interpretNLG/example/newstest2020.dong-nmt.768.en-zh.json)

