# RumorRvNN

## Reproduction of paper
```
Jing Ma, Wei Gao, Kam-Fai Wong. Rumor Detection on Twitter with Tree-structured Recursive Neural Networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018.
```

## Run 
```
BU_RvNN & Twitter15: python main.py --model BU_RvNN --dataset 15
TD_RvNN & Twitter15: python main.py --model TD_RvNN --dataset 15
BU_RvNN & Twitter16: python main.py --model BU_RvNN --dataset 16
TD_RvNN & Twitter16: python main.py --model TD_RvNN --dataset 16
```

## Performance
Twitter15 Performance:

||Acc.|NR F1|FR F1|TR F1|UR F1|
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
|BU|0.726|0.712|0.726|0.775|0.691|
|TD|0.762|0.715|0.753|0.835|0.746|

Twitter16 Performance:

||Acc.|NR F1|FR F1|TR F1|UR F1|
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
|BU|0.724|0.595|0.776|0.805|0.667|
|TD|0.782|0.655|0.816|0.878|0.761|

## Dependencies
Python 3.x

Pytorch 1.4
