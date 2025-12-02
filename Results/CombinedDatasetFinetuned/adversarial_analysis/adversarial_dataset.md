# Adversarial Test Suite Analysis - CombinedDatasetFinetuned

## question_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| how              |        44 |      129 |                  30 |
| if               |        10 |       39 |                   5 |
| other            |        59 |      129 |                  19 |
| what             |       512 |     1120 |                 222 |
| when             |        21 |       61 |                  14 |
| where            |        44 |       99 |                  19 |
| who              |       105 |      205 |                  36 |
| why              |        11 |       52 |                  15 |

## compositional_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| causal           |        26 |       99 |                  19 |
| compositional    |       169 |      340 |                  55 |
| counting         |        41 |       72 |                   9 |
| entity lookup    |       479 |     1087 |                 225 |
| other            |        66 |      165 |                  34 |
| temporal         |        25 |       71 |                  18 |

## multihop

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| False            |       547 |     1360 |                 283 |
| True             |       259 |      474 |                  77 |

## None

| result_type       |   Count |
|:------------------|--------:|
| correct           |     806 |
| failed            |    1834 |
| partially correct |     360 |

---

