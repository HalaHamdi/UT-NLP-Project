# Adversarial Test Suite Analysis - SquadFinetuned

## question_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| how              |        29 |      147 |                  27 |
| if               |         5 |       37 |                  12 |
| other            |        27 |      168 |                  12 |
| what             |       324 |     1371 |                 159 |
| when             |        16 |       66 |                  14 |
| where            |        33 |      110 |                  19 |
| who              |        81 |      236 |                  29 |
| why              |         7 |       61 |                  10 |

## compositional_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| causal           |        22 |      108 |                  14 |
| compositional    |        98 |      422 |                  44 |
| counting         |        25 |       89 |                   8 |
| entity lookup    |       327 |     1295 |                 169 |
| other            |        30 |      206 |                  29 |
| temporal         |        20 |       76 |                  18 |

## multihop

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| False            |       386 |     1596 |                 208 |
| True             |       136 |      600 |                  74 |

## None

| result_type       |   Count |
|:------------------|--------:|
| correct           |     522 |
| failed            |    2196 |
| partially correct |     282 |

---

