# Adversarial Test Suite Analysis - PipelinedAdvFinetuned

## question_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| how              |        45 |      127 |                  31 |
| if               |        11 |       31 |                  12 |
| other            |        67 |      128 |                  12 |
| what             |       512 |     1145 |                 197 |
| when             |        19 |       64 |                  13 |
| where            |        41 |      101 |                  20 |
| who              |       113 |      198 |                  35 |
| why              |         6 |       55 |                  17 |

## compositional_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| causal           |        26 |       92 |                  26 |
| compositional    |       171 |      332 |                  61 |
| counting         |        44 |       66 |                  12 |
| entity lookup    |       485 |     1117 |                 189 |
| other            |        65 |      168 |                  32 |
| temporal         |        23 |       74 |                  17 |

## multihop

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| False            |       552 |     1383 |                 255 |
| True             |       262 |      466 |                  82 |

## None

| result_type       |   Count |
|:------------------|--------:|
| correct           |     814 |
| failed            |    1849 |
| partially correct |     337 |

---

