# Adversarial Test Suite Analysis - combined_electra_base_model

## question_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| how              |        63 |      101 |                  39 |
| if               |        13 |       29 |                  12 |
| other            |       107 |       75 |                  25 |
| what             |       748 |      810 |                 296 |
| when             |        29 |       50 |                  17 |
| where            |        58 |       73 |                  31 |
| who              |       164 |      143 |                  39 |
| why              |        20 |       44 |                  14 |

## compositional_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| causal           |        48 |       76 |                  20 |
| compositional    |       231 |      247 |                  86 |
| counting         |        48 |       57 |                  17 |
| entity lookup    |       722 |      779 |                 290 |
| other            |       117 |      109 |                  39 |
| temporal         |        36 |       57 |                  21 |

## multihop

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| False            |       843 |      973 |                 374 |
| True             |       359 |      352 |                  99 |

## None

| result_type       |   Count |
|:------------------|--------:|
| correct           |    1202 |
| failed            |    1325 |
| partially correct |     473 |

---

