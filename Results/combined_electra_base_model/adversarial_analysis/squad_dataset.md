# Adversarial Test Suite Analysis - combined_electra_base_model

## question_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| how              |       918 |       91 |                 167 |
| if               |        51 |        8 |                  12 |
| other            |       125 |       20 |                  24 |
| what             |      4921 |      411 |                 805 |
| when             |       982 |       28 |                  54 |
| where            |       376 |       38 |                  83 |
| who              |      1147 |       69 |                  76 |
| why              |        96 |       25 |                  43 |

## compositional_type

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| causal           |       229 |       40 |                  66 |
| compositional    |      1176 |      103 |                 171 |
| counting         |       850 |       57 |                  98 |
| entity lookup    |      4997 |      408 |                 784 |
| other            |       211 |       48 |                  77 |
| temporal         |      1153 |       34 |                  68 |

## multihop

| multi_question   |   correct |   failed |   partially correct |
|:-----------------|----------:|---------:|--------------------:|
| False            |      6881 |      541 |                1016 |
| True             |      1735 |      149 |                 248 |

## None

| result_type       |   Count |
|:------------------|--------:|
| correct           |    8616 |
| failed            |     690 |
| partially correct |    1264 |

---

