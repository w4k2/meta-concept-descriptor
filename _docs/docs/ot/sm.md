---
layout: page
title:  "Metafeature selection"
has_children: true
---

# Selection of metafeatures for effective concept identification

The selection was based on output of Experiment 4 and the ranking of mean (accumulated for synthetic and semi-synthetic streams) F-statistic.

The F-statistic, descibing the importance of each metafeature in the concept recognition task, was calculated for each stream. The values were accumulated for each stream type and sorted. 

The ranks, defined as the order of sorted, accumulated F-statistics values, had the most significant impact on the selection of metafeatures. Moreover, the diversity of the groups from which the metafeatures were selected was taken into account, trying to avoid combinations that strongly correlated with each other (e.g. C1 and C2 from complexity category, which both express imbalance ratio). The largest pool of metafeatures was determined based on rankings for real-world streams, assuming that they best represent real applications of stream processing methods.

| Metafeature | Rank in synthetic | Rank in semi-synthetic | Rank in real-world |
| --- | --- | --- | --- |
| int | --- | --- | 4 |
| nre | --- | --- | 5 |
| c1 | --- | --- | 7 |
| f1.mean | 5 | --- | 10 |
| cl_c.mean | --- | --- | 3 |
| cl_c.sd | --- | --- | 8 |
| cl_ent | --- | --- | 6 |
| j_ent.mean | --- | --- | 2 |
| j_ent.sd | --- | --- | 1 |
| mi.mean | --- | --- | 13 |
| mi.sd | --- | --- | 9 |
| wn.mean | --- | --- | 16 |
| g_m.sd | --- | --- | 15 |
| mean.mean | 1 | 1 | 12 |
| mean.sd | 4 | --- | --- |
| med.mean  | 3 | 3 | --- |
| t_m.mean | 2 | 2 | --- |

