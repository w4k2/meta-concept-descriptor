---
layout: page
title:  "Drift Annotation procedure"
has_children: true
nav_order: 10
---

## Drift annotation procedure

Real concept drifts are associated with a change (usually a decrease) in the quality of the classification achieved by the classifier. If the classifier was trained using data from a concept other than the current one, its recognition quality should decrease, as the classifier is not *familiar* with the current data deistribution. If an increase in quality is observed, it can be suspected that the data distribution is close to the previous concept and there are fewer samples in areas of overlap between class samples - which is also related to the change in concept.

A human expert marked the locations of drifts based on the classification quality of three classifiers: Gaussian Naive Bayes (GNB) and Multilayer Perceptron (MLP) and Extreme Learning Machine (ELM) in the Test-Then-Train experimental protocol. For every chunk of data, the classifiers were first used in the inference and quality evaluation procedure, then trained using a new portion of data. Such a protocol should allow for the most accurate determination of real concept drifts at the beginning of stream processing, in particular a clear identification of the change between the first and second concept. Training the classifier with subsequent portions of data, especially in the case of MLP, which is *forgetting* the previous data distributions, should enable the identification of further concept changes.

Partial fitting MLP by default performs only one iteration of weight optimization, so at the beginning of stream processing the recognition quality using MLP is lower and later, if the concept is stable, it increases.

It should be emphasized that the processed streams were previously divided into chunks and pruned of those batches containing only single class samples. This makes the identified drift moments specific to the transformed streams used in the experiments and should not be used as unambiguous drift moments in the original streams for the purposes of other studies.

Below we present the classification results using scetterplot (top row) and plot (bottom row) for the processed streams. The quality obtained by GNB is marked in blue, the MLP is marked in gold, and in red - ELM. The x-axis shows the identified moments of drift, determined based on changes in classification quality.

### Electricity
![electricity](/meta-concept-descriptor/fig_stream/electricity.png)

### Covtype
![covtype](/meta-concept-descriptor/fig_stream/covtypeNorm-1-2vsAll-pruned.png)

### Poker
![poker](/meta-concept-descriptor/fig_stream/poker-lsn-1-2vsAll-pruned.png)

### Insect abrupt
![insect-abrupt](/meta-concept-descriptor/fig_stream/INSECTS-abrupt_imbalanced_norm.png)

### Insect gradual
![insect-grad](/meta-concept-descriptor/fig_stream/INSECTS-gradual_imbalanced_norm.png)

### Insect incremental
![insect-abrupt](/meta-concept-descriptor/fig_stream/INSECTS-incremental_imbalanced_norm.png)