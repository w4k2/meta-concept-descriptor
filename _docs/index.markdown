---
layout: default
title: Home
nav_order: 1
permalink: /
---

# On metaattribute ability of implicit concept identification
{: .fs-9 }

Concept drift in data stream processing remains an intriguing challenge and states a popular research topic. Methods that actively process data streams use drift detectors, which performance is often based on monitoring the variability of different stream properties. This publication provides an overview and analysis of metafeatures variability describing data streams with concept drifts. Five experiments conducted on synthetic, semi-synthetic, and real-world data streams examine the ability and potential of over 160 metaattributes from 9 categories to recognize concepts in non-stationary data streams. The work reveals the distinctions in the considered sources of streams and specifies 17 metafeatures with a high ability of concept identification. The presented final group of metafeatures can continue to be reduced, depending on the stream characteristics.
{: .fs-6 .fw-300 }

{: .note }
> This website contains all the figures and supplementary materials for the article "On metaattribute ability of implicit concept identification". 

## [Experiment 1] – Chunk metafeature similarity with drifting concepts

This experiment aimed to verify whether the values expressing chunk metafeatures cluster into groups that would describe a single concept.

Streams were processed in batch mode. In the first experiment, the values of all analyzed metafeatures were calculated for each chunk of the generated data streams. The values from the entire stream course were examined. The expected result of the experiment was the grouping of metaattribute values in cohesive clusters depending on the current concept. In the case of streams with non-recurrent drift, metafeatures from adjacent chunks of data streams, unless a sudden drift has occurred, will lie close to each other in multidimensional space.

## [Experiment 2] – Concept classification in metafeature subgroups

After the first experiment, the calculated metaattribute values and concept indexes were retained for further analysis.

The second experiment aimed to verify the potential of metafeatures to identify concepts objectively. The ability of five classifiers, whose implementation came from the scikit-learn \citep{scikit-learn} library, to correctly recognize the concept was evaluated. The following classifiers were tested: GNB, KNN, SVM, DT, and MLP, each time assuming their default hyperparameterization. The quality of the classification was evaluated in a 5-times repeated stratified 2-fold cross-validation protocol. In the case of synthetic streams with sudden drift and semi-synthetic streams with the nearest drift, the problem was not characterized by a large imbalance. However, in the case of other streams, some of the concepts were short-term, significantly contributing to the class imbalance of the meta-problem. Therefore, it was decided to use the balanced accuracy metric for all cases. The experiment separately evaluated three stream categories (synthetic, semi-synthetic, and real-world).

Disjoint subgroups of the calculated metafeatures were considered to determine which of them carry information valuable in the concept recognition task. In the meta-problem, each chunk of data was identified as a single pattern whose features were described by the values of metaattributes from a given group. The label of the instance was the concept identifier. Patterns were randomly shuffled before starting the experiment.

## [Experiment 3] – Attribute selection

The third experiment aimed to analyze the performance of single metaattributes from promising categories in a recognition task.

In contrast to the second experiment, where disjoint groups of metaattributes were considered, in the third experiment, all metaattributes from categories previously considered promising were evaluated. Using the ANOVA F-test, the most informative features of each problem were selected. The considered number of metafeatures taken into account, starting with a single one and concluding with the analysis of the entire set of metafeatures. From this range, 29 values sampled from the space of a quadratic function were examined. As in a second experiment, the ability to assign a data chunk to a concept using selected metaattributes was tested. Similarly, three categories of streams were analyzed separately. For each repetition and each fold, the attributes taken into account in the classification were independently selected. The experiment was performed using a 5-times repeated 2-fold stratified cross-validation protocol.

## [Experiment 4] – Variance analysis and global metaatribute selection

The previous experiment aimed to select the most informative metaattributes for each stream, stream parameterization and replication (for synthetic and semi-synthetic), the source dataset (for semi-synthetic), and each fold. The aim of Experiment 4 is to analyze valuable metaattributes in the first place in three categories of stream origin, then the potential selection of a versatile collection of metafeatures that allow for effective identification of concepts for all streams.

In this experiment, the values of the F-statistic indicator of the ANOVA test were analyzed. F-statistic informs about the degree of linear dependence of a given attribute and the target label \citep{kim2017understanding}. As in the previous experiment, conclusions were drawn separately for three groups of streams (synthetic, semi-synthetic, and real-world). The evaluated F-statistics of the ANOVA test were determined for each type of stream and its replication.

The experiment aimed to select a group of metaattributes with high concept recognition capability for all three studied stream sources. In addition, the ability to classify using selected metaattributes was tested and compared to classification using a much larger, complete set of metafeatures.

## [Experiment 5] – Covariance analysis of selected metafeatures

The last experiment was aimed at analyzing the covariance of selected metaattributes. The experiment was divided into two parts -- the study of the covariance of the entire set describing the data stream, and the successive fragments of the stream in windows of 25 chunks. Both the covariance over the entire stream and the standard deviation of the covariance of features from successive windows were analyzed.

The experiment was to determine whether it was possible to exclude some of the analyzed measures based on their correlation. Reducing the number of metaattributes taken into account could, in practical applications, reduce the computational complexity of methods determining their values for the purpose of the concept identification task, without negatively affecting the quality of recognition. In this experiment, streams from three sources will be analyzed, which will allow determining differences and similarities in metaattribute values in changing concepts.

## [Metafeatures] – List of considered metafeatures

The description is based on the docummentation of [pymfe](https://pymfe.readthedocs.io/en/latest/) and [problexity](https://problexity.readthedocs.io/en/latest/index.html) libraries.

## [Drift annotation] – Drift annotation procedure

Real concept drifts are associated with a change (usually a decrease) in the quality of the classification achieved by the classifier. If the classifier was trained using data from a concept other than the current one, its recognition quality should decrease, as the classifier is not *familiar* with the current data deistribution. If an increase in quality is observed, it can be suspected that the data distribution is close to the previous concept and there are fewer samples in areas of overlap between class samples - which is also related to the change in concept.

## [Metafeature selection] – Selection of metafeatures for effective concept identification

The selection was based on output of Experiment 4 and the ranking of mean (accumulated for synthetic and semi-synthetic streams) F-statistic.

The F-statistic, descibing the importance of each metafeature in the concept recognition task, was calculated for each stream. The values were accumulated for each stream type and sorted. 


----

[Experiment 1]: docs/e1/e1
[Experiment 2]: docs/e2/e2
[Experiment 3]: docs/e3/e3
[Experiment 4]: docs/e4/e4
[Experiment 5]: docs/e5/e5
[Metafeatures]: docs/ot/cm
[Drift annotation]: docs/ot/da
[Metafeature selection]: docs/ot/sm