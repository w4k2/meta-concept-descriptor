# List of considered metafeatures

The description is based on the docummentation of [pymfe](https://pymfe.readthedocs.io/en/latest/) and [problexity](https://problexity.readthedocs.io/en/latest/index.html) libraries.

### Clustering

| Metafeature name/abbreviation | Description |
| --- | --- |
| CH | Calinski-Harabasz Index — The ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters, where dispersion is defined as the sum of distances squared. |
| INT | INT index -  Sum of pairwise normalized interclass distances, normalized according to the number of clusters. |
| NRE | Normalized relative entropy — an indicator of uniformity distributed of instances among clusters. |
| PB | Pearson correlation between class matching and instance distances. |
| SC | Number of clusters with size smaller than a given size(default=15). |
| SIL | Mean Silhouette value — The Silhouette Coefficient is calculated using the mean intra-cluster distance and the mean nearest-cluster distance for each sample. |
| VDB | Davies-Bouldin Index — The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. |
| VDU | Dunn Index — The lowest intercluster distance divided by the highest intracluster distance. |
            
### Complexity
| Metafeature name / name abbreviation | Description |
| --- | --- |
| C1 | Entropy of Class Proportions. |    
| C2 | Imbalance Ratio |    
| Cls_coef | Generates an epsilon-Nearest Neighbours graph. The edges are selected based on the Gower distance between samples, normalized to the range between 0 and 1. Edges between instances of distinct classes are removed. The neighborhood of each vertex is calculated – the instances directly connected to it. Then, the number of edges between the sample’s neighbors is calculated and divided by the maximum possible number of edges between them. The final measure is calculated based on the neighborhood of each point.|    
| Density | Generates an epsilon-Nearest Neighbours graph as decribed in Cls_coef. The measure calculates the number of edges in the final graph divided by the total possible number of edges.|    
| F1 | Maximum Fisher's discriminant ratio. |    
| F1v | Directional vector maximum Fisher's discriminant ratio. |    
| F2 | Volume of overlapping region. |    
| F3 | Maximum individual feature efficiency. |    
| F4 | Collective feature efficiency. |    
| Hubs | Generates an epsilon-Nearest Neighbours graph as decribed in Cls_coef. The measure scores each sample by the number of connections to neighbors, weighted by the number of connections the neighbors have. |    
| L1 | Sum of the error distance by linear programming. |    
| L2 | Error rate of linear classifier. |    
| L3 | Non linearity of linear classifier. |    
| LSC | Local set average cardinality. |    
| N1 | Fraction of borderline points. |
| N2 | Ratio of intra/extra class NN distance. |    
| N3 | Error rate of NN classifier. |    
| N4 | Nonlinearity of NN classifier. |    
| T1 | Fraction of hyperspheres covering data. |        
| T2 | Number of features per dimension. |        
| T3 | Number of PCA dimensions per points. |        
| T4 | Ration of the PCA dimension to the original dimension. |        

### Concept

| Metafeature name / name abbreviation | Description |
| --- | --- |
| Cohesiveness | Improved version of the weighted distance, that captures how dense or sparse is the distribution. |
| Conceptvar | Concept variation that estimates the variability of class labels among data samples. |
| Impconceptvar | Improved concept variation that estimates the variability of class labels. |
| wg_dist | Weighted distance, that captures how dense or sparse is the distribution. |


### General

| Metafeature name / name abbreviation | Description |
| --- | --- |       
| attr_to_inst | The ratio between the number of attributes and instances. |
| cat_to_num | Proportion of categorical and numerical attributes. |
| freq_class | Relative frequency of each distinct class. |
| inst_to_attr | Ratio of number of instances and number of predictive attributes. |
| nr_attr | Total number of attributes in the data without transformations. |
| nr_bin | Number of binary attributes. |
| nr_cat | Number of categorical attributes. |
| nr_class | Number of distinct classes. |
| nr_inst | Number of instances. |
| nr_num | Number of numerical attributes. |
| num_to_cat | Ratio of numerical and categorical features. |

### Information theory


| Metafeature name / name abbreviation | Description |
| --- | --- |
| attr_conc | Concentration coefficient for each pair of distinct predictive attribute. |
| attr_ent | Shannon’s Entropy of each predictive attribute. |
| class_conc | Concentration coefficient between each attribute and class. |
| class_ent | Target attribute Shannon’s entropy. |
| eq_num_attr | Number of attributes equivalent for a predictive task. |
| joint_ent | Estimated joint entropy between each predictive attribute and the target attribute. |
| mut_inf | Mutual information between each attribute and target. |
| ns_ratio | Estimated noisiness of the predictive attributes. |


### Itemset
        
| Metafeature name / name abbreviation | Description |
| --- | --- | 
| one_itemset | The one itemset is the individual frequency of each attribute in binary format. |
| two_itemset | The two-item set meta-feature can be seen as the correlation information of each one attributes value pairs in binary format. |

### Landmarking
        
| Metafeature name / name abbreviation | Description |
| --- | --- |        
| best_node | Performance of a the best single decision tree node. Construct a single decision tree node model induced by the most informative attribute to establish the linear separability. |
| elite_nn | Performance of Elite Nearest Neighbor. Elite nearest neighbor uses the most informative attribute in the dataset to induce the 1-nearest neighbor. With the subset of informative attributes it is expected that the models should be noise tolerant. |
| linear_discr | Performance of the Linear Discriminant classifier. The Linear Discriminant Classifier is used to construct a linear split (non parallel axis) in the data to establish the linear separability. |
| naive_bayes | Performance of the Naive Bayes classifier. It assumes that the attributes are independent and each example belongs to a certain class based on the Bayes probability. |
| one_nn | Performance of the 1-Nearest Neighbor classifier. It uses the euclidean distance of the nearest neighbor to determine how noisy is the data. |
| random_node | Performance of the single decision tree node model induced by a random attribute. |
| worst_node | Performance of the single decision tree node model induced by the worst informative attribute. |


### Model-based
        
| Metafeature name / name abbreviation | Description |
| --- | --- |
| leaves | Number of leaf nodes in the DT model. |
| leaves_branch | Size of branches in the DT model. The size of branches consists in the depth of all leaves of the DT model. |
| leaves_corrob | Leaves corroboration of the DT model. The Leaves corroboration is the proportion of examples that belong to each leaf of the DT model. |
| leaves_homo | DT model Homogeneity for every leaf node. The DT model homogeneity is calculated by the number of leaves divided by the structural shape (which is calculated by the ft_tree_shape method) of the DT model. |
| leaves_per_class | Proportion of leaves per class in DT model. This quantity is computed by the proportion of leaves of the DT model associated with each class. |
| nodes | Number of non-leaf nodes in DT model. |
| nodes_per_attr | Ratio of nodes per number of attributes in DT model. |
| nodes_per_inst | Ratio of non-leaf nodes per number of instances in DT model. |
| nodes_per_level | Ratio of number of nodes per tree level in DT model. |
| nodes_repeated | Number of repeated nodes in DT model. The number of repeated nodes is the number of repeated attributes that appear in the DT model. |
| tree_depth | Depth of every node in the DT model. |
| tree_imbalance | Tree imbalance for each leaf node. |
| tree_shape | Tree shape for every leaf node. The tree shape is the probability of arrive in each leaf given a random walk. We call this as the structural shape of the DT model. |
| var_importance | Features importance of the DT model for each attribute. It is calculated using the Gini index to estimate the amount of information used in the DT model. |


### Statistical

| Metafeature name / name abbreviation | Description |
| --- | --- |
| can_cor | Canonical correlations of data. |   
| cor |  Correlation of distinct dataset column pairs. |      
| cov | Absolute value of the covariance of distinct dataset attribute pairs. |      
| eigenvalues | Eigenvalues of covariance matrix from dataset.|      
| g_mean | Geometric mean of each attribute.|      
| gravity | Distance between minority and majority classes center of mass.|      
| h_mean | Harmonic mean of each attribute. |      
| iq_range | Interquartile range (IQR) of each attribute.|      
| kurtosis | Kurtosis of each attribute. |      
| lh_trace | Lawley-Hotelling trace. |      
| mad | Median Absolute Deviation (MAD) adjusted by a factor. |      
| max | Maximum value from each attribute.|      
| mean | Mean value from each attribute.|      
| median | Median value from each attribute.|      
| min | Minimum value from each attribute.|      
| nr_cor_attr | Number of distinct highly correlated pair of attributes. |      
| nr_disc | Number of canonical correlation between each attribute and class.|      
| nr_norm | Number of attributes normally distributed based in a given method.|      
| nr_outliers | Number of attributes with at least one outlier value. |      
| p_trace | Pillai’s trace. |      
| range | Range (max - min) of each attribute. |      
| roy_root | Roy’s largest root.|      
| sd | Standard deviation of each attribute.|      
| sd_ratio | Statistical test for homogeneity of covariances.|      
| skewness | Skewness for each attribute. |      
| sparsity | Sparsity metric for each attribute. |      
| t_mean | Trimmed mean of each attribute. |      
| var | Variance of each attribute. |      
| w_lambda |  Wilks’ Lambda value. |      
