import numpy as np

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)

clustering = ['ch', 'int', 'nre', 'pb', 'sc', 'sil', 'vdb', 'vdu']
complexity = ['c1', 'c2', 'cls_coef', 'density', 'f1.mean', 'f1.sd', 'f1v.mean', 'f1v.sd', 'f2.mean', 'f2.sd', 'f3.mean', 'f3.sd', 'f4.mean', 'f4.sd', 'hubs.mean', 'hubs.sd', 'l1.mean', 'l1.sd', 'l2.mean', 'l2.sd', 'l3.mean', 'l3.sd', 'lsc', 'n1', 'n2.mean', 'n2.sd', 'n3.mean', 'n3.sd', 'n4.mean', 'n4.sd', 't1.mean', 't1.sd', 't2', 't3', 't4']
concept = ['cohesiveness.mean', 'cohesiveness.sd', 'conceptvar.mean', 'conceptvar.sd', 'impconceptvar.mean', 'impconceptvar.sd', 'wg_dist.mean', 'wg_dist.sd']
general = ['attr_to_inst', 'cat_to_num', 'freq_class.mean', 'freq_class.sd', 'inst_to_attr', 'nr_attr', 'nr_bin', 'nr_cat', 'nr_class', 'nr_inst', 'nr_num', 'num_to_cat']
info_theory = ['attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean', 'attr_ent.sd', 'class_conc.mean', 'class_conc.sd', 'class_ent', 'eq_num_attr', 'joint_ent.mean', 'joint_ent.sd', 'mut_inf.mean', 'mut_inf.sd', 'ns_ratio']
itemset = ['one_itemset.mean', 'one_itemset.sd', 'two_itemset.mean', 'two_itemset.sd']
landmarking = ['best_node.mean', 'best_node.sd', 'elite_nn.mean', 'elite_nn.sd', 'linear_discr.mean', 'linear_discr.sd', 'naive_bayes.mean', 'naive_bayes.sd', 'one_nn.mean', 'one_nn.sd', 'random_node.mean', 'random_node.sd', 'worst_node.mean', 'worst_node.sd']
model_based = ['leaves', 'leaves_branch.mean', 'leaves_branch.sd', 'leaves_corrob.mean', 'leaves_corrob.sd', 'leaves_homo.mean', 'leaves_homo.sd', 'leaves_per_class.mean', 'leaves_per_class.sd', 'nodes', 'nodes_per_attr', 'nodes_per_inst', 'nodes_per_level.mean', 'nodes_per_level.sd', 'nodes_repeated.mean', 'nodes_repeated.sd', 'tree_depth.mean', 'tree_depth.sd', 'tree_imbalance.mean', 'tree_imbalance.sd', 'tree_shape.mean', 'tree_shape.sd', 'var_importance.mean', 'var_importance.sd']
statistical = ['can_cor.mean', 'can_cor.sd', 'cor.mean', 'cor.sd', 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 'g_mean.mean', 'g_mean.sd', 'gravity', 'h_mean.mean', 'h_mean.sd', 'iq_range.mean', 'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd', 'lh_trace', 'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 'median.mean', 'median.sd', 'min.mean', 'min.sd', 'nr_cor_attr', 'nr_disc', 'nr_norm', 'nr_outliers', 'p_trace', 'range.mean', 'range.sd', 'roy_root', 'sd.mean', 'sd.sd', 'sd_ratio', 'skewness.mean', 'skewness.sd', 'sparsity.mean', '[sparsity.sd]', 't_mean.mean', 't_mean.sd', 'var.mean', 'var.sd', 'w_lambda']


measure_labels = [
    clustering,
    complexity,
    concept,
    general,
    info_theory,
    itemset,
    landmarking,
    model_based,
    statistical
]