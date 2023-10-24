import numpy as np

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)

clustering = ['ch', 'int', 'nre', 'pb', 'sc', 'sil', 'vdb', 'vdu']
complexity = ['c1', 'c2', 'cls_coef', 'density', 'f1.mean', 'f1.sd', 'f1v.mean', 'f1v.sd', 'f2.mean', 'f2.sd', 'f3.mean', 'f3.sd', 'f4.mean', 'f4.sd', 'hubs.mean', 'hubs.sd', 'l1.mean', 'l1.sd', 'l2.mean', 'l2.sd', 'l3.mean', 'l3.sd', 'lsc', 'n1', 'n2.mean', 'n2.sd', 'n3.mean', 'n3.sd', 'n4.mean', 'n4.sd', 't1.mean', 't1.sd', 't2', 't3', 't4']
concept = ['coh.mean', 'coh.sd', 'con.mean', 'con.sd', 'impc.mean', 'impc.sd', 'wg_d.mean', 'wg_d.sd']
general = ['a_to_inst', 'cat_to_num', 'freq_class.mean', 'freq_class.sd', 'inst_to_a', 'nr_attr', 'nr_bin', 'nr_cat', 'nr_class', 'nr_inst', 'nr_num', 'num_to_cat']
info_theory = ['a_conc.mean', 'a_conc.sd', 'a_ent.mean', 'a_ent.sd', 'cl_c.mean', 'cl_c.sd', 'cl_ent', 'eq_num_attr', 'j_ent.mean', 'j_ent.sd', 'mi.mean', 'mi.sd', 'ns_ratio']
itemset = ['one_i.mean', 'one_i.sd', 'two_i.mean', 'two_i.sd']
landmarking = ['bn.mean', 'bn.sd', 'enn.mean', 'enn.sd', 'lind.mean', 'lind.sd', 'nb.mean', 'nb.sd', 'onn.mean', 'onn.sd', 'rn.mean', 'rn.sd', 'wn.mean', 'wn.sd']
model_based = ['leaves', 'l_br.mean', 'l_b.sd', 'l_cor.mean', 'l_cor.sd', 'l_h.mean', 'l_h.sd', 'l_cl.mean', 'l_cl.sd', 'nodes', 'n_p_attr', 'n_p_inst', 'n_p_l.mean', 'n_p_l.sd', 'n_r.mean', 'n_r.sd', 'td.mean', 'td.sd', 'ti.mean', 'ti.sd', 'ts.mean', 'ts.sd', 'v_i.mean', 'v_i.sd']
statistical = ['c_cor.mean', 'c_cor.sd', 'cor.mean', 'cor.sd', 'cov.mean', 'cov.sd', 'eig.mean', 'eig.sd', 'g_m.mean', 'g_m.sd', 'gravity', 'h_m.mean', 'h_m.sd', 'iq_r.mean', 'iq_r.sd', 'kur.mean', 'kur.sd', 'lh_trace', 'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 'med.mean', 'med.sd', 'min.mean', 'min.sd', 'nr_cor_attr', 'nr_disc', 'nr_norm', 'nr_outliers', 'p_trace', 'ran.mean', 'ran.sd', 'roy_root', 'sd.mean', 'sd.sd', 'sd_ratio', 'skew.mean', 'skew.sd', 'spar.mean', 'spar.sd', 't_m.mean', 't_m.sd', 'var.mean', 'var.sd', 'w_lambda']


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

measure_labels_selected = [
    clustering,
    complexity,
    # concept,
    # general,
    info_theory,
    # itemset,
    landmarking,
    # model_based,
    statistical
]

measure_labels_selected_flat = np.array(sum(measure_labels_selected, []))

selected2_measure_names_draft = ['mean.mean', 't_m.mean', 'med.mean', 'mean.sd', 'f1.mean', 'mi.mean', 'g_m.mean', 'wn.mean',
                           'j_ent.sd', 'j_ent.mean', 'cl_c.mean', 'int', 'nre', 'cl_ent', 'c1', 'cl_c.sd' ,'mi.sd']

selected2_indexes = []
for m_id, m in enumerate(measure_labels_selected_flat):
    if m in selected2_measure_names_draft:
        selected2_indexes.append(m_id)
        
selected2_measure_names = measure_labels_selected_flat[selected2_indexes]
    
drift_gt = {
    'covtypeNorm-1-2vsAll-pruned': [ 57, 121, 131, 155, 205, 260, 295, 350],
    'electricity': [ 20,  38,  55, 115, 145],
    'poker-lsn-1-2vsAll-pruned': [  45,   90,  110,  120,  160,  182,  245,  275,  292,  320,  358,  400,  450,  468,
    480,  516,  540,  550,  590,  600,  640,  710,  790,  831,  850,  880,  900,  920,
    965, 1000, 1010],
    'INSECTS-abrupt_imbalanced_norm': [125],
    'INSECTS-gradual_imbalanced_norm': [  9,  60,  90, 125, 190],
    'INSECTS-incremental_imbalanced_norm': [  9,  35,  60, 180, 220]
}