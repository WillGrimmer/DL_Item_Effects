"""Estimators: multi-arm AIPW, effect curves, shrinkage and FDR."""
from .aipw import (MultiArmFit, crossover_label, effect_table, fit_multiarm, pairwise,
                   pairwise_table, vs_actual, vs_alternatives)
from .fdr import benjamini_hochberg, two_sided_p
from .heterogeneity import dr_curve, empirical_bayes
