import numpy as np
from time import time
import random

from my_datasets.data import Synthetic
from tmu.tsetlin_machine import TMSSClassifier



num_clauses_l = [80]
T_l = [50]
S_l = [30]
num_features = 200
min_samples_per_sub_pattern = 100
max_samples_per_sub_pattern = 100
reset_guess_threshold = 500 # if LAs assigned to label decision stayed in a loop more than that threshold without being rewarded as all together at single loop. We resample and reinitialize these LAs with new random initial labels and states.
pattern_search_perc = 1 # percentage of all labels learning automata that must be rewarded together to accept their labels
epsilon_l = [.5] # percentage of how likely to penalize a wrong labeled decided from an LA. leaving a window for exploration (1 - epsilon). If all LAs penalized at once for example, they will be in a loop due to the symetric nature of the issue.

syn = Synthetic(min_samples_per_sub_pattern, max_samples_per_sub_pattern, num_features)
(X_train, X_test, _, Y_test), all_patterns = syn.load_data()

for num_clauses in num_clauses_l:
    for T in T_l:
        if T > num_clauses:
            continue
        for S in S_l:
            for epsilon in epsilon_l:
                sstm = TMSSClassifier(num_clauses, T, S, platform='CUDA', pattern_search_exit= pattern_search_perc, epsilon = epsilon, reset_guess_threshold = reset_guess_threshold)

                sstm.fit(num_clauses, num_features, [X_train])
                print('\nFinal Results--> (Grouped Samples):')
                for k, v in sstm.grouped_samples.items():
                    print('Cluster Size: ', len(v))
                    #print(v)       
                    print('Cluster Included Patterns Info:')
                    sstm.get_cluster_info(all_patterns, v)
                    print('----------------------------------------')
                if len(sstm.grouped_samples.keys()) == 4:
                    print('Parameters are good: ', num_clauses, T, S, epsilon)
                    exit()
            