import numpy as np
from time import time
import random

from my_datasets.data import Synthetic
from tmu.tsetlin_machine import TMSSClassifier


num_clauses_l = [8]
T_l = [5]
S_l = [10]
num_features = 400
min_samples_per_sub_pattern = 200
max_samples_per_sub_pattern = 200
reset_guess_threshold = 100 # if LAs assigned to label decision stayed in a loop more than that threshold without being rewarded as all together at single loop. We resample and reinitialize these LAs with new random initial labels and states.
pattern_search_perc = 1.0 # percentage of all labels learning automata that must be rewarded together to accept their labels
epsilon =0.8 # percentage of how likely to penalize a wrong labeled decided from an LA. leaving a window for exploration (1 - epsilon). If all LAs penalized at once for example, they will be in a loop due to the symetric nature of the issue.

syn = Synthetic(min_samples_per_sub_pattern, max_samples_per_sub_pattern, num_features)
(X_train, X_test, Y_train, Y_test), all_patterns = syn.load_data()

results = []
runs = 20
while runs != 0:
    for num_clauses in num_clauses_l:
        for T in T_l:
            if T > num_clauses:
                continue
            for S in S_l:
                good_counter = 0
                sstm = TMSSClassifier(num_clauses, T, S, platform='CUDA', pattern_search_exit= pattern_search_perc, epsilon = epsilon, reset_guess_threshold = reset_guess_threshold)

                sstm.fit(num_clauses, num_features, [X_train])
                print('\nFinal Results--> (Grouped Samples):')
                for k, v in sstm.grouped_samples.items():
                    print('Cluster Size: ', len(v))
                    #print(v)    
                    print('Cluster Included Patterns Info:')
                    good = sstm.get_cluster_info(all_patterns, v)
                    if good:
                        good_counter += 1
                    print('----------------------------------------')
                if good_counter == 2:
                    results.append(1)
                    #print('Parameters are good: ', num_clauses, T, S)
                    #exit()
                else:
                    results.append(0)
                    
    runs -= 1
   
print('Percentage of Successful Learning was ---->', np.mean(results))
print('Experiment Setup used [num_features, num_samples_per_subpattern] ---->', num_features, min_samples_per_sub_pattern)

            