import numpy as np
from time import time
import random
from tmu.tsetlin_machine import LCTM
import itertools


num_sample_per_sub_pattern = 300
sub_patterns_limit = 2
number_of_classes = 2

num_clauses_l = [8]
T_l = [5]
S_l = [100]
reset_guess_threshold = 200 # if LAs assigned to label decision stayed in a loop more than that threshold without being rewarded as all together at single loop. We resample and reinitialize these LAs with new random initial labels and states.
pattern_search_perc = 1 # percentage of all labels learning automata that must be rewarded together to accept their labels
epsilon =0.8 # percentage of how likely to penalize a wrong labeled decided from an LA. leaving a window for exploration (1 - epsilon). If all LAs penalized at once for example, they will be in a loop due to the symetric nature of the issue.



f = open('..//my_datasets/MNISTTraining.txt')
X_train = []
all_data = f.readlines()
all_patterns = []

for y in range(number_of_classes):
    sub_patterns = []
    for l in all_data:
        x = [int(ll) for ll in l[:-3].split(' ')]
        if y == int((l[-2])):
            for i in range(num_sample_per_sub_pattern):
                X_train.append(np.array(x))

            if x not in sub_patterns:
                sub_patterns.append(x)

            if len(sub_patterns) == sub_patterns_limit:
                all_patterns += sub_patterns
                break
            
        
num_subpatterns = len(all_patterns)
print('Sub-patterns in Data: ', num_subpatterns)

X_train = np.array(X_train)
num_features = len(X_train[0])

results = []
runs = 1
while runs != 0:
    for num_clauses in num_clauses_l:
        for T in T_l:
            if T > num_clauses:
                continue
            for S in S_l:
                good_counter = 0
                lctm = LCTM(num_clauses, T, S, platform='CUDA', pattern_search_exit= pattern_search_perc, epsilon = epsilon, reset_guess_threshold = reset_guess_threshold)

                lctm.fit(num_clauses, num_features, [X_train])
                print('\nFinal Results--> (Grouped Samples):')
                all_samples = []
                all_labels  = []
                for k, v in lctm.grouped_samples.items():
                    for sample in v:
                        all_samples.append(sample)
                        all_labels.append(k)
                    print('Cluster Size: ', len(v))
                    #print(v)    
                    print('Cluster Included Patterns Info:')
                    print('Cluster Clauses: ', lctm.interpretability_clauses[k])

                    good = lctm.get_cluster_info(all_patterns, v)
                    if good:
                        good_counter += 1
                    print('----------------------------------------')

                # to extract learning (total penalities over inside loops of the LCTM)        
                '''for loop_index, info in lctm.learning_info.items():
                    print('Loop: ', loop_index)
                    print(info)
                    print('-----------------------')'''
                if good_counter == num_subpatterns:
                    results.append(1)
                    #print('Parameters are good: ', num_clauses, T, S)
                    #exit()
                else:
                    results.append(0)
                    
                    
    runs -= 1
   
print('Percentage of Successful Learning was ---->', np.mean(results))