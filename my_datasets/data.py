import numpy as np
import random
from sklearn.model_selection import train_test_split


class Synthetic(object):
    
    def __init__(self, min_num_samples, max_num_samples, feat_fim):
        self.max_num_samples = max_num_samples
        self.min_num_samples = min_num_samples
        self.feat_dim = feat_fim
        self.samples = {}
    
    def create_patterns(self):
        if self.feat_dim not in [20, 60, 100, 200, 400, 2000, 4000, 6000, 8000, 10000]:
            print('CANNOT CREATE SYNTHETIC DATA WITH THE GIVEN NUMBER OF FEATURES PER PATTERN!')
            exit()
        else:
            all_patterns = []
            start_index = 0
            for r in range(1):  # we create 2 * 10 sub patterns in the dataset, later we can decide on how many classes that copnclude some of these sub patterns
                end_index = (self.feat_dim/10) * (r+1) - 1 # for example 2000 features will be divided to 10 parts, each= 200. each part with 200 will be assigned ones for its even indices, odds otherwise. Then we do the opposite for same part. rest of parts are staying zeros. That gives us 2 * 10 parts total distict sub patterns.
                
                # even features assigned to 1
                pattern = np.zeros(self.feat_dim, dtype=np.uint32)
                for e, _ in enumerate(pattern):
                    if e >= start_index and e <= end_index:
                        if e % 2 == 0:
                            pattern[e] = 1
                    
                all_patterns.append(pattern)
                # odd features assigned to 1
                pattern = np.zeros(self.feat_dim, dtype=np.uint32)
                for e, _ in enumerate(pattern):
                    if e >= start_index and e <= end_index:
                        if e % 2 != 0:
                            pattern[e] = 1
                
                all_patterns.append(pattern)
                start_index = end_index

            # in case we merge or control how distict these 20 subpatterns are we will need some info about diversity percentages
            # below culate the diversity between created sub patterns on average and also with detailed divesity between each sub pattern and the others
            all_avgs = []
            detailed_diversity = {}
            for e, pattern in enumerate(all_patterns):
                detailed_diversity.update({e:[]})
                for e2, other_pattern in enumerate(all_patterns):
                    if e != e2 and len(all_patterns) > 1:
                        diverse_counter = 0
                        for e3, p in enumerate(pattern):
                            if other_pattern[e3] != p:
                                diverse_counter += 1

                        pattern_div = diverse_counter/self.feat_dim
                        detailed_diversity[e].append((e2, pattern_div))
                    else:
                        pattern_div = 0
                        detailed_diversity[e].append((e2, pattern_div))
                if len(all_patterns) > 1:
                    avg_pattern_div = pattern_div/(len(all_patterns)-1)
                else:
                    avg_pattern_div = 0
                all_avgs.append(avg_pattern_div)

            print('----------------------------------------------------------------------------------')
            print('Synthetic Data Has Been Created with Overall Avg Diversity Between patterns: ', np.mean(all_avgs)) 
            print('----------------------------------------------------------------------------------')

            '''print('\n----------------------------------------------------------------------------------')
            print('Detailed Diversity Between patterns: \n')
            for k, v in detailed_diversity.items():
                print('Pattern ', k, ' Diversity Between other Patterns: ', v)
            print('----------------------------------------------------------------------------------')'''


            classes_samples = {}    
            for x, pattern in enumerate(all_patterns):
                print('Pattern ', x, ':', pattern)
                cls_samples = np.zeros((random.choice([self.min_num_samples, self.max_num_samples]), self.feat_dim), dtype=np.uint32)
                for i in range(len(cls_samples)):
                    cls_samples[i] = pattern
                classes_samples.update({x: cls_samples})

            return classes_samples, all_patterns
    
    
    def load_data(self):
        X, Y = [], []
        classes_samples, all_patterns = self.create_patterns()
        for k in classes_samples.keys():
            for sample in classes_samples[k]:
                X.append(sample)
                Y.append(k)
        return train_test_split(np.array(X), np.array(Y), random_state=42, shuffle=True, test_size=0.30), all_patterns

                
            
        