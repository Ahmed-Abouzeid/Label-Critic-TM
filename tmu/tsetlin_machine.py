
# Copyright (c) 2021 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import tmu.tools
import sys
import numpy as np
import random
import collections
from tmu.clause_bank import ClauseBank
from tmu.weight_bank import WeightBank
from las.la_2 import LA_2
import pycuda.autoinit

from scipy.sparse import csr_matrix

from time import time


class TMBasis():
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1,
                 number_of_state_bits=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        self.number_of_clauses = number_of_clauses
        self.number_of_state_bits = number_of_state_bits
        self.T = int(T)
        self.s = s
        self.platform = platform
        self.patch_dim = patch_dim
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.weighted_clauses = weighted_clauses

        self.clause_drop_p = clause_drop_p
        self.literal_drop_p = literal_drop_p

        self.X_train = np.zeros(0, dtype=np.uint32)
        self.X_test = np.zeros(0, dtype=np.uint32)

        self.initialized = False

    def initialize(self, X, patch_dim):
        if len(X.shape) == 2:
            self.dim = (X.shape[1], 1, 1)
        elif len(X.shape) == 3:
            self.dim = (X.shape[1], X.shape[2], 1)
        elif len(X.shape) == 4:
            self.dim = (X.shape[1], X.shape[2], X.shape[3])

        if self.patch_dim == None:
            self.patch_dim = (X.shape[1], 1)

        self.number_of_features = int(
            self.patch_dim[0] * self.patch_dim[1] * self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (
                    self.dim[1] - self.patch_dim[1]))
        self.number_of_literals = self.number_of_features * 2

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.number_of_ta_chunks = int((self.number_of_literals - 1) / 32 + 1)

    def clause_co_occurrence(self, X, percentage=False):
        clause_outputs = csr_matrix(self.transform(X))
        if percentage:
            return clause_outputs.transpose().dot(clause_outputs).multiply(1.0 / clause_outputs.sum(axis=0))
        else:
            return clause_outputs.transpose().dot(clause_outputs)

    def transform(self, X):
        encoded_X = self.clause_bank.prepare_X(
            tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim,
                             0))
        transformed_X = np.empty((X.shape[0], self.number_of_clauses), dtype=np.uint32)
        for e in range(X.shape[0]):
            transformed_X[e, :] = self.clause_bank.calculate_clause_outputs_update(encoded_X, e)
        return transformed_X

    def transform_patchwise(self, X):
        encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                     self.patch_dim, 0)
        transformed_X = np.empty((X.shape[0], self.number_of_clauses * self.number_of_patches), dtype=np.uint32)
        for e in range(X.shape[0]):
            transformed_X[e, :] = self.clause_bank.calculate_clause_outputs_patchwise(encoded_X, e)
        return transformed_X.reshape((X.shape[0], self.number_of_clauses, self.number_of_patches))

    def get_ta_action(self, clause, ta):
        return self.clause_bank.get_ta_action(clause, ta)

    def get_ta_state(self, clause, ta):
        return self.clause_bank.get_ta_state(clause, ta)

    def set_ta_state(self, clause, ta, state):
        return self.clause_bank.set_ta_state(clause, ta, state)


class TMClassifier(TMBasis):
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1,
                 number_of_state_bits=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)

    def initialize(self, X, Y):
        super().initialize(X, self.patch_dim)

        self.number_of_classes = int(np.max(Y) + 1)

        self.weight_banks = []
        for i in range(self.number_of_classes):
            self.weight_banks.append(WeightBank(np.concatenate((np.ones(self.number_of_clauses // 2, dtype=np.int32),
                                                                -1 * np.ones(self.number_of_clauses // 2,
                                                                             dtype=np.int32)))))

        self.clause_banks = []
        if self.platform == 'CPU':
            for i in range(self.number_of_classes):
                self.clause_banks.append(
                    ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits,
                               self.number_of_patches))
        elif self.platform == 'CUDA':
            from tmu.clause_bank_cuda import ClauseBankCUDA
            for i in range(self.number_of_classes):
                self.clause_banks.append(
                    ClauseBankCUDA(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits,
                                   self.number_of_patches, X, Y))
        else:
            print("Unknown Platform")
            sys.exit(-1)

        self.positive_clauses = np.concatenate((np.ones(self.number_of_clauses // 2, dtype=np.int32),
                                                np.zeros(self.number_of_clauses // 2, dtype=np.int32)))
        self.negative_clauses = np.concatenate((np.zeros(self.number_of_clauses // 2, dtype=np.int32),
                                                np.ones(self.number_of_clauses // 2, dtype=np.int32)))

    def fit(self, X, Y):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_banks[0].prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_train = X.copy()

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

        clause_active = []
        for i in range(self.number_of_classes):
            clause_active.append(np.ascontiguousarray(
                np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(
                    np.int32)))

        for e in range(X.shape[0]):
            target = Ym[e]

            clause_outputs = self.clause_banks[target].calculate_clause_outputs_update(self.encoded_X_train, e)
            class_sum = np.dot(clause_active[target] * self.weight_banks[target].get_weights(), clause_outputs).astype(
                np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)

            update_p = (self.T - class_sum) / (2 * self.T)

            if self.weighted_clauses:
                self.weight_banks[target].increment(clause_outputs, update_p, clause_active[target], False)
            self.clause_banks[target].type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                                      clause_active[target] * self.positive_clauses,
                                                      self.encoded_X_train, e)
            self.clause_banks[target].type_ii_feedback(update_p, clause_active[target] * self.negative_clauses,
                                                       self.encoded_X_train, e)

            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)

            clause_outputs = self.clause_banks[not_target].calculate_clause_outputs_update(self.encoded_X_train, e)
            class_sum = np.dot(clause_active[not_target] * self.weight_banks[not_target].get_weights(),
                               clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)

            update_p = (self.T + class_sum) / (2 * self.T)

            if self.weighted_clauses:
                self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active[not_target], False)
            self.clause_banks[not_target].type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                                          clause_active[not_target] * self.negative_clauses,
                                                          self.encoded_X_train, e)
            self.clause_banks[not_target].type_ii_feedback(update_p, clause_active[not_target] * self.positive_clauses,
                                                           self.encoded_X_train, e)

        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_banks[0].prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
        for e in range(X.shape[0]):
            max_class_sum = -self.T
            max_class = 0
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(),
                                   self.clause_banks[i].calculate_clause_outputs_predict(self.encoded_X_test,
                                                                                         e)).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y

    def transform(self, X):
        encoded_X = self.clause_banks[0].prepare_X(
            tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim, self.patch_dim,
                             0))
        transformed_X = np.empty((X.shape[0], self.number_of_classes, self.number_of_clauses), dtype=np.uint32)
        for e in range(X.shape[0]):
            for i in range(self.number_of_classes):
                transformed_X[e, i, :] = self.clause_banks[i].calculate_clause_outputs_update(encoded_X, e)
        return transformed_X.reshape((X.shape[0], self.number_of_classes * self.number_of_clauses))

    def transform_patchwise(self, X):
        encoded_X = tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                     self.patch_dim, 0)
        transformed_X = np.empty(
            (X.shape[0], self.number_of_classes, self.number_of_clauses // 2 * self.number_of_patches), dtype=np.uint32)
        for e in range(X.shape[0]):
            for i in range(self.number_of_classes):
                transformed_X[e, i, :] = self.clause_bank[i].calculate_clause_outputs_patchwise(encoded_X, e)
        return transformed_X.reshape(
            (X.shape[0], self.number_of_classes * self.number_of_clauses, self.number_of_patches))

    def clause_precision(self, the_class, polarity, X, Y):
        clause_outputs = self.transform(X).reshape(X.shape[0], self.number_of_classes, 2, self.number_of_clauses // 2)[
                         :, the_class, polarity, :]
        if polarity == 0:
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
        else:
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
        return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                        true_positive_clause_outputs / (true_positive_clause_outputs + false_positive_clause_outputs))

    def clause_recall(self, the_class, polarity, X, Y):
        clause_outputs = self.transform(X).reshape(X.shape[0], self.number_of_classes, 2, self.number_of_clauses // 2)[
                         :, the_class, polarity, :]
        if polarity == 0:
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
        else:
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
        return true_positive_clause_outputs / Y[Y == the_class].shape[0]

    def get_weight(self, the_class, polarity, clause):
        if polarity == 0:
            return self.weight_banks[the_class].get_weights()[clause]
        else:
            return self.weight_banks[the_class].get_weights()[self.number_of_clauses // 2 + clause]

    def set_weight(self, the_class, polarity, clause, weight):
        if polarity == 0:
            self.weight_banks[the_class].get_weights()[clause] = weight
        else:
            self.weight_banks[the_class].get_weights()[self.number_of_clauses // 2 + clause] = weight

    def get_ta_action(self, the_class, polarity, clause, ta):
        if polarity == 0:
            return self.clause_banks[the_class].get_ta_action(clause, ta)
        else:
            return self.clause_banks[the_class].get_ta_action(self.number_of_clauses // 2 + clause, ta)

    def get_ta_state(self, the_class, polarity, clause, ta):
        if polarity == 0:
            return self.clause_banks[the_class].get_ta_state(clause, ta)
        else:
            return self.clause_banks[the_class].get_ta_state(self.number_of_clauses // 2 + clause, ta)

    def set_ta_state(self, the_class, polarity, clause, ta, state):
        if polarity == 0:
            return self.clause_banks[the_class].set_ta_state(clause, ta, state)
        else:
            return self.clause_banks[the_class].set_ta_state(self.number_of_clauses // 2 + clause, ta, state)


class TMCoalescedClassifier(TMBasis):
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1,
                 number_of_state_bits=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)

    def initialize(self, X, Y):
        super().initialize(X, self.patch_dim)

        self.number_of_classes = int(np.max(Y) + 1)

        if self.platform == 'CPU':
            self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits,
                                          self.number_of_patches)
        elif self.platform == 'CUDA':
            from tmu.clause_bank_cuda import ClauseBankCUDA
            self.clause_bank = ClauseBankCUDA(self.number_of_clauses, self.number_of_literals,
                                              self.number_of_state_bits, self.number_of_patches, X, Y)
        else:
            print("Unknown Platform")
            sys.exit(-1)

        self.weight_banks = []
        for i in range(self.number_of_classes):
            self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))

    def fit(self, X, Y):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_bank.prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_train = X.copy()

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

        clause_active = np.ascontiguousarray(
            np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(
                np.int32))
        for e in range(X.shape[0]):
            target = Ym[e]

            clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.encoded_X_train, e)

            class_sum = np.dot(clause_active * self.weight_banks[target].get_weights(), clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)
            update_p = (self.T - class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                             clause_active * (self.weight_banks[target].get_weights() >= 0),
                                             self.encoded_X_train, e)
            self.clause_bank.type_ii_feedback(update_p, clause_active * (self.weight_banks[target].get_weights() < 0),
                                              self.encoded_X_train, e)
            self.weight_banks[target].increment(clause_outputs, update_p, clause_active, True)

            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)

            class_sum = np.dot(clause_active * self.weight_banks[not_target].get_weights(), clause_outputs).astype(
                np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)
            update_p = (self.T + class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                             clause_active * (self.weight_banks[not_target].get_weights() < 0),
                                             self.encoded_X_train, e)
            self.clause_bank.type_ii_feedback(update_p,
                                              clause_active * (self.weight_banks[not_target].get_weights() >= 0),
                                              self.encoded_X_train, e)

            self.weight_banks[not_target].decrement(clause_outputs, update_p, clause_active, True)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))
        for e in range(X.shape[0]):
            max_class_sum = -self.T
            max_class = 0
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            for i in range(self.number_of_classes):
                class_sum = np.dot(self.weight_banks[i].get_weights(), clause_outputs).astype(np.int32)
                class_sum = np.clip(class_sum, -self.T, self.T)
                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y

    def clause_precision(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()
        if positive_polarity == 0:
            positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
        else:
            positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = clause_outputs[Y != the_class].sum(axis=0)
            false_positive_clause_outputs = clause_outputs[Y == the_class].sum(axis=0)

        return np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                        1.0 * true_positive_clause_outputs / (
                                true_positive_clause_outputs + false_positive_clause_outputs))

    def clause_recall(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        weights = self.weight_banks[the_class].get_weights()

        if positive_polarity == 0:
            positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
        else:
            positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
            true_positive_clause_outputs = positive_clause_outputs[Y != the_class].sum(axis=0)

        return true_positive_clause_outputs / Y[Y == the_class].shape[0]

    def get_weight(self, the_class, clause):
        return self.weight_banks[the_class].get_weights()[clause]

    def set_weight(self, the_class, clause, weight):
        self.weight_banks[the_class].get_weights()[clause] = weight


class TMOneVsOneClassifier(TMBasis):
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1,
                 number_of_state_bits=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)

    def initialize(self, X, Y):
        super().initialize(X, self.patch_dim)

        self.number_of_classes = int(np.max(Y) + 1)
        self.number_of_outputs = self.number_of_classes * (self.number_of_classes - 1)

        if self.platform == 'CPU':
            self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits,
                                          self.number_of_patches)
        elif self.platform == 'CUDA':
            from tmu.clause_bank_cuda import ClauseBankCUDA
            self.clause_bank = ClauseBankCUDA(self.number_of_clauses, self.number_of_literals,
                                              self.number_of_state_bits, self.number_of_patches, X, Y)
        else:
            print("Unknown Platform")
            sys.exit(-1)

        self.weight_banks = []
        for i in range(self.number_of_outputs):
            self.weight_banks.append(WeightBank(np.ones(self.number_of_clauses).astype(np.int32)))

    def fit(self, X, Y):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X_train = self.clause_bank.prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_train = X.copy()

        Ym = np.ascontiguousarray(Y).astype(np.uint32)

        clause_active = np.ascontiguousarray(
            np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(
                np.int32))
        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.encoded_X_train, e)

            target = Ym[e]
            not_target = np.random.randint(self.number_of_classes)
            while not_target == target:
                not_target = np.random.randint(self.number_of_classes)

            output = target * (self.number_of_classes - 1) + not_target - (not_target > target)

            class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)
            update_p = (self.T - class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                             clause_active * (self.weight_banks[output].get_weights() >= 0),
                                             self.encoded_X_train, e)
            self.clause_bank.type_ii_feedback(update_p, clause_active * (self.weight_banks[output].get_weights() < 0),
                                              self.encoded_X_train, e)
            self.weight_banks[output].increment(clause_outputs, update_p, clause_active, True)

            output = not_target * (self.number_of_classes - 1) + target - (target > not_target)

            class_sum = np.dot(clause_active * self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
            class_sum = np.clip(class_sum, -self.T, self.T)
            update_p = (self.T + class_sum) / (2 * self.T)

            self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback,
                                             clause_active * (self.weight_banks[output].get_weights() < 0),
                                             self.encoded_X_train, e)
            self.clause_bank.type_ii_feedback(update_p, clause_active * (self.weight_banks[output].get_weights() >= 0),
                                              self.encoded_X_train, e)
            self.weight_banks[output].decrement(clause_outputs, update_p, clause_active, True)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0], dtype=np.uint32))

        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)

            max_class_sum = -self.T * self.number_of_classes
            max_class = 0
            for i in range(self.number_of_classes):
                class_sum = 0
                for output in range(i * (self.number_of_classes - 1), (i + 1) * (self.number_of_classes - 1)):
                    output_sum = np.dot(self.weight_banks[output].get_weights(), clause_outputs).astype(np.int32)
                    output_sum = np.clip(output_sum, -self.T, self.T)
                    class_sum += output_sum

                if class_sum > max_class_sum:
                    max_class_sum = class_sum
                    max_class = i
            Y[e] = max_class
        return Y

    def clause_precision(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        precision = np.zeros((self.number_of_classes - 1, self.number_of_clauses))
        for i in range(self.number_of_classes - 1):
            other_class = i + (i >= the_class)
            output = the_class * (self.number_of_classes - 1) + i
            weights = self.weight_banks[output].get_weights()
            if positive_polarity:
                positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
                false_positive_clause_outputs = positive_clause_outputs[Y == other_class].sum(axis=0)
                precision[i] = np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                                        true_positive_clause_outputs / (
                                                true_positive_clause_outputs + false_positive_clause_outputs))
            else:
                positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == other_class].sum(axis=0)
                false_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
                precision[i] = np.where(true_positive_clause_outputs + false_positive_clause_outputs == 0, 0,
                                        true_positive_clause_outputs / (
                                                true_positive_clause_outputs + false_positive_clause_outputs))

        return precision

    def clause_recall(self, the_class, positive_polarity, X, Y):
        clause_outputs = self.transform(X)
        recall = np.zeros((self.number_of_classes - 1, self.number_of_clauses))
        for i in range(self.number_of_classes - 1):
            other_class = i + (i >= the_class)
            output = the_class * (self.number_of_classes - 1) + i
            weights = self.weight_banks[output].get_weights()
            if positive_polarity:
                positive_clause_outputs = (weights >= 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == the_class].sum(axis=0)
                recall[i] = true_positive_clause_outputs / Y[Y == the_class].shape[0]
            else:
                positive_clause_outputs = (weights < 0)[:, np.newaxis].transpose() * clause_outputs
                true_positive_clause_outputs = positive_clause_outputs[Y == other_class].sum(axis=0)
                recall[i] = true_positive_clause_outputs / Y[Y == other_class].shape[0]
        return recall

    def get_weight(self, output, clause):
        return self.weight_banks[output].get_weights()[clause]

    def set_weight(self, output, weight):
        self.weight_banks[output].get_weights()[output] = weight


class TMRegressor(TMBasis):
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1,
                 number_of_state_bits=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0):
        super().__init__(number_of_clauses, T, s, platform=platform, patch_dim=patch_dim,
                         boost_true_positive_feedback=boost_true_positive_feedback,
                         number_of_state_bits=number_of_state_bits, weighted_clauses=weighted_clauses,
                         clause_drop_p=clause_drop_p, literal_drop_p=literal_drop_p)

    def initialize(self, X, Y):
        super().initialize(X, self.patch_dim)

        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

        if self.platform == 'CPU':
            self.clause_bank = ClauseBank(self.number_of_clauses, self.number_of_literals, self.number_of_state_bits,
                                          self.number_of_patches)
        elif self.platform == 'CUDA':
            from tmu.clause_bank_cuda import ClauseBankCUDA
            self.clause_bank = ClauseBankCUDA(self.number_of_clauses, self.number_of_literals,
                                              self.number_of_state_bits, self.number_of_patches, X, Y)
        else:
            print("Unknown Platform")
            sys.exit(-1)

        self.weight_bank = WeightBank(np.ones(self.number_of_clauses).astype(np.int32))

    def fit(self, X, Y):
        if self.initialized == False:
            self.initialize(X, Y)
            self.initialized = True

        if not np.array_equal(self.X_train, X):
            self.encoded_X = self.clause_bank.prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.encoded_Y = np.ascontiguousarray(
                ((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32))
            self.X_train = X.copy()

        clause_active = np.ascontiguousarray(
            np.random.choice(2, self.number_of_clauses, p=[self.clause_drop_p, 1.0 - self.clause_drop_p]).astype(
                np.int32))
        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_update(self.encoded_X_train, e)

            pred_y = np.dot(clause_active * self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
            pred_y = np.clip(pred_y, 0, self.T)
            prediction_error = pred_y - encoded_Y[e];

            update_p = (1.0 * prediction_error / self.T) ** 2

            if pred_y < encoded_Y[e]:
                self.clause_bank.type_i_feedback(update_p, self.s, self.boost_true_positive_feedback, clause_active,
                                                 self.encoded_X_train, e)
                if self.weighted_clauses:
                    self.weight_bank.increment(clause_outputs, update_p, clause_active, False)
            elif pred_y > encoded_Y[e]:
                self.clause_bank.type_ii_feedback(update_p, clause_active, self.encoded_X_train, e)
                if self.weighted_clauses:
                    self.weight_bank.decrement(clause_outputs, update_p, clause_active, False)
        return

    def predict(self, X):
        if not np.array_equal(self.X_test, X):
            self.encoded_X_test = self.clause_bank.prepare_X(
                tmu.tools.encode(X, X.shape[0], self.number_of_patches, self.number_of_ta_chunks, self.dim,
                                 self.patch_dim, 0))
            self.X_test = X.copy()

        Y = np.ascontiguousarray(np.zeros(X.shape[0]))
        for e in range(X.shape[0]):
            clause_outputs = self.clause_bank.calculate_clause_outputs_predict(self.encoded_X_test, e)
            pred_y = np.dot(self.weight_bank.get_weights(), clause_outputs).astype(np.int32)
            Y[e] = 1.0 * pred_y * (self.max_y - self.min_y) / (self.T) + self.min_y
        return Y

    def get_weight(self, clause):
        return self.weight_bank.get_weights()[clause]

    def set_weight(self, clause, weight):
        self.weight_banks.get_weights()[clause] = weight


class TMSSClassifier(TMClassifier):
    '''The class is responsible to learn multiclasses while their ground truthes are absent. The mechanism is keep dividing the data into two groups (binary classification) using the original TM
    where that is done recursivly untill we break the data into smaller classes till no further separation needed. For example 20 classes dataset will be consider two classes, each class associated data samples will be broken into another two and so on until the data being examined is only distributed in either self.g_A, or self.g_B means there is no needed separation. The too many classes represent separating unique subpatterns in the data. later on, some of these separated classes (subpatterns) can be coupled again using the associated clauses to construct the final classes in the data in cases some classes are sharing more than one of these subpatterns'''
    
    def __init__(self, number_of_clauses, T, s, platform='CPU', patch_dim=None, boost_true_positive_feedback=1,
                 number_of_state_bits=8, weighted_clauses=False, clause_drop_p=0.0, literal_drop_p=0.0,
                 pattern_search_exit=0.8, epsilon = 0.5, reset_guess_threshold = 500):
        
        self.pattern_search_exit = pattern_search_exit
        self.epsilon = epsilon
        self.reset_guess_threshold = reset_guess_threshold
        self.g_A = []
        self.g_B = []
        self.g_non = []
        self.grouped_samples = {}
        self.interpretability_clauses = {}
        
        self.number_of_clauses = number_of_clauses
        self.T = T
        self.s = s
        self.platform = platform
        self.patch_dim = patch_dim
        self.number_of_state_bits = number_of_state_bits
        self.weighted_clauses = weighted_clauses
        self.clause_drop_p = clause_drop_p
        self.literal_drop_p = literal_drop_p
        self.boost_true_positive_feedback = boost_true_positive_feedback

    def initialize_coreTM(self, X, Y):
        '''this method reintialize the core parent: TM multi classifer each time we conduct a binary classification on the data during the recursion '''
        super().__init__(self.number_of_clauses, self.T, self.s, platform=self.platform, patch_dim=self.patch_dim,
                         boost_true_positive_feedback=self.boost_true_positive_feedback,
                         number_of_state_bits=self.number_of_state_bits, weighted_clauses=self.weighted_clauses,
                         clause_drop_p=self.clause_drop_p, literal_drop_p=self.literal_drop_p)
        super().initialize(X, Y)

    def check_permutated_gt(self, preds, Y_test_permutated):
        '''since classes ground truthes are unknown, then the TMSSClassifer will give guessed labels values to samples, final predictions might use class labels that are not matched with test data labels (known),
        so we permutate test data labels to test for all possible assigned labels. for example some samples in the test will have gt:1 but in the prediction they were assigned 2.
        if test samples assigned 1 but in prediction were assigned 2, then classification was done completly correct and only different label was given, 2 instead of 1. Usually this function will run multiple
        times on all possible permutations of test sample labels. The maximum accuracy among these multiple runs will be considered the peformance measure of the TMSSClassifier'''

        accuracy = (preds == Y_test_permutated).mean()
        print('--------------------------------------')
        print('Accuracy on Permutated Test Data: ', accuracy)
        print('--------------------------------------')
        return accuracy

    def remove_contradiction_0(self, class_clauses_dict):
        '''ensures that a class pos or neg clause has no contradicted literals. for example: x1 and not x1 in the same aggregated class clause either pos or neg.'''

        all_literals = [l for c in class_clauses_dict.values() for l in c]
        processed = []
        contradicted = []
        for l in collections.Counter(all_literals).items():
            for l_ in collections.Counter(all_literals).items():
                if l == l_:
                    continue
                if l[0][1:] == l_[0][1:] and l_[0] not in processed:
                    idx = [l[1], l_[1]].index(max([l[1], l_[1]]))
                    if l[1] == l_[1]:
                        contradicted.append(l[0])
                        contradicted.append(l_[0])
                        continue
                    if idx == 0:
                        processed.append(l[0])
                    else:
                        processed.append(l_[0])

                    contradicted.append(l_[0])

        for l in contradicted:
            if l in all_literals:
                all_literals.remove(l)

        return set(all_literals)

    def get_final_clause(self, class_clauses_dict):
        '''calls the generate logic function'''
        literals = self.remove_contradiction_0(class_clauses_dict)

        return self.generate_logic(literals)

    def generate_logic(self, literals):
        '''the generate logoc helper function returns a clause with the ∧ operator so other functions can use the clauses further with a proper expected format '''
        final_clause = ""
        for e, l in enumerate(literals):
            if e != len(literals) - 1:
                final_clause += l + " ∧ "
            else:
                final_clause += l
        return final_clause

    def remove_contradiction_1(self, class_clauses_dict_1, class_clauses_dict_2):
        '''ensures there will not be contradictions between two or more  classes pos/neg clauses or between pos and neg clauses in the same class. always literals must be opposite x1 and not x1 for instance'''
        final_1 = self.get_final_clause(class_clauses_dict_1)
        final_2 = self.get_final_clause(class_clauses_dict_2)

        list_final_1 = final_1.split(" ∧ ")
        list_final_2 = final_2.split(" ∧ ")
        to_remove = []
        if list_final_1[0] != '' and list_final_2[0] != '':
            for l in list_final_1:
                for l_ in list_final_2:
                    if l == l_:
                        to_remove.append(l)

            for r in to_remove:
                list_final_1.remove(r)
                list_final_2.remove(r)

        return self.generate_logic(list_final_1), self.generate_logic(list_final_2)

    def remove_contradiction_2(self, class_clauses_dict_1, class_clauses_dict_2):
        '''ensures there will not be contradictions between pos and neg clauses in two or more classes. for example x1 in pos cls 1 cannot be also not x1 in neg cls 2. That ensure distinct literals per classes clauses always.'''

        final_1 = self.get_final_clause(class_clauses_dict_1)
        final_2 = self.get_final_clause(class_clauses_dict_2)

        list_final_1 = final_1.split(" ∧ ")
        list_final_2 = final_2.split(" ∧ ")
        to_remove = set()
        if list_final_1[0] != '' and list_final_2[0] != '':
            for l in list_final_1:
                for l_ in list_final_2:
                    if l[1:] == l_[1:] and l[0] != l_[0]:
                        to_remove.add(l)
                        to_remove.add(l_)

            for r in to_remove:
                if r in list_final_1:
                    list_final_1.remove(r)
                if r in list_final_2:
                    list_final_2.remove(r)

        return self.generate_logic(list_final_1), self.generate_logic(list_final_2)

    def convert_to_dict_clauses(self, str_clause):
        '''a helper function tha converts a clause with the ∧ operator to a dictionary so other functions can process with the expected proper format'''
        list_clause = str_clause.split(" ∧ ")
        return {0: list_clause}

    
    
    def tune_information(self, old_pattern_1, old_pattern_2, new_pattern_1, new_pattern_2):
        '''an important function that makes sure that accumulated patterns of the two hypothesized classes are being accumulated to their previouse patterns in a correct way. Sometimes pattern A and B is being assigned for classes A and B but due to the noise in labels, maybe next training on updated labels will result in swapped clauses patterns for A, B which cause the accumulation to be contradicted, old A+ new B and old B + new A. hence, we apply the contradiction function to check the two patterns (new, old) with least contradiction so we add them together in the process.'''

        if  old_pattern_1 == [] and old_pattern_2 == []:
            return new_pattern_1, new_pattern_2


        final_A, final_B = [], []
        logical_clause = self.generate_logic(old_pattern_1 + new_pattern_1)
        dict_clause = self.convert_to_dict_clauses(logical_clause)
        pattern_a = list(self.remove_contradiction_0(dict_clause))
        
        logical_clause = self.generate_logic(old_pattern_1 + new_pattern_2)
        dict_clause = self.convert_to_dict_clauses(logical_clause)
        pattern_b = list(self.remove_contradiction_0(dict_clause))
        
        if len(pattern_a) >= len(pattern_b):
            final_A = pattern_a
            
            logical_clause = self.generate_logic(old_pattern_2 + new_pattern_2)
            dict_clause = self.convert_to_dict_clauses(logical_clause)
            final_B = list(self.remove_contradiction_0(dict_clause))
        else:
            final_A = pattern_b
            
            logical_clause = self.generate_logic(old_pattern_2 + new_pattern_1)
            dict_clause = self.convert_to_dict_clauses(logical_clause)
            final_B = list(self.remove_contradiction_0(dict_clause))
        
        
        return final_A, final_B
            

    
    def vote(self, final_pattern_A, final_pattern_B, sample_feats):
        '''this function votes for whether the passed data sample is class A or class B based on the given accumulated two patterns a binary classifier original TM tries to learn 
        from guessed labels. The patterns are the clauses and are being accumulated since training is repeated till a group of Learning Automata is not being penalized anymore. Each LA is assigned to a sample to learn its class, first by guessing, then by evaluating a reward function with regard to the patterns being accumulately learned.'''
        
        vote_A, vote_B = 0, 0
        for e, l in enumerate(sample_feats):
            if l == 0:
                continue
            for p in final_pattern_A:
                if p != '':
                    if e == int(p[2:]):
                        if p[0] == ' ':
                            vote_A += 1
                        elif p[0] == '¬':
                            vote_B += 1
                        
        for e, l in enumerate(sample_feats):
            if l == 0:
                continue
            for p in final_pattern_B:
                if p != '':
                    if e == int(p[2:]):
                        if p[0] == ' ':
                            vote_A -= 1
                        elif p[0] == '¬':
                            vote_B -= 1

        if vote_A > vote_B:
            return 0
        elif vote_A < vote_B:
            return 1
        else:
            return 2

    def reward(self, la_id, la, X_train, final_pattern_A, final_pattern_B):
        '''the reward function that evaluates accumulated patterns with regard to the LA decided label for each associated data sample. The function also updates the current learned two classes group 
        of data samples by updating the class members g_A, g_B, g_non where we use the information stored in them later. these grouping class attributes are being filled only with the current two classes being learned 
        and will not include all classes. For example if data has 20 classes.'''
        v = self.vote(final_pattern_A, final_pattern_B, X_train[la_id])
        s = ""
        sample_list = list(X_train[la_id])
        sample = s.join([str(elem) for elem in sample_list])
        if v == 0:
            self.g_A.append(sample)
        elif v == 1:
            self.g_B.append(sample)
        else:
            self.g_non.append(sample)

        if v == la.get_label():
            la.reward()
            return 1
            
        else:
            if random.random() <= self.epsilon: # here we do not penalize all the LAS if they all were mistaken to avoid symetric error loops while labeling the data. It can be considered as sampling the labels.  
                la.penalize()
                return 0
            return 1

    def labels_feedback(self, las, X_train, final_pattern_A, final_pattern_B, pattern_search_exit):
        '''computes either penalized or rewarded associated LAs with data samples by calling the reward function above. The function uses a pattern_search_exit parameter to allow
        exiting the TM binary classifier training repeatition if enough number of LAs are being rewarded together after one single training. That helps for fast convergence if necessary while some penalities (wrong labels)
        cannot harm the accuracy of classification since TM can still classify with some noise in the data.'''
        
        print('Computing Feedbacks...')
        c = 0
        for la_id, la in enumerate(las):
            c += self.reward(la_id, la, X_train, final_pattern_A, final_pattern_B)
        print(len(X_train) - c, ' Penalized Automata out of ', len(X_train))
        if c >= len(X_train) * pattern_search_exit:
            return 1
        else:
            return 0

    def clean_learned_patterns(self, num_clauses, num_features):
        '''extracts all learned clauses and clean all contradictions by elimenating literals that conflicts with each others. for example: not x1 and x1 cannot found together
        in pos clauses of a class or neg clauses of a class. x1 and x1 cannot be part of a class pos clause and at same time its neg clause.'''

        all_classes_patterns = []
        for x in range(2):
            class_pos_clauses = {}
            class_neg_clauses = {}
            for j in range(num_clauses // 2):
                l = []
                for k in range(num_features * 2):
                    if self.get_ta_action(x, 0, j, k) == 1:
                        if k < num_features:
                            l.append(" x%d" % (k))
                        else:
                            l.append("¬x%d" % (k - num_features))

                class_pos_clauses.update({j: l})

            for j in range(num_clauses // 2):
                l = []
                for k in range(num_features * 2):
                    if self.get_ta_action(x, 1, j, k) == 1:
                        if k < num_features:
                            l.append(" x%d" % (k))
                        else:
                            l.append("¬x%d" % (k - num_features))

                class_neg_clauses.update({j: l})

            all_classes_patterns.append((class_pos_clauses, class_neg_clauses))
            print('Class ', x, ' Pattern Has Collected.')

        '''for e, (class_pos, class_neg) in enumerate(all_classes_patterns):
            print('Class: ', e)
            print('pos: ', class_pos)
            print('neg: ', class_neg)'''

        print('Clean Contradiction in Classes Patterns Have Started....')
        # we will aggregate all pos clauses for each class to be one longe pos clause. Same for neg. While doing this we remove duplicates and contradictions between pos and neg if found
        aggregated_classes_clauses = []
        for class_pos, class_neg in all_classes_patterns:
            pos_aggr_clause, neg_aggr_clause = self.remove_contradiction_1(class_pos, class_neg)
            aggregated_classes_clauses.append((pos_aggr_clause, neg_aggr_clause))

        # now we remove contradictions between each aggregated pos clause for a class with all other classes clauses (pos and neg) to obtain completly distict one aggregated pos and neg clause for each class
        cleaned_classes_clauses = []
        for e, (class_i_pos, class_i_neg) in enumerate(aggregated_classes_clauses):
            pos_cl_i = class_i_pos
            neg_cl_i = class_i_neg

            for e2, (class_j_pos, class_j_neg) in enumerate(aggregated_classes_clauses):
                if e != e2:
                    pos_cl_i, pos_cl_j = self.remove_contradiction_1(self.convert_to_dict_clauses(pos_cl_i),
                                                                     self.convert_to_dict_clauses(class_j_pos))
                    neg_cl_i, neg_cl_j = self.remove_contradiction_1(self.convert_to_dict_clauses(neg_cl_i),
                                                                     self.convert_to_dict_clauses(class_j_neg))

                    pos_cl_i, _ = self.remove_contradiction_2(self.convert_to_dict_clauses(pos_cl_i),
                                                              self.convert_to_dict_clauses(neg_cl_j))
                    neg_cl_i, _ = self.remove_contradiction_2(self.convert_to_dict_clauses(neg_cl_i),
                                                              self.convert_to_dict_clauses(pos_cl_j))

            cleaned_classes_clauses.append((pos_cl_i, neg_cl_i))

        return cleaned_classes_clauses

    def guess_labels(self, samples_num):
        '''a function that initializes guess labels based on random selection of the associated LAs to data samnples'''
        las = []
        Y = np.zeros(samples_num)
    
        for i in range(samples_num):
            la = LA_2(100)
            las.append(la)
            Y[i] = la.get_label()
        print('----------------------------')
        print('Labels Were Guessed By LAs.')
        print('----------------------------')

        return las, Y

    def fetch_patterns(self, num_clauses, num_features, X, Y, las):
        '''train on hypothesized number of classes (2) to get some non-contradiced sub patterns for each.'''

        # for i in range(num_iterations):
        while True:
            super().fit(X, Y)
            print('Initial Training Has Finished.')

            cleaned_patterns = self.clean_learned_patterns(num_clauses, num_features)
            if (cleaned_patterns[0][0] == '' and cleaned_patterns[1][1] == '') or (cleaned_patterns[1][0] == '' and cleaned_patterns[0][1] == ''):
                print('NOT ENOUGH PATTERN DISCOVERED --> REPEATING INITIAL TRAINING')
                las, Y = self.guess_labels(len(X))
                continue
            
            group1_final_pattern = cleaned_patterns[0][0].split(" ∧ ") + cleaned_patterns[1][1].split(
                " ∧ ")  # positive clause from hypothesized class 1 + negative clause from hypothesized class 2
            group2_final_pattern = cleaned_patterns[1][0].split(" ∧ ") + cleaned_patterns[0][1].split(
                " ∧ ")  # positive clause from hypothesized class 2 + negative clause from hypothesized class 1
            if group1_final_pattern != [''] and group2_final_pattern != ['']:
                return group1_final_pattern, group2_final_pattern, las
            
 

    def get_current_labels(self, las):
        '''retrieve the current decided label from an LA associated to a data sample'''
        Y = np.zeros(len(las))
        for i, la in enumerate(las):
            Y[i] = la.get_label()

        return Y
    
    def get_x_train(self, feat_size):
        '''retrieve the current two separated group of data after the binary classification task was done. We retrievethat from self.g_A, and self.g_B'''
        x1_train = np.zeros((len(self.g_A), feat_size), dtype=object)
        x2_train = np.zeros((len(self.g_B), feat_size), dtype=object)

        for i ,sample_string in enumerate(self.g_A):
            x1_train[i] = [int(feat) for feat in sample_string]
            
        for i ,sample_string in enumerate(self.g_B):
            x2_train[i] = [int(feat) for feat in sample_string]
            
        return x1_train, x2_train
                

    def reset_grouping(self):
        '''reset the groups g_A, g_B and g_non, during the recursive training so each time we split data into two classes, we 
        get an empty place holders for the potential groups which we will use to separate data into two classes'''
        
        self.g_A = []
        self.g_B = []
        self.g_non = []
        
    
    def evaluate_grouping(self, data_size):
        '''evaluate during the TMSSClassifier training if one of the grouping attemps were stuck in bad grouping where either overlappeing between groups occurred or non grouped samples at all exist.'''
        false_grouping = 0
        for sample_string in self.g_A:
            if sample_string in self.g_B:
                false_grouping += 1

        print('Percentage of False Grouping: ', (false_grouping+len(self.g_non)) / data_size)
        print('Not Grouped Samples: ', len(self.g_non))

    
    def get_cluster_info(self, data_all_patterns, clustered_samples):
        '''After TMSSClassifier training is done, we evaluate the subpatterns of the data if they were exclusive to the classes they should be or not. That function usually used with synthetic data
         where we know which subpatterns should be related to which class. For example two sub patterns could not be part of one classified data group.'''
        patterns_info = {}
        for e, p in enumerate(data_all_patterns):
            counter = 0
            for s in clustered_samples:
                strng = ''
                p_str = strng.join([str(elem) for elem in p])
                s_str = strng.join([str(elem) for elem in s])

                if p_str == s_str:
                    counter += 1
                    patterns_info.update({e:counter})
        
        for k, v in patterns_info.items():
            print('Pattern ', k, ' Occurrence Percentage: ', round(v/len(clustered_samples), 4))
                
    
    def fit(self, num_clauses, num_features, all_X):
        '''train the TMSSClassifier object. Perform the recursive grouping (breaking data into two classes recursivly)'''
        
        for X in all_X:
            las, Y = self.guess_labels(len(X))
            self.initialize_coreTM(X, Y)
            final_pattern_A, final_pattern_B = [], []
            signal = 0
            counter = 0
            while signal == 0:
                try:
                    print('Aquiring Pattern... Attempt #:', counter)
                    self.reset_grouping()
                    current_pattern_a, current_pattern_b, las = self.fetch_patterns(num_clauses, num_features, X, Y, las)
                    final_pattern_A, final_pattern_B = self.tune_information(final_pattern_A, final_pattern_B, current_pattern_a, current_pattern_b)
                    signal = self.labels_feedback(las, X, final_pattern_A, final_pattern_B, self.pattern_search_exit)
                    counter += 1
                    if counter > self.reset_guess_threshold:
                        las, Y = self.guess_labels(len(X))
                        counter = 0

                except Exception as e:
                    print('Final tuned_pattern A was: ', final_pattern_A)
                    print('Final tuned_pattern B was: ', final_pattern_B)

                    print(e)
                    exit()
                    

                Y = self.get_current_labels(las)
                self.evaluate_grouping(len(X))
                x1, x2 = self.get_x_train(num_features)
                print('\n¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤')

            if len(x1) != 0 and len(x2) != 0:
                new_all_X = [x1, x2]
                self.fit(num_clauses, num_features, new_all_X)
            else:
                random_key = str(random.random() * random.random())
                if len(x1) == 0:
                    self.grouped_samples.update({random_key: x2})
                else:
                    self.grouped_samples.update({random_key: x1})
                print('------------------------------------')
                print('Sub Patterns Found So Far:', len(self.grouped_samples))
                print('------------------------------------')
                print()
            





