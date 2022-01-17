#!/usr/bin/env python3

import pickle
import numpy as np
import os
import itertools
from abc import ABC, abstractmethod
from sklearn.svm import SVC, LinearSVC
import math


class SecondaryStructureModel(ABC):

    @abstractmethod
    def fit(self, x, y):
        pass


    @abstractmethod
    def predict(self, x):
        pass


    @staticmethod
    def build_xy(dataset):
        x = []
        y = []
        names = []
        for key in dataset.keys():
            sample = dataset[key]
            x.append(sample['profile'])
            y.append(sample['ss'])
            names.append(sample)
            assert x[-1].shape[0] == len(y[-1])
        return (x, y, names)


    @staticmethod
    def evaluate_prediction(y_true, y_hat):
        predictions = {'H': {}, 'E': {}, '-': {}}
        metrics = {'H': {}, 'E': {}, '-': {}}
        for k in predictions.keys():
            predictions[k] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        acc = [0, 0]
        assert len(y_hat) == len(y_true)
        for i, y_t in enumerate(y_true):
            y_h = y_hat[i]
            assert len(y_h) == len(y_t)
            for j in range(len(y_h)):
                if y_h[j] == y_t[j]:
                    acc[0] += 1
                else:
                    acc[1] += 1
                for ss in predictions.keys():
                    if y_h[j] == ss:
                        if ss == y_t[j]:
                            predictions[ss]['tp'] += 1
                        else:
                            predictions[ss]['fp'] += 1
                    else:
                        if ss != y_t[j]:
                            predictions[ss]['tn'] += 1
                        else:
                            predictions[ss]['fn'] += 1

        for ss in predictions.keys():
            metrics[ss]['sensitivity'] = predictions[ss]['tp'] / max(0.0001, float(predictions[ss]['tp'] +
                                                                       predictions[ss]['fn']))
            metrics[ss]['precision'] = predictions[ss]['tp'] / max(0.00001, float(predictions[ss]['tp'] +
                                                                       predictions[ss]['fp']))
            metrics[ss]['mcc'] = ((predictions[ss]['tp'] * predictions[ss]['tn']) -\
                                  (predictions[ss]['fp'] * predictions[ss]['fn'])) / \
                                  max(0.0001, float(math.sqrt((predictions[ss]['tp'] + predictions[ss]['fp']) *
                                                  (predictions[ss]['tp'] + predictions[ss]['fn']) *
                                                  (predictions[ss]['tn'] + predictions[ss]['fp']) *
                                                  (predictions[ss]['tn'] + predictions[ss]['fn']))))
            metrics[ss]['accuracy'] = (predictions[ss]['tn'] + predictions[ss]['tp']) / \
                max(0.00001, float(predictions[ss]['tp'] + predictions[ss]['fp'] + predictions[ss]['tn'] + predictions[ss]['fn']))
        metrics['accuracy'] = acc[0] / float(sum(acc))
        return acc[0] / float(sum(acc)), metrics


    @staticmethod
    def crossvalidation(X_cv, Y_cv, model_class, args=(17,), evaluate=True,
                        models_dir="../models/", label="model", use_serialized_models=False):
        print(label)
        model = model_class(*args)
        assert len(X_cv) == len(Y_cv)
        k = len(X_cv)
        results = []
        ss_list = ['H', 'E', '-']
        metrics = {'accuracy':0, 'H': {}, 'E': {}, '-': {}}
        for ss in ss_list:
            metrics[ss]['sensitivity'] = 0
            metrics[ss]['accuracy'] = 0
            metrics[ss]['precision'] = 0
            metrics[ss]['mcc'] = 0
        for i in range(0,k):
            print("Cross valid stage : " + str(i + 1))
            x_train, y_train = list(X_cv), list(Y_cv)
            x_test = x_train.pop(i)
            y_test = y_train.pop(i)
            x_train, y_train = list(itertools.chain.from_iterable(x_train)),\
                list(itertools.chain.from_iterable(y_train))

            assert len(x_test) + len(x_train) == len(list(itertools.chain.from_iterable(X_cv)))
            if not os.path.isfile(models_dir + label + "_CV" + str(i) + ".pickle"):

                model.fit(x_train, y_train)

                # Serialize model
                if use_serialized_models:
                    with open(models_dir + label + "_CV" + str(i) + ".pickle", "wb") as f:
                        pickle.dump(model, f)

                        print("Model saved in " + models_dir)

            elif use_serialized_models:
                with open(models_dir + label + "_CV" + str(i) + ".pickle", "rb") as f:
                    model = pickle.load(f)
                    print("Model read")



            y_hat = model.predict(x_test)
            cv_result = SecondaryStructureModel.evaluate_prediction(y_test, y_hat)
            SecondaryStructureModel.pretty_print_metrics(cv_result[1])
            results.append(cv_result)
        for i in range(k):
            for ss in ss_list:
                metrics[ss]['sensitivity'] += results[i][1][ss]['sensitivity']
                metrics[ss]['precision'] += results[i][1][ss]['precision']
                metrics[ss]['accuracy'] += results[i][1][ss]['accuracy']
                metrics[ss]['mcc'] += results[i][1][ss]['mcc']

            metrics['accuracy'] += results[i][0]
        for ss in ss_list:
            metrics[ss]['sensitivity'] /= float(k)
            metrics[ss]['precision'] /= float(k)
            metrics[ss]['accuracy'] /= float(k)
            metrics[ss]['mcc'] /= float(k)
        metrics['accuracy'] /= float(k)
        SecondaryStructureModel.pretty_print_metrics(metrics)
        return metrics


    @staticmethod
    def pretty_print_metrics(metrics):
        ss = ['H', 'E', '-']
        if "accuracy" in metrics:
            print("Q3 Accuracy : " + str(metrics["accuracy"]))
        print("\n\nSS\tAcc\tMCC\tPre\tSen")
        for s in ss:
            print(s + "\t" + str(round(metrics[s]["accuracy"], 4)) + "\t"
                  + str(round(metrics[s]["mcc"],4)) +
                  "\t" + str(round(metrics[s]["precision"],4)) +
                  "\t" + str(round(metrics[s]["sensitivity"],4)))


    @staticmethod
    def build_cv_xy(dataset, cv_path="../data/cv/", toy=False):
        X_cv = []
        Y_cv = []
        for filename in os.listdir(cv_path):
            with open(cv_path + '/' + filename) as f:
                set_ids = f.read().splitlines()
            x = []
            y = []
            for key in set_ids:
                if key not in dataset:
                    # print(key)
                    continue
                sample = dataset[key]
                x.append(sample['profile'])
                y.append(sample['ss'])
                assert x[-1].shape[0] == len(y[-1])
            if toy:
                x = x[:len(x) // 10]
                y = y[:len(y) // 10]
            print(len(x))
            X_cv.append(x)
            Y_cv.append(y)

        return (X_cv, Y_cv)



class GorModel(SecondaryStructureModel):
    def __init__(self, window=17, *args):
        self.window = window
        self.ss = ['H', 'E', '-']
        self.residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.parameters = {}
        for a in self.ss:
            self.parameters[a] = np.zeros((self.window,20))
        self.parameters["aa"] = np.zeros((self.window,20))
        self.parameters["ss"] = [0, 0, 0]


    def fit(self, x, y):
        assert len(x) == len(y)
        print("Fitting the model")
        n = 0
        # Cycle each protein profile
        for idx, profile in enumerate(x):
            seq_len = profile.shape[0]
            n += seq_len
            ss = y[idx]  # Secondary structures relative to the given sequence
            # For each value, compute the window and update parameters
            for i in range(seq_len):
                d = self.window // 2
                # Should add 0 valued columns to the left(start) of the window matrix
                if i - d < 0:
                    padding = np.zeros((d - i, 20))
                    w = profile[:i + d + 1]
                    w = np.concatenate((padding, w))
                # Sould add 0 valued columns to the right(end) of the window matrix
                elif i + d >= seq_len:
                    padding = np.zeros((i + d - seq_len + 1, 20))
                    w = profile[i - d:]
                    w = np.concatenate((w, padding))
                # Sliding window trivially selectable
                else:
                    w = profile[i - d:i + d + 1]
                assert w.shape == (self.window, 20)

                # Update combined probability
                self.parameters[ss[i]] += w
                # Update residue margin probability
                self.parameters["aa"] += w
                # Update SS margin probability
                self.parameters["ss"][self.ss.index(ss[i])] += 1
        # Compute sum of rows of margin res probabilities, to perform matrix norm
        n_ss = np.sum(self.parameters["aa"], 1)

        # Normalize residue margin probabilities
        self.parameters["aa"] /= n # n_ss[:, None]
        for a in self.ss:
            # Normalize residue,ss combined probabilities
            self.parameters[a] /= n # np.divide(self.parameters[a], n_ss[:, None])
            # Normalize ss margin probabilities
            self.parameters["ss"][self.ss.index(a)] /= n

            # Compute the information function
            self.parameters[a] = np.divide(self.parameters[a],
                                             ((self.parameters["ss"][self.ss.index(a)]) *
                                              self.parameters["aa"]))
            self.parameters[a] = np.log(self.parameters[a])
        print("Model training finished")


    def predict(self, x):
        print("Predicting")
        y = []
        for idx, profile in enumerate(x):
            y_ss = ""
            seq_len = profile.shape[0]
            for i in range(seq_len):
                d = self.window // 2
                # Build Window
                if i - d < 0:
                    padding = np.zeros((d-i, 20))
                    w = profile[:i + d + 1]
                    w = np.concatenate((padding, w))
                elif i + d >= seq_len:
                    padding = np.zeros((i + d - seq_len + 1, 20))
                    w = profile[i - d:]
                    w = np.concatenate((w, padding))
                else:
                    w = profile[i - d : i + d + 1]
                assert w.shape[0] == self.window

                preds = [0, 0, 0]
                # Compute prediction for each ss
                for a in self.ss:
                    preds[self.ss.index(a)] = np.sum(w * self.parameters[a])
                # Select best prediction
                pred = self.ss[preds.index(max(preds))]
                y_ss += pred
            assert len(y_ss) == seq_len
            y.append(y_ss)
        assert len(y) == len(x)
        return y





class SVM(SecondaryStructureModel):
    def __init__(self, window=17, C=2, gamma=0.5):
        self.window = window
        self.ss = ['H', 'E', '-']
        # self.model = SVC(C=C, gamma=gamma)
        self.model = LinearSVC()

    def fit(self, x, y):
        assert len(x) == len(y)
        print("Fitting the model")
        assert len(x) == len(y)
        x_w = []
        y_ss = []
        for idx, profile in enumerate(x):
            seq_len = profile.shape[0]
            seq_ss = y[idx]
            for i in range(seq_len):
                ss = seq_ss[i]
                d = self.window // 2
                if i - d < 0:
                    padding = np.zeros((d-i, 20))
                    w = profile[:i + d + 1]
                    w = np.concatenate((padding, w))
                elif i + d >= seq_len:
                    padding = np.zeros((i + d - seq_len + 1, 20))
                    w = profile[i - d:]
                    w = np.concatenate((w, padding))
                else:
                    w = profile[i - d : i + d + 1]
                assert w.shape[0] == self.window

                x_w.append(w.flatten())

                y_ss.append(self.ss.index(ss))
        print("N  samples " + str(len(x_w)))
        print("N  features " + str(len(x_w[0])))
        self.model.fit(x_w, y_ss)



    def predict(self, x):
        print("Predicting")
        y = []
        for idx, profile in enumerate(x):
            y_ss = ""
            seq_len = profile.shape[0]
            for i in range(seq_len):
                d = self.window // 2
                if i - d < 0:
                    padding = np.zeros((d-i, 20))
                    w = profile[:i + d + 1]
                    w = np.concatenate((padding, w))
                elif i + d >= seq_len:
                    padding = np.zeros((i + d - seq_len + 1, 20))
                    w = profile[i - d:]
                    w = np.concatenate((w, padding))
                else:
                    w = profile[i - d : i + d + 1]
                assert w.shape[0] == self.window
                x_w = w.flatten()
                preds = list(self.model.decision_function([x_w])[0])
                y_ss += self.ss[preds.index(max(preds))]
            y.append(y_ss)
        assert len(y) == len(x)
        return y






TRAIN_DATASET = "../data/datasets/training_set.pickle"
BLIND_DATASET = "../data/datasets/blindset.pickle"

def main():
    with open(TRAIN_DATASET, 'rb') as f:
        training_dataset = pickle.load(f)
        print(len(training_dataset))
    with open(BLIND_DATASET, 'rb') as f:
        blind_dataset = pickle.load(f)
        empty_profiles = 0
        one_hot_blind = {}
        for key in blind_dataset:
            entry = blind_dataset[key]
            if entry['empty_profile']:
                one_hot_blind[key] = entry
                empty_profiles += 1
        print("Empty profiles " + str(empty_profiles))

    
    
    # Perform CV
    print("Crossvalidation")
    X_cv, Y_cv = SecondaryStructureModel.build_cv_xy(training_dataset, toy=False)
    metrics = SecondaryStructureModel.crossvalidation(X_cv, Y_cv, GorModel, args=(17,), label="gor")




    # Perform BlindSet test
    model = GorModel()
    # with open("../models/MODEL.pickle", 'rb') as f:
        # model = pickle.load(f)
    x_train, y_train, names = model.build_xy(training_dataset)
    model.fit(x_train, y_train)
    x_blind, y_blind, _ = model.build_xy(blind_dataset)
    x_blind_zero, y_blind_zero, _ = model.build_xy(one_hot_blind)
    y_hat = model.predict(x_blind)
    y_hat_zero = model.predict(x_blind_zero)
    print(model.evaluate_prediction(y_blind, y_hat))
    print(model.evaluate_prediction(y_blind_zero, y_hat_zero))


if __name__ == "__main__":
    main()
