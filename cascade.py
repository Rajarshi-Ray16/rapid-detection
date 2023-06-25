from adaboost import * 



class Cascade:
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training):
        pos, neg = [], []
        for ex in training:
            if ex[1] == 1:
                pos.append(ex)
            else:
                neg.append(ex)
        
        for feature_num in self.layers:
            if len(neg) == 0:
                print("Stopping early. FPR = 0")
                break
            clf = vclassifier(feature_num)
            clf.train(pos+neg, len(pos), len(neg))
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives

    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1