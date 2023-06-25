import numpy as np 
import pickle
import time
import math
from sklearn.feature_selection import SelectPercentile




# for the given test data number of positives, number of negatives



f= open("train.pkl", 'rb') 
training = pickle.load(f)





class vclassifier:
    def __init__(self,epoch):
        self.alphas = []
        self.clfs = []
        self.epoch =  epoch
        
    def train(self,data,positives,negatives):
        train_intergral = []
        weights = []

        for dp in data:
            train_intergral.append(integral_image_cal(dp[0]))
            if dp[1] == 1:
                weights.append(1 / (2 * positives))
            else:
                weights.append(1 / (2 * negatives))
        weights = np.array(weights)

        height,width = data[0][0].shape

        features = []
        for i in range(1,width+1):
            for j in range(1,height+1):

                for a in range(width-i):
                    for b in range(height-j):
                
                        #parts of 2
                        #right-left
                        if a+ 2*i<width:
                            features.append(([(a+i,b,i,j)],[(a,b,i,j)]))
                        #bottom-top
                        if b+ 2*j < height:
                            features.append(( [(a,b,i,j)],[(a,b+j,i,j)]))
                        
                        #parts of 3
                        #middle- (left+right)
                        if a+3*i<width:
                            features.append(([(a+i,b,i,j)],[(a,b,i,j),(a+2*i,b,i,j)]))
                        #middle-(up+bottom)
                        if b+ 3*j < height:
                            features.append(([(a,b+j,i,j)],[(a,b,i,j),(a,b+2*j,i,j)]))

                        #parts of 4

                        if a+ 2*i <width and b+2*j <height:
                            features.append(([(a+i,b,i,j),(a,b+j,i,j)],[(a,b,i,j),(a+i,b+j,i,j)]))
        features = np.array(features)
        print("No of features created ",features.shape)
        
        features = features[:int(len(features)/10)]
        data = data[:int(len(data)/10)]
        X = np.zeros((len(features), len(data)))
        y = np.zeros(len(data))
        print(y.shape)
        for i  in range(len(data)):
            if data[i][1]:
                y[i] =1
        tot = len(features)
        i=0
        for pos_regions, neg_regions in features:
            for j in range(len(data)):
                X[i][j]= self.util(data[j][0],pos_regions,neg_regions)
            i+=1 
            
            print( round(i/tot*100,2), end='\r', flush=True)


        indices = SelectPercentile().fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        

    
        for t in range(self.epoch):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.weak_classifier_train(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            


    def weak_classifier_train(self,X,y,features,weights):
        num_pos=0
        num_neg=0
        l = len(y)
        for i in range(l):
            if y[i] == 1:
                num_pos += weights[i]
            else:
                num_neg += weights[i]
        
        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + num_pos - pos_weights, pos_weights + num_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = weakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def util(self,mat,pos_regions,neg_regions):
        sum=0
        for pos in pos_regions:
            x,y,w,h = pos
            sum+= mat[x+w][y+h]+mat[x][y] - (mat[x][y+h] +  mat[x+w][y])
        for neg in neg_regions:
            x,y,w,h = neg
            sum-= mat[x+w][y+h]+mat[x][y] - (mat[x][y+h] +  mat[x+w][y])
        """for pos in pos_regions:
            x,y,w,h = pos
            sum+= mat[y+h][x+w]+mat[y][x] - (mat[y+h][x] +  mat[y][x+w])
        for neg in neg_regions:
            x,y,w,h = neg
            sum+= mat[y+h][x+w]+mat[y][x] - (mat[y+h][x] +  mat[y][x+w])"""
       
        return sum 
    def classify(self, image):
        total = 0
        ii = integral_image_cal(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

                

def integral_image_cal(mat):
    sh = mat.shape
    integral_image = np.zeros(sh)
    row_sum = np.zeros(sh[1])

    for x in range(sh[0]): 

        for y in range(sh[1]):
            if(y==0):
                row_sum[y] = mat[x][y]
            else:
                row_sum[y] = row_sum[y-1]+mat[x][y]
            if(x==0):
                integral_image[x][y] = row_sum[y] 
            else:
                integral_image[x][y] = integral_image[x-1][y] + row_sum[y] 

    return integral_image





class weakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.pos_regions = positive_regions
        self.neg_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, mat):
        sum=0
        for pos in self.pos_regions:
            x,y,w,h = pos
            sum+= mat[x+w][y+h]+mat[x][y] - (mat[x][y+h] +  mat[x+w][y])
        for neg in self.neg_regions:
            x,y,w,h = neg
            sum-= mat[x+w][y+h]+mat[x][y] - (mat[x][y+h] +  mat[x+w][y])
       
        
        return 1 if self.polarity * sum < self.polarity * self.threshold else 0

def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))






