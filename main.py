import numpy as np 
import pickle
from adaboost import *
from cascade import *



def train_phase_viola(epoch):
    f1= open("train.pkl", 'rb') 
    data_train = pickle.load(f1)
    clf = vclassifier(epoch)
    clf.train(data_train,2429,4548)
    f2 = open("train_model_"+str(epoch)+".pkl","wb")
    pickle.dump(clf,file=f2)

def test_phase_viola(file):
    f1=open(file,"rb")
    f2 = open("test.pkl","rb")
    clf = pickle.load(f1)
    data = pickle.load(f2)
    evaluate(clf,data)

def train_phase_cascade(num_layers):
    f1= open("train.pkl", 'rb') 
    data_train = pickle.load(f1)
    clf=Cascade(num_layers)
    clf.train(data_train)
    evaluate(clf,data_train)
    f2 = open("train_model_cascade_"+str(num_layers)+".pkl","wb")
    pickle.dump(clf,f2)

def test_phase_cascade(file):
    f1 = open(file,"rb")
    f2 = open("test.pkl","rb")
    clf = pickle.load(f1)
    data = pickle.load(f2)
    evaluate(clf,data)



def main():
    epoch = 25
    #train_phase_viola(epoch)
    #test_phase_viola("train_model_"+str(epoch)+".pkl")
    train_phase_cascade([1,5,10])

   

if __name__=="__main__":
    main()
