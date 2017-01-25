
## this is just a test!
# Create first network with Keras

from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy

# fix random seed for reproducibility

def get_data():
    return numpy.loadtxt('preprocessed.csv',delimiter=",", skiprows=2)
    
seed =7

numpy.random.seed(seed)

## load a dataset

dataset = get_data()   #numpy.loadtxt("pima.csv", delimiter=",")
# split into input (X) and output (Y) variables

rc,cc = dataset.shape
cc = cc-1
to = int(numpy.round(rc*0.75,0))

Xt = dataset[0:to,0:cc]
Yt = dataset[0:to,cc]

Xte = dataset[to:,0:cc]
Yte = dataset[to:,cc]



def create_model():
	# create model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2, activity_l2
    
    model = Sequential()
    model.add(Dense(20, input_dim=16, init='uniform', activation='relu'))
    model.add(Dense(15, init='uniform', activation='relu'))
    model.add(Dense(12, init='uniform', activation='relu'))
    model.add(Dense(8, W_regularizer=l2(0.03), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(4, init='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.3))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def dl_part(Xt,Yt,Xte,Yte):
    
    model = create_model();
    # Fit the model
    print ("Training Deep learning algorithm..")
    model.fit(Xt, Yt, nb_epoch=10, batch_size=5, verbose=1)
    # evaluate the model
    scores = model.evaluate(Xt, Yt)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    ### then predict for the test set!


    predictions = model.predict(Xte)


    print (accuracy_score(Yte,predictions), "% is final accuracy! Deep learning..")



def svm_part(Xt,Yt,Xte,Yte):    
    ## support vectors
    from sklearn import svm
    print ("Training SVM..")
    clf = svm.SVC()
    clf.fit(Xt,Yt)

    predictions_svm = clf.predict(Xte)


    print (accuracy_score(Yte,predictions_svm), "% is final accuracy! SVM..")

#svm_part(Xt,Yt,Xte,Yte)    


def gbm_part(Xt,Yt,Xte,Yte):    
    ## support vectors
    from sklearn.ensemble import GradientBoostingClassifier
    print ("Training GBM..")
    clf = GradientBoostingClassifier(n_estimators=1500, learning_rate=1.0, max_depth=2, random_state=0).fit(Xt,Yt)
#    clf.fit(Xt,Yt)

    predictions_gbm = clf.predict(Xte)


    print (accuracy_score(Yte,predictions_gbm), "% is final accuracy! GBM..")

#gbm_part(Xt,Yt,Xte,Yte)



def nnet_part(Xt,Yt,Xte,Yte): 
    ## support vectors
    from sklearn.neural_network import MLPClassifier
    print ("Training MLPC..")
    clf = MLPClassifier(random_state=1, hidden_layer_sizes=(15,5)).fit(Xt,Yt)
#    clf.fit(Xt,Yt)

    predictions_nnet = clf.predict(Xte)


    print (accuracy_score(Yte,predictions_nnet), "% is final accuracy! NNet..")

dl_part(Xt,Yt,Xte,Yte)
#nnet_part(Xt,Yt,Xte,Yte)
#gbm_part(Xt,Yt,Xte,Yte)
#svm_part(Xt,Yt,Xte,Yte)
