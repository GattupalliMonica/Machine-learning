import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cluster

# Reading the spambase data using pandas library.
dataread_x=pd.read_csv("spambase.data",names=range(57), dtype=np.float)
dataread_y=pd.read_csv("spambase.data",names=[57])

# converting the data to numpy array for further calculations.
X=dataread_x.values.astype(float)
Y=dataread_y.values

# KBinsDiscretizer is used with number of bins 14 by following uniform strategy  as a partof dataprepeocessing and the object is also created  
dis= KBinsDiscretizer(n_bins=14, encode='ordinal', strategy='uniform')

# creation of Hypothesis State space which take all true and False instance 
H=[]
for i in range(57):
    H.append([])

'''Least General Generalization method is used to reduce the Hypothesis state space where Algorithm 4.1 is used 
to compute the reduced hypothesis state space  after taking each instance as imput algorithm 4.1 call the 
     algorithm 4.3  for applying the original techniques of LGG and the reduced  hypothesis state space is stored in H '''
    
def LGG_SET(H,X):
   #Initilization of LGG with first instance from dataset
    for i in range(57):
        H[i].append(X[0][i])
    while(i<len(X)):
        H=LGG_conj_ID(H,X[i])
        i=i+1
    return(H)

''' Algorithm 4.3 is internal disjuction where the state space is reduced based on the techinque of generalization of two conjuctions.'''
def LGG_conj_ID(H,X):
   [H[i].append(X[i]) for i in range(len(H)) if(X[i] not in H[i])]
          
   return(H)
#Yprediction is created
Yprediction=[]

# training_data used for training the data in the hypothesis space
def training_data(X,y):
    T=[]
    [T.append(X[i]) for i in range(len(X)) if(y[i]==1)]                   
    # trained data is set back to LGG_SET 
    return(LGG_SET(H,T)) 

# testing_data method used to test the reduced hypothesis space 
def testing_data(X,y):
    for i in range(len(X)):
        c=0
        for j in range(len(X[i])):
            # checking the data values in the Hypothesis space
            if(X[i][j] in H[j]):
                c= c+1
        Yprediction.append(1) if(c==len(X[i])) else Yprediction.append(0)                     
        
        
# space used by Hypothesis
def Space_H(H):
    s=1
    for i in range(len(H)):
      if s>=1:

        s= s*len(H[i])
    print("\n space used by hypothesis:",s)

# train_test_split technique is used for data spliting where size of test data is set to 0.3
X_train1, X_test1, y_train, y_test = train_test_split(X, Y, test_size=0.3 , random_state= 15)

# X_train1 is Standardization  and stored in X_train2.
X_train2=preprocessing.scale(X_train1)

# fitting the X_train2 and transform and storing in X_train
dis.fit(X_train2)
X_train=dis.transform(X_train2)

#X_test1 is Standardization  and stored in X_test2.
X_test2=preprocessing.scale(X_test1)
# fitting the X_test2 and transform and storing in X_test
dis.fit(X_test2)
X_test=dis.transform(X_test2)

# passing the X_train and y_trainn to the training data for the hypothesis generation
H=training_data(X_train,y_train)

# testing the generated state space by passing into the testing_data 
testing_data(X_test,y_test)
print("\nGenerated HYPOTHSIS SPACE\n",H)

#spaced occupied by generated Hypothesis space
Space_H(H)

# Contingency _matrix creation for calculating f-measure
cm=cluster.contingency_matrix(y_test, Yprediction,sparse=False)
print("\n Contingency_matrix",cm)
f=((2*cm[0][0])/((2*cm[0][0])+cm[1][0]+cm[0][1]))
print("\n statistics of the  model",f)