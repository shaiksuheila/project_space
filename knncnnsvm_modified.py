 "https://drive.google.com/drive/folders/13XNkglB3VovC4iUls9SP3xOn3gnJtzcR?usp=share_link   :dataset link"
                        
import pandas as pd
a=pd.read_csv(r'C:\Users\Ibrahim Shaik\Downloads\onlinefraud.csv\onlinefraud.csv')
        
            # SUMMARIZATION---->DATA VISUALIZATION
    
a.shape       #total rows and columns
a.index  
a.columns     # COLUMN NAMES
a.dtypes      # TYPE OF DATA PRESENT IN EACH COLUMN
a.describe()  #works only when column consists of numbers
a.isna().sum()

                    #DATA PRE_PROCESSING
                    
a.drop(['nameOrig'],axis=1,inplace=True)
a.drop(['nameDest'],axis=1,inplace=True)
a.drop(['isFlaggedFraud'],axis=1,inplace=True)
a.dtypes
a['type'].value_counts()
a["type"] = a["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
a['type'].value_counts()
a['isFraud'].value_counts()

## balancing dataset

non_fraud=a[a['isFraud']==0]
fraud=a[a['isFraud']==1]
non_fraud.shape, fraud.shape

##extracting nonfraud dataset of size= fraud dataset size
 
non_fraud=non_fraud.sample(fraud.shape[0])
non_fraud.shape

##appending nonfraud dataset to fraud dataset
for i in range(3):
    fraud=fraud.append(non_fraud, ignore_index=True)

print(fraud.shape)
c=fraud
c.shape


x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)  #test_size denotes 20% of data splitted for testing and remaining for training

## reshaping the data for convolution

xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtest=xtest.reshape(xtest.shape[0],xtest.shape[1],1)
xtrain.shape,xtest.shape

## CNN Algorithm


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense


model = Sequential()
model.add(Dense(xtrain.shape[1],activation='relu',input_dim= xtrain.shape[1]))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,batch_size=5,epochs=50)
ypred = model.predict(xtest)
if (model.predict([[1,2,2346.00,56378888888888888.00,4035679.00,46000.00,324.00]])):
    print("Fraud")
else:
    print("NOT a FRAUD")


##CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,BatchNormalization,Dropout
# import model
model=Sequential()
# layers
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=xtrain[1].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# build ANN
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#time
# fitting the model
history=model.fit(xtrain,ytrain,epochs=20,validation_data=(xtest,ytest))
ypred=model.predict(xtest)
print(model.predict([[1,2,2346.00,56378888888888888.00,4035679.00,46000.00,324.00]]))


##importing kNN algorithm named KNeighborsClassifier from neighbors module
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)

a["isFraud"] = a["isFraud"].map({0: "No Fraud", 1: "Fraud"})


###SVM algorithm
from sklearn.svm import SVC
smodel=SVC(kernel='linear')

model.fit(xtrain,ytrain)
ypred=model.predict(xtest) #prediction values

#comparing prediction values given by algorithm and testing values and checking for accuracy

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100) 
a.head()
a.columns
a
## giving input and checking output
print(model.predict([[1,2,2346.00,56378888888888888.00,4035679.00,46000.00,324.00]]))
