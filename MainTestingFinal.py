
###--------------Import Libs------------------#
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np    
import random 
random.seed(1)
from models import dna_pro_model
from metrics_all import scores,precision_score,recall_score,accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import datetime
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve,confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve

###--------------END Import Libs------------------#

print("Program started",datetime.datetime.now())
EPOCHS=100
INIT_LR=1e-3

def randomForestModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    print("starting random forest regression...")  

    n_estimators = 100
    max_depth = 20
    min_samples_split = 10
    min_samples_leaf = 4
    max_features = 'sqrt'    

    ###--------------Hyperparameters------------------#
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42 
    ###--------------END Hyperparameters------------------#
    )

    rfc.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)
    y_pred_val = rfc.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    #y_pred_val = rfc.predict_proba(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1))
    y_pred_val1 = rfc.predict_proba(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1))
    test = y_pred_val.tolist()
    test1 = y_pred_val.tolist()
    y_pred_val2=([i[1]for i in y_pred_val1])
    return y_pred_val2

def neuralNetwork(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    neuralNetworkModel = MLPClassifier()
    neuralNetworkModel.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)
    y_pred_val = neuralNetworkModel.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    test = y_pred_val.tolist()
    return y_pred_val

def decisionTree(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    decisionTreeModel = DecisionTreeClassifier()
    decisionTreeModel.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)
    y_pred_val = decisionTreeModel.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    test = y_pred_val.tolist()
    return y_pred_val

def knn(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    knnModel = k=KNeighborsClassifier(n_neighbors=3)
    knnModel.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)
    y_pred_val = knnModel.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    test = y_pred_val.tolist()    
    return y_pred_val

def supportVectorMachine(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    svm = SVC(random_state=50)
    svm.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)
    y_pred_val = svm.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    test = y_pred_val.tolist()
    return y_pred_val


def logisticRegressionModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    logisticReg = LogisticRegression()
    logisticReg.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)
    y_pred_val = logisticReg.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    test = y_pred_val.tolist()    
    return y_pred_val


def CNNModel(dna_tra_fea,pro_tra_fea, y_tra3, dna_val_fea,pro_val_fea, y_val3,shape0,shape1,shape2,shape3,shape4):
    
    model=None
    model=dna_pro_model(INIT_LR,EPOCHS,shape0,shape1,shape2,shape3,shape4)
    model.summary()   
    model.fit([dna_tra_fea,pro_tra_fea], y_tra3, epochs=EPOCHS, batch_size=8)
    y_pred_val = model.predict([dna_val_fea,pro_val_fea]).flatten()
    test = y_pred_val.tolist()    
    return y_pred_val

def shapeFea(bac_tra_fea,pha_tra_fea,bac_val_fea,pha_val_fea):
    sq=int(math.sqrt(bac_tra_fea.shape[1]))
    if pow(sq,2)==bac_tra_fea.shape[1]:
        bac_tra_fea2=bac_tra_fea.reshape((-1,sq,sq))
        pha_tra_fea2=pha_tra_fea.reshape((-1,sq,sq))
        bac_val_fea2=bac_val_fea.reshape((-1,sq,sq))
        pha_val_fea2=pha_val_fea.reshape((-1,sq,sq))
    else:
        bac_tra_fea2=np.concatenate((bac_tra_fea,np.zeros((bac_tra_fea.shape[0],int(pow(sq+1,2)-bac_tra_fea.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        pha_tra_fea2=np.concatenate((pha_tra_fea,np.zeros((pha_tra_fea.shape[0],int(pow(sq+1,2)-pha_tra_fea.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        bac_val_fea2=np.concatenate((bac_val_fea,np.zeros((bac_val_fea.shape[0],int(pow(sq+1,2)-bac_val_fea.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        pha_val_fea2=np.concatenate((pha_val_fea,np.zeros((pha_val_fea.shape[0],int(pow(sq+1,2)-pha_val_fea.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
    return bac_tra_fea2, pha_tra_fea2, bac_val_fea2, pha_val_fea2

def getFeaures(data,file_path1,file_path2,strs):
    phage_features=[]
    host_features=[]
    labels=[]
    for i in data:
        phage_features.append(np.loadtxt(file_path1+i[0]+strs).tolist())
     #   print(file_path2,i,i[0],i[1],i[1].split('.')[0],i[1].split('.')[1])
        host_features.append(np.loadtxt(file_path2+i[1]+strs).tolist())
        labels.append(i[-1])
    return np.array(phage_features), np.array(host_features), np.array(labels)

def getNegatives(X_tra,X_val):  
    print("***In getNegatives",len(X_tra),len(X_val))
    X_tra_pos=[mm for mm in X_tra if mm[2]==1]
    X_neg=[str(mm[0])+','+str(mm[1]) for mm in X_tra+X_val if mm[2]==0]
    training_neg=[]
    phage=list(set([mm[0]for mm in X_tra_pos]))
    host=list(set([mm[1]for mm in X_tra_pos]))
    for p in phage:
        for h in host:
            if str(p)+','+str(h) in X_neg:
                continue
            else:
                training_neg.append([p,h,0])
    print("End of obtain neg",len(training_neg))
    return random.sample(training_neg,len(X_tra_pos))
    
 

result_all=[]
pred_all=[]
test_y_all=[]

data1=pd.read_csv('dna_pos_neg3.csv',header=None,sep=',')
data1=data1[data1[2]==1]
data1=data1.values.tolist()
#----------------------------------Training/Testing Split----------------------------------------#
train_indices, test_indices = train_test_split(range(len(data1)), test_size=0.30, random_state=42)
#----------------------------------END Training/Testing Split----------------------------------------#

train_data = [data1[int(i)][:3] for i in train_indices]
test_data = [data1[int(i)][:3] for i in test_indices]

testing_phages=list(set([i[0]for i in test_data]))
testing_hosts=list(set([i[1]for i in test_data]))

all_hosts=list(set([i[1]for i in data1]))


x_test=[]
x_test1=[]
#-------------------------------Balance data with random negative interactions-------------#
for p in testing_phages:
    testing_hosts = []
    testing_hosts=list(set([i[1]for i in test_data if i[0] == p]))
    for h in all_hosts:
        
        if h in testing_hosts:
            label=1
        else:
            label=0  
        x_test.append([p,h,label])    

x_test_df = pd.DataFrame(x_test,columns=['phage','ohost','score'])

#print('x_test_df shape',x_test_df.shape)

get_neg=getNegatives(test_data, train_data)
neg_select = getNegatives(train_data, test_data)

x_train_pos_neg = train_data+neg_select
x_train_pos_neg_df = pd.DataFrame(x_train_pos_neg,columns=['phage','host','score'])
#print('size of train_pos_neg_ after adding all hosts',x_train_pos_neg_df.shape)

#-------------------------------END Balance data with random negative interactions-------------#

#-------------------------------Feature Matrix Generation----------------------------------#

dna_tra_pha_fea,dna_tra_bac_fea,y_tra=getFeaures(train_data+neg_select,'C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/phage_dna_norm_features/', 'C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/host_dna_norm_features/','.txt')
dna_val_pha_fea,dna_val_bac_fea,y_val=getFeaures(x_test,'C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/phage_dna_norm_features/','C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/host_dna_norm_features/','.txt')
print("dna_tra_pha_fea,dna_tra_bac_fea,y_tra",dna_tra_pha_fea.shape,dna_tra_bac_fea.shape,y_tra.shape)
print("dna_val_pha_fea,dna_val_bac_fea,y_val",dna_val_pha_fea.shape,dna_val_bac_fea.shape,y_val.shape)


pro_tra_pha_fea,pro_tra_bac_fea,_=getFeaures(train_data+neg_select,'C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/phage_protein_normfeatures/','C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/host_protein_normfeatures/','.txt')
pro_val_bac_fea,pro_val_pha_fea,_=getFeaures(x_test,'C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/phage_protein_normfeatures/','C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/host_protein_normfeatures/','.txt')
print("pro_tra_pha_fea,pro_tra_bac_fea,y_tra",pro_tra_pha_fea.shape,pro_tra_bac_fea.shape,y_tra.shape)
print("pro_val_bac_fea,dna_val_bac_fea,y_val",pro_val_bac_fea.shape,pro_val_pha_fea.shape,y_val.shape)


dna_tra_pha_fea3,dna_tra_bac_fea3,dna_val_pha_fea3,dna_val_bac_fea3=shapeFea(dna_tra_pha_fea,dna_tra_bac_fea,dna_val_pha_fea,dna_val_bac_fea)
pro_tra_pha_fea3,pro_tra_bac_fea3,pro_val_bac_fea3,pro_val_pha_fea3=shapeFea(pro_tra_pha_fea,pro_tra_bac_fea,pro_val_bac_fea,pro_val_pha_fea)
X_dna=np.array([dna_tra_pha_fea3,dna_tra_bac_fea3]).transpose(1,2,3,0)
X_pro=np.array([pro_tra_pha_fea3,pro_tra_bac_fea3]).transpose(1,2,3,0)

#print("X_dna,X_pro",X_dna.shape,X_pro.shape)


alldata=[(X_dna[i,:,:,:],X_pro[i,:,:,:],y_tra[i]) for i in range(len(X_dna))]
random.shuffle(alldata)

DNA_allfeatures,Pro_allfeatures,labels=np.array([i[0] for i in alldata]),np.array([i[1] for i in alldata]),[i[2] for i in alldata]
test_y_all=test_y_all+y_val.tolist() 

#print('DNA_allfeatures shape',DNA_allfeatures.shape)

#print('Pro_allfeatures shape',Pro_allfeatures.shape)

new_labels = np.asarray(labels,dtype=np.int32)
#-------------------------------END Feature Matrix Generation----------------------------------#

#-------------------------------Machine Learning Model training and prediction-------------#

#Neural Network
neuralNetwork_pred = neuralNetwork(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()

#Logistic Regression
logistic_pred=logisticRegressionModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
x_test_df['logistic_pred_score'] = logistic_pred.tolist()

#SVM
svm_pred=supportVectorMachine(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
x_test_df['svm_pred_score'] = svm_pred.tolist()

#KNN
knn_pred=knn(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
x_test_df['knn_pred_score'] = knn_pred.tolist()

#Decision Tree
decision_pred=decisionTree(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
x_test_df['decision_pred_score'] = decision_pred.tolist()

#Random forest
randomForest_pred=randomForestModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
x_test_df['randomForest_pred_score'] = randomForest_pred

#CNN
cnn_pred=CNNModel(DNA_allfeatures, Pro_allfeatures,new_labels, np.array([dna_val_pha_fea3,dna_val_bac_fea3]).transpose(1,2,3,0),
                                            np.array([pro_val_bac_fea3,pro_val_pha_fea3]).transpose(1,2,3,0),y_val,
                                            DNA_allfeatures.shape[1],DNA_allfeatures.shape[2],Pro_allfeatures.shape[1],
                                            Pro_allfeatures.shape[2],2)

x_test_df['pred_score'] = cnn_pred.tolist()

#-------------------------------END Machine Learning Model training and prediction-------------#


#-------------------------------Collect Alignment Scores------------------------------------#

#-----------------------BLASTPhage-Host-----------------------------------#
#print('xtestdf before ph merge',x_test_df.shape)
ph_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/phScoreFile.csv')
x_test_df = pd.merge(x_test_df,ph_data,left_on=['phage','ohost'], right_on=['phage','phhost'])
#print('xtestdf after ph merge',x_test_df.shape)

#-----------------------END BLASTPhage-Host-----------------------------------#


#-----------------------BLASTHost-Host-----------------------------------#
#pp_data = pd.read_excel('C:/Users/pavana/AppData/Local/anaconda3/envs/my_python/final_merged_phage_phage_file.xlsx')
pp_data = pd.read_excel('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/final_file_dec9.xlsx')
x_test_df = pd.merge(x_test_df,pp_data,on=['phage'])
#print('xtestdf after pp merge',x_test_df.shape)


data2 = pd.read_excel('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/dna_pos_neg3_p.xlsx')

x_test_df = pd.merge(x_test_df,data2,on=['mphage'])
#print('xtestdf after pos neg merge',x_test_df.shape)

hh_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/hhScoreFile.csv')
x_test_df = pd.merge(x_test_df,hh_data,on=['mhost','ohost'])
#print('xtestdf after hh merge',x_test_df.shape)
#print('x_test_df.shape',x_test_df.shape)
#print('xtestdf after hh',x_test_df )

#-----------------------END BLASTHost-Host-----------------------------------#

#-----------------------AlignmentScore normalization---------------------#
x_test_df.bit_score = (x_test_df.bit_score - x_test_df.bit_score.min())/(x_test_df.bit_score.max() - x_test_df.bit_score.min())
x_test_df.hh_score = (x_test_df.hh_score - x_test_df.hh_score.min())/(x_test_df.hh_score.max() - x_test_df.hh_score.min())
#-----------------------END AlignmentScore normalization---------------------#


#----------------------------------------------------------Composite Model assembly---------------------------------------------------------#

x_test_df['MaxRecord'] = (x_test_df.groupby('phage')['bit_score'].transform('max') == x_test_df['bit_score']).astype(int)   #highest bitscore for Phage-host
#highest bitscore for phage-host and phahe-phage/host-host
x_test_df['MaxRecord1'] = ((x_test_df.groupby('phage')['bit_score'].transform('max') == x_test_df['bit_score']) & (x_test_df.groupby('phage')['hh_score'].transform('max') == x_test_df['hh_score'])).astype(int) 

alpha = 0.9
gamma = 0.6


#CNN

def formula_cnn(x):
    if x['MaxRecord'] == 1:
      return  (x['pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['pred_score'])

x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)


def formula_cnn1(x):
    if x['MaxRecord1'] == 1:
      return  (x['pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['pred_score'])

x_test_df['result_new_cnn1'] = x_test_df.apply(formula_cnn1, axis=1)

#NN

def formula_nn(x):
    if x['MaxRecord'] == 1:
      return  (x['neuralNetwork_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['neuralNetwork_pred_score'])

x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)

def formula_nn1(x):
    if x['MaxRecord1'] == 1:
      return  (x['neuralNetwork_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['neuralNetwork_pred_score'])

x_test_df['result_new_nn1'] = x_test_df.apply(formula_nn1, axis=1)


#logistic

def formula_lr(x):
    if x['MaxRecord'] == 1:
      return  (x['logistic_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['logistic_pred_score'])

x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)


def formula_lr1(x):
    if x['MaxRecord1'] == 1:
      return  (x['logistic_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['logistic_pred_score'])

x_test_df['result_new_lr1'] = x_test_df.apply(formula_lr1, axis=1)

#KNN

def formula_knn(x):
    if x['MaxRecord'] == 1:
      return  (x['knn_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['knn_pred_score'])

x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)

def formula_knn1(x):
    if x['MaxRecord1'] == 1:
      return  (x['knn_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['knn_pred_score'])

x_test_df['result_new_knn1'] = x_test_df.apply(formula_knn1, axis=1)

#SVM
def formula_svm(x):
    if x['MaxRecord'] == 1:
      return  (x['svm_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['svm_pred_score'])

x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)

def formula_svm1(x):
    if x['MaxRecord1'] == 1:
      return  (x['svm_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['svm_pred_score'])

x_test_df['result_new_svm1'] = x_test_df.apply(formula_svm1, axis=1)


#decision

def formula_dt(x):
    if x['MaxRecord'] == 1:
      return  (x['decision_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['decision_pred_score'])

x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)

def formula_dt1(x):
    if x['MaxRecord1'] == 1:
      return  (x['decision_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['decision_pred_score'])

x_test_df['result_new_dt1'] = x_test_df.apply(formula_dt1, axis=1)


#Random Forest
x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf(x):
    if x['MaxRecord'] == 1:
      return  (x['randomForest_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['randomForest_pred_score'])

x_test_df['result_new_rf'] = x_test_df.apply(formula_rf, axis=1)


x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf1(x):
    if x['MaxRecord1'] == 1:
      return  (x['randomForest_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
    else:
      return  (x['randomForest_pred_score'])

x_test_df['result_new_rf1'] = x_test_df.apply(formula_rf1, axis=1)



#print('xtestdf after formula')
#print(tabulate(x_test_df.head(),headers='keys'))
x_test_df.to_csv("TestResult_bitscores.csv")  # Bitscore addition Validation file




#--------------------------------------------Taxonomy Tree addition------------------------------------#


#print('xtestdf before hosttaxonomyNew',x_test_df.shape)
tree_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/hosttaxonomyNew.csv')
x_test_df = pd.merge(x_test_df,tree_data,left_on=['ohost'], right_on=['hostNC'])
#print('xtestdf after hosttaxonomyNew',x_test_df.shape)

phy_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/phyFile.csv')
x_test_df = pd.merge(x_test_df,phy_data,left_on=['phage','hostphylum'], right_on=['phage','phy'])
#print('xtestdf after phy_dat',x_test_df.shape)
#print('xtestdf after phy_dat',x_test_df)
#x_test_df.to_csv("PHY_DATA.csv")


cla_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/claFile.csv')
x_test_df = pd.merge(x_test_df,cla_data,left_on=['phage','hostclass'], right_on=['phage','cla'])
#print('xtestdf after cla_dat',x_test_df.shape)
#print('xtestdf after phy_dat',x_test_df)


ord_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/ordFile.csv')
x_test_df = pd.merge(x_test_df,ord_data,left_on=['phage','hostorder'], right_on=['phage','ord'])
print('xtestdf after ord_dat',x_test_df.shape)
#print('xtestdf after phy_dat',x_test_df)


fly_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/flyFile.csv')
x_test_df = pd.merge(x_test_df,fly_data,left_on=['phage','hostfamily'], right_on=['phage','fly'])
#print('xtestdf after fly_dat',x_test_df.shape)
#print('xtestdf after phy_dat',x_test_df)

gns_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/gnsFile.csv')
x_test_df = pd.merge(x_test_df,gns_data,left_on=['phage','hostgenus'], right_on=['phage','gns'])
#print('xtestdf after gns_dat',x_test_df.shape)
#print('xtestdf after phy_dat',x_test_df)

phyyval=[]
phyyval = x_test_df['scorephy']
clayval=[]
clayval = x_test_df['scorecla']
ordyval=[]
ordyval = x_test_df['scoreord']
flyyval=[]
flyyval = x_test_df['scorefly']
gnsyval=[]
gnsyval = x_test_df['scoregns']


data2 = pd.read_excel('C:/Users/shrey/anaconda3/envs/newCoMPHI/CoMPHI/dna_pos_neg3.xlsx')
x_test_df = pd.merge(x_test_df,data2,left_on=['phage'], right_on=['mphage'])
#print('xtestdf after pos neg merge',x_test_df.shape)


#Random Forest
x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf_phy(x):
    if x['Phylumpn'] == x['hostphylum']:
      return  1
    else:
      return  (0. if x['randomForest_pred_score'] < 0.5 else 1.)
      

x_test_df['rf_phy_Score'] = x_test_df.apply(formula_rf_phy, axis=1)


x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf_cla(x):
    if x['Classpn'] == x['hostclass']:
      return  1
    else:
      return  (0. if x['randomForest_pred_score'] < 0.5 else 1.)
 

x_test_df['rf_cla_Score'] = x_test_df.apply(formula_rf_cla, axis=1)

x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf_ord(x):
    if x['Orderpn'] == x['hostorder']:
      return  1
    else:
      return  (0. if x['randomForest_pred_score'] < 0.5 else 1.)
    

x_test_df['rf_ord_Score'] = x_test_df.apply(formula_rf_ord, axis=1)

x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf_fly(x):
    if x['Familypn'] == x['hostfamily']:
      return  1
    else:
      return  (0. if x['randomForest_pred_score'] < 0.5 else 1.)


x_test_df['rf_fly_Score'] = x_test_df.apply(formula_rf_fly, axis=1)

x_test_df['randomForest_pred_score'] = randomForest_pred
def formula_rf_gns(x):
    if x['Genuspn'] == x['hostgenus']:
      return  1
    else:
      return  (0. if x['randomForest_pred_score'] < 0.5 else 1.)
   

x_test_df['rf_gns_Score'] = x_test_df.apply(formula_rf_gns, axis=1)


print('xtestdf after tree formula',x_test_df.shape)
print('xtestdf after tree formula',x_test_df)

#--------------------------------------------END Taxonomy Tree addition------------------------------------#
#----------------------------------------------------------END Composite Model assembly---------------------------------------------------------#


#----------------------------------------------------------Composite model Metrics---------------------------------------------------------------#

#CNN
#CNN only
model = 'cnn'
x_test_df['pred_score'] = cnn_pred.tolist()
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['pred_score'],model)
print('cnn scores', final_scores )

#CNN with both alignment scores
model = 'cnnMax'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)
result_new_scores_cnn = scores(list(map(int, y_val.tolist())),x_test_df['result_new_cnn'],model)
print('New Maxrecord result cnn', result_new_scores_cnn )

#CNN with only BlastPhage-Host alignment scores
model = 'cnnMax1'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn1'] = x_test_df.apply(formula_cnn1, axis=1)
result_new_scores_cnn1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_cnn1'],model)
print('New Maxrecord result cnn 1', result_new_scores_cnn1 )


#Taxonomy metrics for CNN
model = 'cnnPhy'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['result_new_cnn'],model)
print('cnn scores phy', final_scoresphy )


model = 'cnnCla'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['result_new_cnn'],model)
print('cnn scores cla', final_scorescla )


model = 'cnnOrd'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['result_new_cnn'],model)
print('cnn scores ord', final_scoresord )



model = 'cnnFly'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['result_new_cnn'],model)
print('cnn scores fly', final_scoresfly )


model = 'cnnGns'
x_test_df['pred_score'] = cnn_pred.tolist()
x_test_df['result_new_cnn'] = x_test_df.apply(formula_cnn, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['result_new_cnn'],model)
print('cnn scores gns', final_scoresgns )




#NN
#NN only
model = 'nn'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['neuralNetwork_pred_score'],model)
print('nn scores', final_scores )


#NN with both alignment scores
model = 'nnMax'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)
result_new_scores_nn = scores(list(map(int, y_val.tolist())),x_test_df['result_new_nn'],model)
print('New Maxrecord result nn', result_new_scores_nn )

#NN with only BlastPhage-Host alignment scores
model = 'nnMax1'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn1'] = x_test_df.apply(formula_nn1, axis=1)
result_new_scores_nn1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_nn1'],model)
print('New Maxrecord result nn 1', result_new_scores_nn1 )

#Taxonomy metrics for NN
model = 'nnPhy'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['result_new_nn'],model)
print('nn scores phy', final_scoresphy )


model = 'nnCla'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['result_new_nn'],model)
print('nn scores cla', final_scorescla )


model = 'nnOrd'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['result_new_nn'],model)
print('nn scores ord', final_scoresord )


model = 'nnFly'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['result_new_nn'],model)
print('nn scores fly', final_scoresfly )


model = 'nngns'
x_test_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
x_test_df['result_new_nn'] = x_test_df.apply(formula_nn, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['result_new_nn'],model)
print('nn scores gns', final_scoresgns )



#logistics
#Logistic Regression only
model = 'lr'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['logistic_pred_score'],model)
print('lr scores', final_scores )

#Logistic Regression with both alignment scores
model = 'lrMax'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)
result_new_scores_lr = scores(list(map(int, y_val.tolist())),x_test_df['result_new_lr'],model)
print('New Maxrecord result lr', result_new_scores_lr )

#Logistic Regression with only BlastPhage-Host alignment scores
model = 'lrMax1'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr1'] = x_test_df.apply(formula_lr1, axis=1)
result_new_scores_lr1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_lr1'],model)
print('New Maxrecord result lr 1', result_new_scores_lr1 )

#Taxonomy metrics for Logistic Regression
model = 'lrPhy'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['result_new_lr'],model)
print('lr scores phy', final_scoresphy )


model = 'lrCla'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['result_new_lr'],model)
print('lr scores cla', final_scorescla )


model = 'lrOrd'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['result_new_lr'],model)
print('lr scores ord', final_scoresord )


model = 'lrFly'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['result_new_lr'],model)
print('lr scores fly', final_scoresfly )


model = 'lrGns'
x_test_df['logistic_pred_score'] = logistic_pred.tolist()
x_test_df['result_new_lr'] = x_test_df.apply(formula_lr, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['result_new_lr'],model)
print('lr scores gns', final_scoresgns )



#SVM only
model = 'svm'
x_test_df['svm_pred_score'] = svm_pred.tolist()
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['svm_pred_score'],model)
print('svm scores', final_scores )

#SVM with both alignment scores
model = 'svmMax'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)
result_new_scores_svm = scores(list(map(int, y_val.tolist())),x_test_df['result_new_svm'],model)
print('New Maxrecord result svm', result_new_scores_svm )

#SVM with only BlastPhage-Host alignment scores
model = 'svmMax1'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm1'] = x_test_df.apply(formula_svm1, axis=1)
result_new_scores_svm1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_svm1'],model)
print('New Maxrecord result svm 1', result_new_scores_svm1 )

#Taxonomy metrics for SVM
model = 'svmPhy'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['result_new_svm'],model)
print('svm scores phy', final_scoresphy )

model = 'svmCla'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['result_new_svm'],model)
print('svm scores cla', final_scorescla )


model = 'svmOrd'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['result_new_svm'],model)
print('svm scores ord', final_scoresord )

model = 'svmFly'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['result_new_svm'],model)
print('svm scores fly', final_scoresfly )


model = 'svmGns'
x_test_df['svm_pred_score'] = svm_pred.tolist()
x_test_df['result_new_svm'] = x_test_df.apply(formula_svm, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['result_new_svm'],model)
print('svm scores gns', final_scoresgns )



#Decision Tree only
model = 'dt'
x_test_df['decision_pred_score'] = decision_pred.tolist()
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['decision_pred_score'],model)
print('dt scores', final_scores )

#Decision Tree with both alignment scores
model = 'dtMax'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)
result_new_scores_dt = scores(list(map(int, y_val.tolist())),x_test_df['result_new_dt'],model)
print('New Maxrecord result dt', result_new_scores_dt )


#Decision Tree with only BlastPhage-Host alignment scores
model = 'dtMax1'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt1'] = x_test_df.apply(formula_dt1, axis=1)
result_new_scores_dt1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_dt1'],model)
print('New Maxrecord result dt 1', result_new_scores_dt1 )

#Taxonomy metrics for Decision Tree
model = 'dtPhy'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['result_new_dt'],model)
print('dt scores phy', final_scoresphy )


model = 'dtCla'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['result_new_dt'],model)
print('dt scores cla', final_scorescla )


model = 'dtOrd'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['result_new_dt'],model)
print('dt scores ord', final_scoresord )


model = 'dtFly'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['result_new_dt'],model)
print('dt scores fly', final_scoresfly )


model = 'dtGns'
x_test_df['decision_pred_score'] = decision_pred.tolist()
x_test_df['result_new_dt'] = x_test_df.apply(formula_dt, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['result_new_dt'],model)
print('dt scores gns', final_scoresgns )




#KNN only
model = 'knn'
x_test_df['knn_pred_score'] = knn_pred.tolist()
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['knn_pred_score'],model)
print('knn scores', final_scores )

#KNN with both alignment scores
model = 'knnMax'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)
result_new_scores_knn = scores(list(map(int, y_val.tolist())),x_test_df['result_new_knn'],model)
print('New Maxrecord result KNN', result_new_scores_knn )

#KNN with only BlastPhage-Host alignment scores
model = 'knnMax1'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn1'] = x_test_df.apply(formula_knn1, axis=1)
result_new_scores_knn1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_knn1'],model)
print('New Maxrecord result KNN 1', result_new_scores_knn1 )


#Taxonomy metrics for KNN
model = 'knnPhy'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['result_new_knn'],model)
print('knn scores phy', final_scoresphy )



model = 'knnCla'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['result_new_knn'],model)
print('knn scores cla', final_scorescla )

precision,recall,thresholds1 =precision_recall_curve(list(map(int, clayval.tolist())),x_test_df['result_new_knn'])


model = 'knnOrd'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['result_new_knn'],model)
print('knn scores ord', final_scoresord )


model = 'knnFly'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['result_new_knn'],model)
print('knn scores fly', final_scoresfly )


model = 'knnGns'
x_test_df['knn_pred_score'] = knn_pred.tolist()
x_test_df['result_new_knn'] = x_test_df.apply(formula_knn, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['result_new_knn'],model)
print('knn scores gns', final_scoresgns )




#Random Forest only
model = 'rf'
x_test_df['randomForest_pred_score'] = randomForest_pred
final_scores = scores(list(map(int, y_val.tolist())),x_test_df['randomForest_pred_score'],model)
print('random forest scores', final_scores )

#Random Forest with both alignment scores
model = 'rfMax'
x_test_df['randomForest_pred_score'] = randomForest_pred
x_test_df['result_new_rf'] = x_test_df.apply(formula_rf, axis=1)
x_test_df['randomForest_pred_score'] = randomForest_pred
result_new_scores_rf = scores(list(map(int, y_val.tolist())),x_test_df['result_new_rf'],model)
print('New Maxrecord result RF', result_new_scores_rf )


#Random Forest with only BlastPhage-Host alignment scores
model = 'rfMax1'
x_test_df['randomForest_pred_score'] = randomForest_pred
x_test_df['result_new_rf1'] = x_test_df.apply(formula_rf1, axis=1)
x_test_df['randomForest_pred_score'] = randomForest_pred
result_new_scores_rf1 = scores(list(map(int, y_val.tolist())),x_test_df['result_new_rf1'],model)
print('New Maxrecord result RF 1', result_new_scores_rf1 )


#Taxonomy metrics for Random Forest
model = 'rfPhy'
x_test_df['randomForest_pred_score'] = randomForest_pred
phyyval=[]
phyyval = x_test_df['scorephy']

final_scoresphy = scores(list(map(int, phyyval.tolist())),x_test_df['rf_phy_Score'],model)
print('random scores phy', final_scoresphy )


model = 'rfCla'
x_test_df['randomForest_pred_score'] = randomForest_pred
x_test_df['result_new_rf'] = x_test_df.apply(formula_rf, axis=1)
clayval=[]
clayval = x_test_df['scorecla']

final_scorescla = scores(list(map(int, clayval.tolist())),x_test_df['rf_cla_Score'],model)
print('random scores cla', final_scorescla )


model = 'rfOrd'
x_test_df['randomForest_pred_score'] = randomForest_pred
x_test_df['result_new_rf'] = x_test_df.apply(formula_rf, axis=1)
ordyval=[]
ordyval = x_test_df['scoreord']

final_scoresord = scores(list(map(int, ordyval.tolist())),x_test_df['rf_ord_Score'],model)
print('random scores ord', final_scoresord )


model = 'rfFly'
x_test_df['randomForest_pred_score'] = randomForest_pred
x_test_df['result_new_rf'] = x_test_df.apply(formula_rf, axis=1)
flyyval=[]
flyyval = x_test_df['scorefly']

final_scoresfly = scores(list(map(int, flyyval.tolist())),x_test_df['rf_fly_Score'],model)
print('random scores fly', final_scoresfly )


model = 'rfGns'
x_test_df['randomForest_pred_score'] = randomForest_pred
x_test_df['result_new_rf'] = x_test_df.apply(formula_rf, axis=1)
gnsyval=[]
gnsyval = x_test_df['scoregns']

final_scoresgns = scores(list(map(int, gnsyval.tolist())),x_test_df['rf_gns_Score'],model)
print('random scores gns', final_scoresgns )

#----------------------------------------------------------END Composite model Metrics---------------------------------------------------------------#



