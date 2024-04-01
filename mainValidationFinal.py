import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np   
import tensorflow as tf 
import random 
random.seed(1)
from models import dna_pro_model
from metrics import scores,precision_score,recall_score,accuracy_score
from sklearn.model_selection import KFold
import math
import datetime
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve,confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve

print("Program started",datetime.datetime.now())
EPOCHS=1
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
    ###--------------Hyperparameters------------------#
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

#Read Similarity data  next_step_df
#print('next_step_df before ph merge',next_step_df.shape)
ph_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/CoMPHI/phScoreFile.csv')

ph_data.bit_score = (ph_data.bit_score - ph_data.bit_score.min())/(ph_data.bit_score.max() - ph_data.bit_score.min())
print('xtestdf after normalization',ph_data )
ph_data['PH_MaxRecord'] = (ph_data.groupby('phage')['bit_score'].transform('max') == ph_data['bit_score']).astype(int)
print('xtestdf after normalization',ph_data )



hh_data = pd.read_csv('C:/Users/shrey/anaconda3/envs/CoMPHI/hhScoreFile.csv')
hh_data.bit_score = (hh_data.hh_score - hh_data.hh_score.min())/(hh_data.hh_score.max() - hh_data.hh_score.min())
print('hh_data after normalization',hh_data )
hh_data['HH_MaxRecord'] = (hh_data.groupby('mhost')['hh_score'].transform('max') == hh_data['hh_score']).astype(int)
print('hh_data after normalization',hh_data )



data1=pd.read_csv('dna_pos_neg3_p.csv',header=None,sep=',')
print(data1.shape)
data1=data1[data1[2]==1]
print(data1.shape)
allinter=[str(data1.loc[i,0])+','+str(data1.loc[i,1]) for i in data1.index]

#*******************FOLD*********************

kf = KFold(n_splits=5)

#*******************FOLD*********************
training=pd.read_csv('dna_pos_neg3_p.csv',header=None,sep=',')
mytraining = training.values.tolist()

#neg_select = getNegatives(training)
#training.append(pd.DataFrame())
fold=0


for train_index, test_index in kf.split(training): 
    print("In fold",fold)
    fold+=1
    ###obtain data
    X_tra=[mytraining[ii] for ii in train_index]
    X_val=[mytraining[ii] for ii in test_index]
    print("X_tra",type(X_tra),len(X_tra))
    print("X_val",type(X_val),len(X_val),X_val[0],X_val[1])
    neg_select_tra = getNegatives(X_tra, X_val)
    neg_select_val = getNegatives(X_val,X_tra)
    print("neg_select1",type(neg_select_val),len(neg_select_val),neg_select_val[0],neg_select_val[1])
    next_step = X_val+neg_select_val
    print("next_step",type(next_step),len(next_step),next_step[0],next_step[1])
    next_step_df = pd.DataFrame(next_step,columns=['phage','ohost','score'])
    '''print(next_step_df.shape)
    print(next_step_df.head())
    print(next_step_df.tail())

    
    print(type(X_tra),type(X_val),type(neg_select_tra),type(X_tra+neg_select_tra))  
    print(len(X_tra),len(X_val),len(neg_select_tra),len(X_tra+neg_select_tra))  '''
    
    dna_tra_pha_fea,dna_tra_bac_fea,y_tra=getFeaures(X_tra+neg_select_tra,'C:/Users/shrey/anaconda3/envs/CoMPHI/phage_dna_norm_features/','C:/Users/shrey/anaconda3/envs/CoMPHI/host_dna_norm_features/','.txt')
    dna_val_pha_fea,dna_val_bac_fea,y_val=getFeaures(X_val+neg_select_val,'C:/Users/shrey/anaconda3/envs/CoMPHI/phage_dna_norm_features/','C:/Users/shrey/anaconda3/envs/CoMPHI/host_dna_norm_features/','.txt')
    '''print("y_train",type(y_tra))
    print("y_test",type(y_val),y_val.size)'''

    pro_tra_pha_fea,pro_tra_bac_fea,_=getFeaures(X_tra+neg_select_tra,'C:/Users/shrey/anaconda3/envs/CoMPHI/phage_protein_normfeatures/','C:/Users/shrey/anaconda3/envs/CoMPHI/host_protein_normfeatures/','.txt')
    pro_val_bac_fea,pro_val_pha_fea,_=getFeaures(X_val+neg_select_val,'C:/Users/shrey/anaconda3/envs/CoMPHI/phage_protein_normfeatures/','C:/Users/shrey/anaconda3/envs/CoMPHI/host_protein_normfeatures/','.txt')
    dna_tra_pha_fea3,dna_tra_bac_fea3,dna_val_pha_fea3,dna_val_bac_fea3=shapeFea(dna_tra_pha_fea,dna_tra_bac_fea,dna_val_pha_fea,dna_val_bac_fea)
    pro_tra_pha_fea3,pro_tra_bac_fea3,pro_val_bac_fea3,pro_val_pha_fea3=shapeFea(pro_tra_pha_fea,pro_tra_bac_fea,pro_val_bac_fea,pro_val_pha_fea)
    X_dna=np.array([dna_tra_pha_fea3,dna_tra_bac_fea3]).transpose(1,2,3,0)
    X_pro=np.array([pro_tra_pha_fea3,pro_tra_bac_fea3]).transpose(1,2,3,0)
    alldata=[(X_dna[i,:,:,:],X_pro[i,:,:,:],y_tra[i]) for i in range(len(X_dna))]
    random.shuffle(alldata)
    DNA_allfeatures,Pro_allfeatures,labels=np.array([i[0] for i in alldata]),np.array([i[1] for i in alldata]),[i[2] for i in alldata]
    test_y_all=test_y_all+y_val.tolist() 
    #print(labels)

    #    print("*********************")
    #    print("alldata_aug",type(alldata),len(alldata),len(alldata[0]))
    #    print("*********************")
    #    print("DNA_allfeatures_aug",type(DNA_allfeatures_aug),DNA_allfeatures_aug.size)
    #    print("*********************")
    #    print("Pro_allfeatures_aug",type(Pro_allfeatures_aug),Pro_allfeatures_aug.size)
    #    print("*********************")
    new_labels = np.asarray(labels,dtype=np.int32)
    #    print("labels_aug",type(labels_aug),len(labels_aug),labels_aug[0])
    #    print("Pro_allfeatures_aug",type(new_labels_aug),new_labels_aug.size)
    #print(new_labels)
    #print(len(new_labels))

    print("y_true",y_val.tolist()[0],y_val.tolist()[1],y_val.tolist()[2],y_val.tolist()[3],len(y_val.tolist()),sum(y_val.tolist()))
        ###prediction model
    CNN_pred=CNNModel(DNA_allfeatures, Pro_allfeatures,new_labels, np.array([dna_val_pha_fea3,dna_val_bac_fea3]).transpose(1,2,3,0),
                                            np.array([pro_val_bac_fea3,pro_val_pha_fea3]).transpose(1,2,3,0),y_val,
                                            DNA_allfeatures.shape[1],DNA_allfeatures.shape[2],Pro_allfeatures.shape[1],
                                            Pro_allfeatures.shape[2],2)
    #next_step_df = next_step_df.append(pd.DataFrame({'pred_score':CNN_pred}))
    next_step_df['pred_score'] = CNN_pred.tolist()

    print('next_step_df head after cnn',next_step_df.head())
    print('next_step_df tail after cnn', next_step_df.tail())

    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['pred_score'])
    print('cnn scores', final_scores )
    
    

    randomForest_pred=randomForestModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['randomForest_pred_score'] = randomForest_pred
    #print('next_step_df after pred shape',next_step_df.shape)
    print('next_step_df head after random',next_step_df.head())
    print('next_step_df tail after random', next_step_df.tail())

    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['randomForest_pred_score'])
    print('random forest scores', final_scores )

    precision,recall,thresholds1 =precision_recall_curve(list(map(int, y_val.tolist())),next_step_df['randomForest_pred_score'])
    df=pd.DataFrame(precision)
    df.to_csv('randomForest_precision.csv', index=False)
    df1=pd.DataFrame(recall)
    df.to_csv('randomForest_recall.csv', index=False)

    neuralNetwork_pred = neuralNetwork(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
    print('next_step_df after neural network pred shape',next_step_df.shape)

    logistic_pred=logisticRegressionModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['logistic_pred_score'] = logistic_pred.tolist()
    print('next_step_df after logistic pred shape',next_step_df.shape)
    #print('xtest_df head',next_step_df.head())
    #print('next_step_df tail', next_step_df.tail())

    svm_pred=supportVectorMachine(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['svm_pred_score'] = svm_pred.tolist()
    print('next_step_df after svm pred shape',next_step_df.shape)

    knn_pred=knn(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['knn_pred_score'] = knn_pred.tolist()
    print('next_step_df after knn pred shape',next_step_df.shape)

    decision_pred=decisionTree(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['decision_pred_score'] = decision_pred.tolist()
    print('next_step_df after decision pred shape',next_step_df.shape)


    #Read Similarity data  next_step_df
    print('next_step_df before ph merge',next_step_df.shape)
    next_step_df = pd.merge(next_step_df,ph_data,left_on=['phage','ohost'], right_on=['phage','phhost'])
    print('next_step_df after ph merge',next_step_df.shape)

  

    #pp_data = pd.read_excel('C:/Users/shrey/anaconda3/envs/CoMPHI/final_merged_phage_phage_file.xlsx')
    pp_data = pd.read_excel('C:/Users/shrey/anaconda3/envs/CoMPHI/final_file_dec9.xlsx')

    next_step_df = pd.merge(next_step_df,pp_data,on=['phage'])
    print('next_step_df after pp merge',next_step_df.shape)

    data2 = pd.read_excel('C:/Users/shrey/anaconda3/envs/CoMPHI/dna_pos_neg3_p.xlsx')
    print('pos neg xlsx',data2 )

    next_step_df = pd.merge(next_step_df,data2,on=['mphage'])
    print('next_step_df after pos neg merge',next_step_df.shape)

    next_step_df = pd.merge(next_step_df,hh_data,on=['mhost','ohost'])
    print('next_step_df after hh merge',next_step_df.shape)


    next_step_df['Both_MaxRecord'] = ((next_step_df['PH_MaxRecord'] == 1) & (next_step_df['HH_MaxRecord'] == 1)).astype(int)
    print('next_step_df after maxrecord both',next_step_df )



    alpha = 0.9
    gamma = 0.3

    #CNN

    def formula_cnn(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['pred_score'])

    next_step_df['result_new_cnn'] = next_step_df.apply(formula_cnn, axis=1)


    def formula_cnn1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['pred_score'])

    next_step_df['result_new_cnn1'] = next_step_df.apply(formula_cnn1, axis=1)

    #Random Forest

    def formula_rf(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['randomForest_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['randomForest_pred_score'])

    next_step_df['result_new_rf'] = next_step_df.apply(formula_rf, axis=1)

    def formula_rf1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['randomForest_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['randomForest_pred_score'])

    next_step_df['result_new_rf1'] = next_step_df.apply(formula_rf1, axis=1)



    #NN

    def formula_nn(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['neuralNetwork_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['neuralNetwork_pred_score'])

    next_step_df['result_new_nn'] = next_step_df.apply(formula_nn, axis=1)

    def formula_nn1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['neuralNetwork_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['neuralNetwork_pred_score'])

    next_step_df['result_new_nn1'] = next_step_df.apply(formula_nn1, axis=1)


    #LR

    def formula_lr(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['logistic_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['logistic_pred_score'])

    next_step_df['result_new_lr'] = next_step_df.apply(formula_lr, axis=1)

    def formula_lr1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['logistic_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['logistic_pred_score'])

    next_step_df['result_new_lr1'] = next_step_df.apply(formula_lr1, axis=1)



    #SVM

    def formula_svm(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['svm_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['svm_pred_score'])

    next_step_df['result_new_svm'] = next_step_df.apply(formula_svm, axis=1)

    def formula_svm1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['svm_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['svm_pred_score'])

    next_step_df['result_new_svm1'] = next_step_df.apply(formula_svm1, axis=1)



    #KNN

    def formula_knn(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['knn_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['knn_pred_score'])

    next_step_df['result_new_knn'] = next_step_df.apply(formula_knn, axis=1)

    def formula_knn1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['knn_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['knn_pred_score'])

    next_step_df['result_new_knn1'] = next_step_df.apply(formula_knn1, axis=1)


    

    #DT

    def formula_dt(x):
        if x['PH_MaxRecord'] == 1:
          return  (x['decision_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['decision_pred_score'])

    next_step_df['result_new_dt'] = next_step_df.apply(formula_dt, axis=1)

    def formula_dt1(x):
        if x['Both_MaxRecord'] == 1:
          return  (x['decision_pred_score'] * (1 - gamma) + (x['bit_score'] * (1 - alpha) + x ['hh_score'] * alpha) * gamma )
        else:
          return  (x['decision_pred_score'])

    next_step_df['result_new_dt1'] = next_step_df.apply(formula_dt1, axis=1)





    #CNN
    next_step_df['pred_score'] = CNN_pred.tolist()
    next_step_df['result_new_cnn'] = next_step_df.apply(formula_cnn, axis=1)
    next_step_df['result_new_cnn1'] = next_step_df.apply(formula_cnn1, axis=1)
    next_step_df['pred_score'] = CNN_pred.tolist()

    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['pred_score'])
    print('cnn scores', final_scores )

    next_step_df['pred_score'] = CNN_pred.tolist()
    next_step_df['result_new_cnn'] = next_step_df.apply(formula_cnn, axis=1)
    next_step_df['result_new_cnn1'] = next_step_df.apply(formula_cnn1, axis=1)
    next_step_df['pred_score'] = CNN_pred.tolist()

    result_new_scores_cnn = scores(list(map(int, y_val.tolist())),next_step_df['result_new_cnn'])
    print('New Maxrecord result CNN', result_new_scores_cnn )

    next_step_df['pred_score'] = CNN_pred.tolist()
    next_step_df['result_new_cnn'] = next_step_df.apply(formula_cnn, axis=1)
    next_step_df['result_new_cnn1'] = next_step_df.apply(formula_cnn1, axis=1)
    next_step_df['pred_score'] = CNN_pred.tolist()

    result_new_scores_cnn1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_cnn1'])
    print('Both Maxrecord result CNN 1', result_new_scores_cnn1 )

    #Random Forest
    next_step_df['randomForest_pred_score'] = randomForest_pred
    next_step_df['result_new_rf'] = next_step_df.apply(formula_rf, axis=1)
    next_step_df['result_new_rf1'] = next_step_df.apply(formula_rf1, axis=1)
    next_step_df['randomForest_pred_score'] = randomForest_pred


    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['randomForest_pred_score'])
    print('random forest scores', final_scores )

    next_step_df['randomForest_pred_score'] = randomForest_pred
    next_step_df['result_new_rf'] = next_step_df.apply(formula_rf, axis=1)
    next_step_df['result_new_rf1'] = next_step_df.apply(formula_rf1, axis=1)
    next_step_df['randomForest_pred_score'] = randomForest_pred

    result_new_scores_rf = scores(list(map(int, y_val.tolist())),next_step_df['result_new_rf'])
    print('New Maxrecord result RF', result_new_scores_rf )

    next_step_df['randomForest_pred_score'] = randomForest_pred
    next_step_df['result_new_rf'] = next_step_df.apply(formula_rf, axis=1)
    next_step_df['result_new_rf1'] = next_step_df.apply(formula_rf1, axis=1)
    next_step_df['randomForest_pred_score'] = randomForest_pred

    result_new_scores_rf1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_rf1'])
    print('Both Maxrecord result RF 1', result_new_scores_rf1 )


    #NN
    next_step_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['neuralNetwork_pred_score'])
    print('neuralNetwork scores', final_scores )


    result_new_scores_nn = scores(list(map(int, y_val.tolist())),next_step_df['result_new_nn'])
    print('New Maxrecord result NN', result_new_scores_nn )

    result_new_scores_nn1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_nn1'])
    print('Both Maxrecord result NN 1', result_new_scores_nn1 )


    #LR
    next_step_df['logistic_pred_score'] = logistic_pred.tolist()
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['logistic_pred_score'])
    print('Logistic Regression scores', final_scores )


    result_new_scores_lr = scores(list(map(int, y_val.tolist())),next_step_df['result_new_lr'])
    print('New Maxrecord result LR', result_new_scores_lr )

    result_new_scores_lr1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_lr1'])
    print('Both Maxrecord result LR 1', result_new_scores_lr1 )


    #SVM
    next_step_df['svm_pred_score'] = svm_pred.tolist()
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['svm_pred_score'])
    print('SVM prediction scores', final_scores )


    result_new_scores_svm = scores(list(map(int, y_val.tolist())),next_step_df['result_new_svm'])
    print('New Maxrecord result SVM', result_new_scores_svm )

    result_new_scores_svm1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_svm1'])
    print('Both Maxrecord result SVM 1', result_new_scores_svm1 )


    #KNN
    next_step_df['knn_pred_score'] = knn_pred.tolist()
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['knn_pred_score'])
    print('KNN prediction scores', final_scores )


    result_new_scores_knn = scores(list(map(int, y_val.tolist())),next_step_df['result_new_knn'])
    print('New Maxrecord result KNN', result_new_scores_knn )

    result_new_scores_knn1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_knn1'])
    print('Both Maxrecord result KNN 1', result_new_scores_knn1 )


    #DT
    next_step_df['decision_pred_score'] = decision_pred.tolist()
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['decision_pred_score'])
    print('decision tree scores', final_scores )


    result_new_scores_dt = scores(list(map(int, y_val.tolist())),next_step_df['result_new_dt'])
    print('New Maxrecord result DT', result_new_scores_dt )

    result_new_scores_dt1 = scores(list(map(int, y_val.tolist())),next_step_df['result_new_dt1'])
    print('Both Maxrecord result DT 1', result_new_scores_dt1 )    



    






    '''final_scores = scores(list(map(int, y_val.tolist())),next_step_df['neuralNetwork_pred_score'])
    print('neural network scores', final_scores )
    
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['logistic_pred_score'])
    print('logistic scores', final_scores )
    
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['svm_pred_score'])
    print('svm scores', final_scores )
    
    
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['decision_pred_score'])
    print('decision scores', final_scores )
    
    
    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['knn_pred_score'])
    print('knn scores', final_scores )'''
    


    '''print(next_step_df.shape)
    print(next_step_df.head())
    print(next_step_df.tail())'''

    '''#Read data
    ph_data = pd.read_excel('C:/Users/shrey/final_merged_phage_host_file.xlsx')
    print(ph_data.head())

    next_step_df = pd.merge(next_step_df,ph_data,on=['phage','host'])
    next_step_df.bit_score = (next_step_df.bit_score - next_step_df.bit_score.min())/(next_step_df.bit_score.max() - next_step_df.bit_score.min())
    print(next_step_df.shape)
    print(next_step_df.head())
    print(next_step_df.tail())
    #next_step = X_val+neg_select1+CNN_pred
    print(CNN_pred)
    result_all.append(CNN_result)

    pred_all=pred_all+CNN_pred.tolist()'''








print("#####")
print("done all iterations")
print(result_all)
print(pred_all)