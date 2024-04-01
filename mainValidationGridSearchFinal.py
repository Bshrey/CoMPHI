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


print("Program started",datetime.datetime.now())
EPOCHS=1
INIT_LR=1e-3

def CNNModel(dna_tra_fea,pro_tra_fea, y_tra3, dna_val_fea,pro_val_fea, y_val3,shape0,shape1,shape2,shape3,shape4):
    
    model=None
    model=dna_pro_model(INIT_LR,EPOCHS,shape0,shape1,shape2,shape3,shape4)
    model.summary()   
    model.fit([dna_tra_fea,pro_tra_fea], y_tra3, epochs=EPOCHS, batch_size=8)
    y_pred_val = model.predict([dna_val_fea,pro_val_fea]).flatten()
    test = y_pred_val.tolist()    
    return y_pred_val

def randomForestModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val3):
    print("starting random forest regression...")  

    param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
    }

    rfc = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='precision', cv=5)

    grid_search.fit(np.concatenate((dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea),axis=1),y_tra)

    y_pred_val = grid_search.predict(np.concatenate((dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea),axis=1)).flatten()
    test = y_pred_val.tolist()
    print("random forest test",type(test),len(test))
    print("random forest preds",type(y_pred_val))
    print("random forest preds",y_pred_val[0],y_pred_val[1],y_pred_val[2],y_pred_val[3])
    print("random forest y_true",type(y_val3))
    print("random forest y_true",y_val3.tolist()[0],y_val3.tolist()[1],y_val3.tolist()[2],y_val3.tolist()[3],len(y_val3.tolist()),sum(y_val3.tolist()))
    print("random forest Accuracy score",precision_score(y_val3,y_pred_val.round()))
    print("random forest Precision score",precision_score(y_val3,y_pred_val.round()))
    print("random forest Recall score",recall_score(y_val3,y_pred_val.round()))
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    return y_pred_val

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
    #print(file_path1)    
    #print(file_path2)
    #print(strs)
    #print(type(data),len(data))
    #print(data)
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
    
    
def getSimPH(X_val, X_tra, simPH_path):
    v_phage=list(set([mm[0]for mm in X_val]))
    phage=list(set([mm[0]for mm in X_tra]))
    host=list(set([mm[1]for mm in X_tra]))
    df=pd.read_excel(simPH_path, header = None)
    
    for p in v_phage:
        con1=df[1].isin(host)
        con2=df[0] = p
        con =con1 & con2
        simph = df[con]  
        simph_list.append(simph)
    return simph_list
    
def getSimPP(X_val, X_tra, simPP_path):
    v_phage=list(set([mm[0]for mm in X_val]))
    phage=list(set([mm[0]for mm in X_tra]))
    host=list(set([mm[1]for mm in X_tra]))
    df=pd.read_excel(simPH_path, header = None)
    
    for p in v_phage:
        con1=df[0].isin(phage)
        con2=df[0] = p
        con =con1 & con2
        simpp = df[con]  
        simpp_list.append(simpp)
    return simpp_list
    
    
def getSimHH(X_val, X_tra, simHH_path):
    v_host=list(set([mm[1]for mm in X_val]))
    phage=list(set([mm[0]for mm in X_tra]))
    host=list(set([mm[1]for mm in X_tra]))
    df=pd.read_excel(simPH_path, header = None)
    
    for h in v_host:
        con1=df[0].isin(host)
        con2=df[0] == h
        con =con1 & con2
        simhh = df[con]  
        simhh_list.append(simhh)
    return simhh_list

result_all=[]
pred_all=[]
test_y_all=[]

data1=pd.read_csv('dna_pos_neg3_p.csv',header=None,sep=',')
print(data1.shape)
data1=data1[data1[2]==1]
print(data1.shape)
allinter=[str(data1.loc[i,0])+','+str(data1.loc[i,1]) for i in data1.index]
#kf = KFold(n_splits=5,random_state=1)#original code with error for random_state
kf = KFold(n_splits=5)
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
    next_step_df = pd.DataFrame(next_step,columns=['phage','host','score'])
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

    randomForest_pred=randomForestModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['randomForest_pred_score'] = randomForest_pred
    print('next_step_df after pred shape',next_step_df.shape)

    '''neuralNetwork_result,neuralNetwork_pred = neuralNetwork(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['neuralNetwork_pred_score'] = neuralNetwork_pred.tolist()
    print('next_step_df after neural network pred shape',next_step_df.shape)

    logistic_result,logistic_pred=logisticRegressionModel(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['logistic_pred_score'] = logistic_pred.tolist()
    print('next_step_df after logistic pred shape',next_step_df.shape)
    #print('xtest_df head',next_step_df.head())
    #print('next_step_df tail', next_step_df.tail())

    svm_result,svm_pred=supportVectorMachine(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['svm_pred_score'] = svm_pred.tolist()
    print('next_step_df after svm pred shape',next_step_df.shape)

    knn_result,knn_pred=knn(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['knn_pred_score'] = knn_pred.tolist()
    print('next_step_df after knn pred shape',next_step_df.shape)

    decision_result,decision_pred=decisionTree(dna_tra_pha_fea,dna_tra_bac_fea,pro_tra_pha_fea,pro_tra_bac_fea,y_tra,dna_val_pha_fea,dna_val_bac_fea,pro_val_bac_fea,pro_val_pha_fea,y_val)
    next_step_df['decision_pred_score'] = decision_pred.tolist()
    print('next_step_df after decision pred shape',next_step_df.shape)'''


    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['randomForest_pred_score'])
    print('random forest scores', final_scores )

    final_scores = scores(list(map(int, y_val.tolist())),next_step_df['pred_score'])
    print('cnn scores', final_scores )

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
    a


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