# Build a binary classifier to predict whether a customer will subscribe to bank campaign scheme

# Importing the packages
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split as tts 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.model_selection import GridSearchCV as GSC
from sklearn.linear_model import LogisticRegression as LR
from imblearn.over_sampling import SMOTE,ADASYN
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score as ase
from sklearn.metrics import recall_score as rs
from sklearn.metrics import roc_auc_score as auc_score
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import itertools
import warnings
import operator
warnings.filterwarnings("ignore")
class Harshit:
    def __init__ (self):
        # Get the directory which containes the data file
        os.chdir("c:/Users/Harshit Mehta/Desktop/bank-additional/bank-additional/")
        # Readng files
        self.df = pd.read_csv("bank-additional-full.csv",sep=";")
        self.df=self.df.sample(frac=1)
        self.df1=self.df.drop(["duration","loan"],axis=1)# duration data is generally not available. loan not a good predictor using chi test
        self.df1['y'] = self.df1['y'].map(dict(yes=1, no=0))
        self.df1["previous"]=self.df1.apply(lambda x:self.remove_outliers(x),axis=1)
    def remove_outliers(self,x):
        if (x["previous"]!=0 and x["pdays"]==999):return 0
        else:return x["previous"]
    def ballpark_model(self):
        self.x_train,self.x_test,self.y_train,self.y_test=tts(pd.get_dummies(self.df1.drop(["y"],axis=1),drop_first=True),self.df1["y"],test_size=0.2,stratify=self.df1["y"],random_state=29)
        self.clf=LR(random_state=29,tol=0.00000000001)
        self.model=self.clf.fit(self.x_train,self.y_train)
    def correlation_numerical_features(self):
        plt.figure(figsize=(15,10))
        corr=self.df1.select_dtypes(["int64","float64"]).corr()
        sns.heatmap(corr,vmin=0,vmax=1,annot=True)
        plt.show()
    def chi_square_test(self):    
        chi_square_value=[]
        value=[]
        cat_columns=[]
        for i in self.df.drop(["y"],axis=1).columns:
            if self.df[i].dtype=="object":
                observed=pd.crosstab(self.df[i],self.df1["y"])
                c, p, dof, expected = ss.chi2_contingency(observed)
                cat_columns.append(i)
                value.append(p)
                chi_square_value.append(c)
        categorical_target_chi_value=pd.DataFrame({"column":cat_columns,"chi_value":chi_square_value,"p-value":value})
        categorical_target_chi_value["importance"]=categorical_target_chi_value["p-value"].apply(self.importance)
        plt.figure(figsize=(20,10))
        plt.title("p value of categorical variable with target variable")
        sns.stripplot(x="column",y="p-value",hue="importance",data=categorical_target_chi_value,size=15)
    def importance(self,x):
            if x<0.05: return "important"
            else: return "not important"
    def contacted(self,x):
        if x==999: return "not contacted previously"
        else: return "contacted"
    def current_campaign_contact(self,x):
        if x==1: return "Less than 1"
        elif x==2:return "two"
        elif x==3:return "three"
        else: return "more than 3 "
    def contacts_performed(self,x):
        if x==0: return "not contacted "
        else: return "contacted"
    def employed_cat(self,x):
        if x<=5099.1:return"Level 1"
        elif x<=5191.0 and x>5099.1:return "Level 2"
        else: return "Level 3"
    def emp_var_rate(self,x):
        if x<=-1.8:return"first_bin"
        elif x>-1.8 and x<=-0.1:return "Second_bin"
        else: return "Third_bin"
    def transformation(self):
        self.df1["pev_count_of_contacts"]=self.df1["previous"].apply(self.contacts_performed)
        self.df1["contacted"]=self.df1["pdays"].apply(self.contacted)
        self.df1["current_count_of_contacts"]=self.df1["campaign"].apply(self.current_campaign_contact)
        self.df1["nr.employed_cat"]=self.df1["nr.employed"].apply(self.employed_cat)
        self.df1["emp.var.rate_cat"]=self.df1["emp.var.rate"].apply(self.emp_var_rate)
        self.df1=self.df1.drop(["emp.var.rate","nr.employed"],axis=1)
        self.interaction_features()
    def interaction_features(self):
        #1-hot encoding categorical features for XGBoost
        self.cat=["job","marital","education","default","housing","contact","month","day_of_week","poutcome","pev_count_of_contacts",
                  "contacted","current_count_of_contacts","nr.employed_cat","emp.var.rate_cat"]
        self.cont=["age","campaign","pdays","previous","cons.price.idx","cons.conf.idx","euribor3m"]
        self.cat_encoded=[]
        for i in self.cat:
            temp = pd.get_dummies(self.df1[i], prefix=i, drop_first=True)
            self.df1[temp.columns] = temp
            self.cat_encoded.extend(temp.columns)
        #Interaction in continuous variables
        self.cont_inter = []
        for i, j in itertools.combinations(self.cont,2):
            self.df1[i+"_"+j]=self.df1[i]*self.df1[j]
            self.cont_inter.append(i+"_"+j)
        #Interaction between continuous and 1-hot categorical variable
        self.cat_cont=[]
        for i in self.cat_encoded:
            for j in self.cont:
                self.df1[i+"_"+j]=self.df1[i]*self.df1[j]
                self.cat_cont.append(i+'_'+j)
    def Xgboost_GridSearchCV(self):
        self.df1=self.df1.drop(self.cat,axis=1)
        self.x_train,self.x_test,self.y_train,self.y_test=tts(self.df1.drop(["y"],axis=1),self.df1["y"],test_size=0.2,stratify=self.df1["y"],random_state=29)
        self.clf=xgb.XGBClassifier(learning_rate=.01,n_estimators=80,random_state=29,scale_pos_weight=8.025803310613437,n_jobs=-1,objective="binary:logistic",booster="gbtree")
        #learning_rate=[.01,.02,.05,0.1,0.2]
        #n_estimators=[80,90,100]
        #params={"learning_rate":learning_rate,"n_estimators":n_estimators}
        #self.xgboost_tuned=GSC(self.model,params,cv=2,scoring="recall",n_jobs=-1)
        #self.model=self.xgboost_tuned.best_estimator_
        self.model=self.clf.fit(self.x_train,self.y_train)
        #self.top_10_features()
    def logistic_model_using_pca_plus_prediction(self):
        self.clf=LR(random_state=29,tol=0.000000000001)
        self.data4=self.df1.drop(["y"],axis=1)
        Y=self.df1["y"]
        scaler=StandardScaler()
        self.data4=scaler.fit_transform(self.data4)
        pca=PCA(n_components=100)# 100 by optimal number of principal components needed
        x=pca.fit_transform(self.data4)
        x_train1,x_test1,y_train1,y_test1=tts(x,Y,test_size=0.2,stratify=Y,random_state=29)
        self.model=self.clf.fit(x_train1,y_train1)
        probs = self.model.predict_proba(x_test1)
        prob1 = self.model.predict_proba(x_train1)
        #y_pred=self.model.predict(x_test1)
        preds = probs[:,1]
        self.fpr, self.tpr, self.threshold = roc_curve(y_train1, prob1[:,1])
        optimal_idx = np.argmax(self.tpr - self.fpr)
        self.optimal=self.threshold[optimal_idx]
        self.out1=pd.DataFrame({"y_true":y_test1,"y_pred":preds})
        self.out1["predicted_class"]=self.out1["y_pred"].apply(self.class_value)
        print(rs(self.out1["y_true"],self.out1["predicted_class"]))
        print(ase(self.out1["y_true"],self.out1["predicted_class"]))
        print(auc_score(y_test1,preds))
    def class_value(self,x):
        if x>=self.optimal:return 1
        else: return 0
    def keras_nn_model(self):
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)
        self.model1 = Sequential()
        self.model1.add(Dense(256, input_dim=428, activation='relu'))
        self.model1.add(Dense(128,activation='relu'))
        self.model1.add(Dense(64,activation='relu'))
        self.model1.add(Dense(4,activation='relu'))
        self.model1.add(Dense(1, activation='sigmoid'))
        scaler=StandardScaler()
        scaled=scaler.fit(self.x_train)
        self.X_train=scaled.fit_transform(self.x_train)
        self.X_test=scaled.fit_transform(self.x_test)
        adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model1.compile(optimizer=adam, loss=keras.losses.binary_crossentropy)
        self.model1.fit(self.X_train,self.y_train,epochs=1,batch_size=256,class_weight={0:1,1:8})
        pred=self.model1.predict(self.X_test)
        pred1=self.model1.predict(self.X_train)
        print(auc_score(self.y_test,pred))
        self.fpr, self.tpr, self.threshold = roc_curve(self.y_train, pred1.ravel())
        optimal_idx = np.argmax(self.tpr - self.fpr)
        self.optimal=self.threshold[optimal_idx]
        # for recall calculation
        self.out=pd.DataFrame({"y_true":self.y_test,"y_pred":pred.ravel()})
        self.out["predicted_class"]=self.out["y_pred"].apply(self.class_value)
        print(rs(self.out["y_true"],self.out["predicted_class"]))
        print(ase(self.out["y_true"],self.out["predicted_class"]))
    #def Oversampling_technique(self):
        #sm=SMOTE(random_state=29)
        #self.x_train1,self.y_trai1n=sm.fit_sample(self.x_train,self.y_train)
    #def xgboost_prediction_smote(self):
        #model=xgb.XGBClassifier(random_state=29,scale_pos_weight=1,n_jobs=-1,objective="binary:logistic",booster="gbtree")
        #clf=model.fit(self.x_train1,self.y_train1)
        #probs = clf.predict_proba(self.x_test.values)
        #y_pred=clf.predict(self.x_test.values)
        #preds = probs[:,1]
        #print(ase(self.y_test.values,y_pred))
        #print(rs(self.y_test.values,y_pred))
        #print(auc_score(self.y_test.values,preds))
    def prediction_on_test(self):
        self.probs = self.model.predict_proba(self.x_test)
        #self.y_pred=self.model.predict(self.x_test)
        self.preds = self.probs[:,1]
        self.prob1=self.model.predict_proba(self.x_train)
        #print(ase(self.y_test,self.y_pred))
        self.fpr, self.tpr, self.threshold = roc_curve(self.y_train, self.prob1[:,1])
        optimal_idx = np.argmax(self.tpr - self.fpr)
        self.optimal=self.threshold[optimal_idx]
        self.out1=pd.DataFrame({"y_true":self.y_test,"y_pred":self.preds})
        self.out1["predicted_class"]=self.out1["y_pred"].apply(self.class_value)
        print(rs(self.out1["y_true"],self.out1["predicted_class"]))
        print(ase(self.out1["y_true"],self.out1["predicted_class"]))
        #print(rs(self.y_test,self.y_pred))
        print(auc_score(self.y_test,self.preds))
    def top_10_features(self):
        lis=self.x_train.columns
        feat_imp = {lis[i]:self.model.feature_importances_[i] for i in range(len(lis))}
        sorted_feat=sorted(feat_imp.items(),key=operator.itemgetter(1),reverse=True)
        feature_dataframe=pd.DataFrame.from_dict(sorted_feat)
        feature_dataframe.columns=["feature","importance"]
        plt.figure(figsize=(20,15))
        sns.stripplot("feature","importance",data=feature_dataframe.iloc[:10,],size=10)
if __name__ == "__main__":
    Hm = Harshit()
    Hm.ballpark_model()
    # prediction using ballpark model
    Hm.prediction_on_test()
    #correlation between numerical attributes originally present
    Hm.correlation_numerical_features()
    Hm.chi_square_test()
    Hm.transformation()
    Hm.Xgboost_GridSearchCV()
    # prediction using xgboost model
    Hm.prediction_on_test()
    Hm.logistic_model_using_pca_plus_prediction()
    Hm.keras_nn_model()
    
