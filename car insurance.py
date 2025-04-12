import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer ## HAndle Missing Values
from sklearn.preprocessing import StandardScaler ## Feature Scaling
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r"C:\Users\Pranat\Desktop\file\car_insurance_claim.csv")
df.drop(['ID', 'KIDSDRIV','BIRTH'], axis=1, inplace=True)
df.dropna(axis=0,inplace=True)
le = LabelEncoder()
def cleaning(data=df):
	df['INCOME'] = df['INCOME'].astype(str)
	df['INCOME'] = df['INCOME'].str.replace('$','',regex=True)
	df['INCOME'] = df['INCOME'].str.replace(',','',regex=True)
	df['INCOME'] = df['INCOME'].astype(int)

	df['HOME_VAL'] = df['HOME_VAL'].astype(str)
	df['HOME_VAL'] = df['HOME_VAL'].str.replace('$','',regex=True)
	df['HOME_VAL'] = df['HOME_VAL'].str.replace(',','',regex=True)
	df['HOME_VAL'] = df['HOME_VAL'].astype(int)

	df['BLUEBOOK'] = df['BLUEBOOK'].astype(str)
	df['BLUEBOOK'] = df['BLUEBOOK'].str.replace('$','',regex=True)
	df['BLUEBOOK'] = df['BLUEBOOK'].str.replace(',','',regex=True)
	df['BLUEBOOK'] = df['BLUEBOOK'].astype(int)

	df['OLDCLAIM'] = df['OLDCLAIM'].astype(str)
	df['OLDCLAIM'] = df['OLDCLAIM'].str.replace('$','',regex=True)
	df['OLDCLAIM'] = df['OLDCLAIM'].str.replace(',','',regex=True)
	df['OLDCLAIM'] = df['OLDCLAIM'].astype(int)

	df['CLM_AMT'] = df['CLM_AMT'].astype(str)
	df['CLM_AMT'] = df['CLM_AMT'].str.replace('$','',regex=True)
	df['CLM_AMT'] = df['CLM_AMT'].str.replace(',','',regex=True)
	df['CLM_AMT'] = df['CLM_AMT'].astype(int)

	df['PARENT1'] = le.fit_transform(df['PARENT1'])

	df['CAR_USE'] = le.fit_transform(df['CAR_USE'])

	df['RED_CAR'] = le.fit_transform(df['RED_CAR'])

	df['REVOKED'] = le.fit_transform(df['REVOKED'])

	df['MSTATUS'] = df['MSTATUS'].astype(str)
	df['MSTATUS'] = df['MSTATUS'].str.replace('z_','',regex=True)
	df['MSTATUS'] = le.fit_transform(df['MSTATUS'])

	df['GENDER'] = df['GENDER'].astype(str)
	df['GENDER'] = df['GENDER'].str.replace('z_','',regex=True)
	df['GENDER'] = le.fit_transform(df['GENDER'])

	df['EDUCATION'] = df['EDUCATION'].astype(str)
	df['EDUCATION'] = df['EDUCATION'].str.replace('z_','',regex=True)
	df['EDUCATION'] = df['EDUCATION'].str.replace('<','',regex=True)
	df['EDUCATION'] = le.fit_transform(df['EDUCATION'])

	df['OCCUPATION'] = df['OCCUPATION'].astype(str)
	df['OCCUPATION'] = df['OCCUPATION'].str.replace('z_','',regex=True)
	df['OCCUPATION'] = le.fit_transform(df['OCCUPATION'])

	df['CAR_TYPE'] = df['CAR_TYPE'].astype(str)
	df['CAR_TYPE'] = df['CAR_TYPE'].str.replace('z_','',regex=True)
	df['CAR_TYPE'] = le.fit_transform(df['CAR_TYPE'])

	df['URBANICITY'] = df['URBANICITY'].astype(str)
	df['URBANICITY'] = df['URBANICITY'].str.replace('z_','',regex=True)
	df['URBANICITY'] = le.fit_transform(df['URBANICITY'])

df.apply(cleaning)

y = df[['CLAIM_FLAG']]
x = df.loc[:, df.columns != 'CLAIM_FLAG']

pca = PCA(n_components=5)
X = pca.fit_transform(x)

def classify(model,x,y):
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	pipeline_model = Pipeline([('sc',StandardScaler()),('clf',model)])
	pipeline_model.fit(x_train,np.ravel(y_train,order='C'))

	print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
	y_pred = pipeline_model.predict(x_test)
	c = confusion_matrix(y_test,y_pred)
	print(c)
	print(classification_report(y_test, y_pred))

model = XGBClassifier()
classify(model,X,y) ##max accuracy 99.8%, with minimum amounts of False positives and False negatives.

#estimators = [('rf',RandomForestClassifier())]
#model = StackingClassifier(estimators=estimators,final_estimator=SVC())
#classify(model,X,y)

#model = SVC()
#classify(model,X,y)

