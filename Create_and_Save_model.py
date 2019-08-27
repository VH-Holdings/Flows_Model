# Packages required
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


def CreateModel(seed):
    model = ensemble.RandomForestClassifier(random_state=seed)
    
    #Training data
    #Train models
    X = TrainData.drop(DropList, axis=1)
    X_train, X_test = train_test_split(X, test_size = 0.3, random_state = seed)
    
    #create model for each ResponseVariable
    for ResponseVariable in ResponseVariables:
        #Create model
        lenc = LabelEncoder()
        Y = TrainData[ResponseVariable]
        Y = lenc.fit_transform(Y)
        np.save(ResponseVariable+'classes.npy', lenc.classes_)
        Y_train, Y_test = train_test_split(Y, test_size = 0.3, random_state = seed)
        
        model = clf.fit(X_train,Y_train)
        
        joblib.dump(model, ResponseVariable+'Model.pkl')


#Train model with data
#Initialise dataframe and training data
FileName = "\LabeledData"
TrainData = pd.read_excel(r'C:\Users\20064032\Documents\Machine learning\Machine learning - Geotab\Trok data'+FileName+'.xlsx')
LabeledData = pd.read_excel(r'C:\Users\20064032\Documents\Machine learning\Machine learning - Geotab\Trok data\TestData.xlsx')

TrainData = TrainData[TrainData['Engine road speed'] < 10]
#LabeledData = LabeledData[LabeledData['Engine road speed'] < 10]

#Initialise variables
DropList = ['Truck', 'Date', 'Outside air temperature', 'Stopped', 'Group', 'Loading_Unloading', 'Loading', 'Unloading', 'LoadingOrUnloading']
ResponseVariables = ['Loading', 'Unloading', 'LoadingOrUnloading']
ModelAcc = pd.read_csv('ModelAcc.csv')
ModelAcc = ModelAcc.drop('Unnamed: 0', axis=1)
LabeledAcc = pd.read_csv('LabeledAcc.csv')
LabeledAcc = LabeledAcc.drop('Unnamed: 0', axis=1)


#Initialise machine learning model
seeds = np.random.randint(1,99999999,5)

#Test accuracy for different seeds
for seed in seeds:
    clf = ensemble.RandomForestClassifier(random_state=seed)
    
    #Train models
    Train, Test = train_test_split(TrainData, test_size = 0.3, random_state = seed)
    X_train = Train.drop(DropList, axis=1)
    X_test = Test.drop(DropList, axis=1)
    
    Acc = [seed]
    ModelScore = [seed]
    for ResponseVariable in ResponseVariables:
        #Create model
        lenc = LabelEncoder()
        Y = TrainData[ResponseVariable]
        Y = lenc.fit_transform(Y)
        np.save(ResponseVariable+'classes.npy', lenc.classes_)        
        Y_train = Train[ResponseVariable]
        Y_test = Test[ResponseVariable]
        
        model = clf.fit(X_train,Y_train)
        
        estimator = model.estimators_[5]
        
        Acc.append(model.score(X_test, Y_test))
        
        #joblib.dump(model, ResponseVariable+'Model.pkl')
        
        #Test Accuracy
        Labeled_X = LabeledData.drop(DropList, axis=1)
        lenc = LabelEncoder()
        Labeled_Y = LabeledData[ResponseVariable]
        Labeled_Y = lenc.fit_transform(Labeled_Y)
        
        ModelScore.append(model.score(Labeled_X, Labeled_Y))
     
    Row = pd.Series(Acc, index=ModelAcc.columns)
    ModelAcc = ModelAcc.append(Row, ignore_index=True)
    Row = pd.Series(ModelScore, index=LabeledAcc.columns)
    LabeledAcc = LabeledAcc.append(Row, ignore_index=True)


#Save model accuracies for future reference
ModelAcc.to_csv('ModelAcc.csv')
LabeledAcc.to_csv('LabeledAcc.csv')


#Calculate average accuracy for labeled data
averages = []
i = 0
for i in range(len(LabeledAcc)):
    averages.append(np.average(LabeledAcc.values[i,1:4]))
    
col = pd.Series(averages)    
LabeledAcc['AverageAcc'] = col.values


#Choose best seed for model based on accuracy
TopSeed = int(LabeledAcc[LabeledAcc['AverageAcc']==max(LabeledAcc['AverageAcc'])]['Seed'].values)
#Create and save top performing model
CreateModel(TopSeed)



#col = X.columns
#
#dotfile = six.StringIO()
#i_tree = 0

#for tree_in_forest in model.estimators_:
#    export_graphviz(tree_in_forest,out_file='tree.dot',
#    feature_names=col,
#    filled=True,
#    rounded=True)
#    (graph,) = pydot.graph_from_dot_file('tree.dot')
#    name = 'tree' + str(i_tree)
#    graph.write_png(name+  '.png')
#    os.system('dot -Tpng tree.dot -o tree.png')
#    i_tree +=1
