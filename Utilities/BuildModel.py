from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from joblib import dump, load
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from keras.models import Sequential
import keras

import pandas as pd
import tensorflow as tf
    
import helper as h
import configparser


def Train_LogisticRegression_Classifier(datasetFilePath,targetAttribute,saveModelPath):

    dataset = pd.read_csv(datasetFilePath)

    target = dataset[targetAttribute]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=target)
    x_train = train_dataset.drop(targetAttribute, axis=1)
    x_test = test_dataset.drop(targetAttribute, axis=1)

    numerical_columns = []
    categorical_columns = []
    
    col_num = len(dataset.columns) 
    i =0 
    for i in range(col_num - 1):
        if(is_numeric_dtype(dataset.iloc[:,i])):
            numerical_columns.append(str(dataset.columns[i]))

    categorical_columns = x_train.columns.difference(numerical_columns)

    # print("Numeric Columns - ")
    # print(numerical_columns)


    # print("Categorical Columns - ")
    # print(categorical_columns)

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    
    clf = Pipeline(steps=[('preprocessor', transformations),
                         ('classifier', LogisticRegression())])
    model = clf.fit(x_train, y_train)

    dump(model, saveModelPath) 

    print('Trained model save to provied path ')

    loaded_model = load(saveModelPath)
    a = loaded_model.predict_proba(x_test[12:16]) 
    print('Probablity with saved modle for Test instance - ')
    print(a)

    pred = loaded_model.predict(x_test) 
    #print(pred)

    accuracy = accuracy_score(y_test, pred)
    print("\n\t\t\t" + saveModelPath + " -------------> Accuracy = " + str(accuracy) + "\n")

def Train_RandomForest_Classifier(datasetFilePath,targetAttribute,saveModelPath):

    dataset = pd.read_csv(datasetFilePath)

    target = dataset[targetAttribute]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=target)
    x_train = train_dataset.drop(targetAttribute, axis=1)
    x_test = test_dataset.drop(targetAttribute, axis=1)

    numerical_columns = []
    categorical_columns = []
    
    col_num = len(dataset.columns) 
    i =0 
    for i in range(col_num - 1):
        if(is_numeric_dtype(dataset.iloc[:,i])):
            numerical_columns.append(str(dataset.columns[i]))

    categorical_columns = x_train.columns.difference(numerical_columns)

    # print("Numeric Columns - ")
    # print(numerical_columns)

    # print("Categorical Columns - ")
    # print(categorical_columns)

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    
    clf = Pipeline(steps=[('preprocessor', transformations),
                         ('classifier', RandomForestClassifier())])
    model = clf.fit(x_train, y_train)

    dump(model, saveModelPath) 

    print('Trained model save to provied path ')

    loaded_model = load(saveModelPath)
    a = loaded_model.predict_proba(x_test[12:16]) 
    print('Probablity with saved modle for Test instance - ')
    print(a)

    pred = loaded_model.predict(x_test) 
    #print(pred)

    accuracy = accuracy_score(y_test, pred)
    print("\n\t\t\t" + saveModelPath + " -------------> Accuracy = " + str(accuracy) + "\n")

def Train_SVM_Classifier(datasetFilePath,targetAttribute,saveModelPath):

    dataset = pd.read_csv(datasetFilePath)

    target = dataset[targetAttribute]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=target)
    x_train = train_dataset.drop(targetAttribute, axis=1)
    x_test = test_dataset.drop(targetAttribute, axis=1)

    numerical_columns = []
    categorical_columns = []
    
    col_num = len(dataset.columns) 
    i =0 
    for i in range(col_num - 1):
        if(is_numeric_dtype(dataset.iloc[:,i])):
            numerical_columns.append(str(dataset.columns[i]))

    categorical_columns = x_train.columns.difference(numerical_columns)

    # print("Numeric Columns - ")
    # print(numerical_columns)

    # print("Categorical Columns - ")
    # print(categorical_columns)

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', transformations),('svc', SVC(gamma='auto',probability=True))])
    model = clf.fit(x_train, y_train)
    dump(model, saveModelPath) 

    print('Trained model save to provied path ')

    loaded_model = load(saveModelPath)
    # a = loaded_model.predict_proba(x_test[12:16]) 
    # print('Probablity with saved modle for Test instance - ')
    # print(a)

    pred = loaded_model.predict(x_test) 
    #print(pred)

    accuracy = accuracy_score(y_test, pred)
    print("\n\t\t\t" + saveModelPath + " -------------> Accuracy = " + str(accuracy) + "\n")

def Train_DNNModel(datasetFilePath,targetAttribute,saveModelPath) :
    dataset = pd.read_csv(datasetFilePath)

    dataset.drop(dataset[dataset['native_country'] == '?'].index, inplace = True)
    dataset.drop(dataset[dataset['occupation'] == '?'].index, inplace = True)
    dataset.drop(dataset[dataset['workclass'] == '?'].index, inplace = True)
    
    target = dataset[targetAttribute]

    numerical_columns = []
    categorical_columns = []

    col_num = len(dataset.columns) 
    i =0 
    for i in range(col_num - 1):
        if(is_numeric_dtype(dataset.iloc[:,i])):
            numerical_columns.append(str(dataset.columns[i]))

    df = dataset.drop(targetAttribute, axis=1)

    categorical_columns = df.columns.difference(numerical_columns)

    oneHotEncoded_DF  = h.one_hot_encode_data(df,categorical_columns)
    #print(oneHotEncoded_DF)

    normalizedDF = h.normalize_data(oneHotEncoded_DF,numerical_columns,targetAttribute)
    #print(normalizedDF)

    df = pd.DataFrame(normalizedDF.columns.values)
    df.to_csv('Train_DNNModel.csv')

    train_dataset, test_dataset, y_train, y_test = train_test_split(normalizedDF,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=target)
    #x_train = train_dataset.drop(targetAttribute, axis=1)
    #x_test = test_dataset.drop(targetAttribute, axis=1)

    ann_model = keras.Sequential()
    ann_model.add(keras.layers.Dense(20, input_shape=(train_dataset.shape[1],), kernel_regularizer=keras.regularizers.l1(0.001), activation=tf.nn.relu))
    ann_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    ann_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    ann_model.fit(train_dataset, y_train, validation_split=0.20, epochs=100, verbose=0, class_weight={0:1,1:2})


    # evaluate the keras model
    _, accuracy = ann_model.evaluate(test_dataset, y_test, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))

    ann_model.save(saveModelPath)
    print("Saved model to disk")


configs = configparser.ConfigParser()

# --------- -------- --------- Adult Income Dataset - AI360
#configs.read('Config/adult_AI360_LR.cfg')
# configs.read('Config/adult_AI360_RF.cfg')
# configs.read('Config/adult_AI360_SVM.cfg')
configs.read('Config/adult_AI360_DNN.cfg')

# --------- -------- --------- German Credit Dataset - AI360
#configs.read('Config/germanCredit_AI360_LR.cfg')
#configs.read('Config/germanCredit_AI360_RF.cfg')
#configs.read('Config/germanCredit_AI360_SVM.cfg')
#configs.read('Config/germanCredit_AI360_DNN.cfg')

# --------- -------- --------- COMPOS Dataset - AI360
# configs.read('Config/compas_AI360_LR.cfg')
# configs.read('Config/compas_AI360_RF.cfg')
# configs.read('Config/compas_AI360_SVM.cfg')
# configs.read('Config/compas_AI360_DNN.cfg')



dataset_filePath = str(configs['values']['dataset_filePath'])
saveModelPath = str(configs['values']['modlePath'])
targetAttribute = str(configs['values']['targetAttr'])

# Train_LogisticRegression_Classifier(dataset_filePath,targetAttribute,saveModelPath)

# Train_RandomForest_Classifier(dataset_filePath,targetAttribute,saveModelPath)

# Train_SVM_Classifier(dataset_filePath,targetAttribute,saveModelPath)

Train_DNNModel(dataset_filePath,targetAttribute,saveModelPath)