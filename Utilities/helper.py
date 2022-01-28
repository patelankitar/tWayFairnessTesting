import pandas as pd

import csv
from os import X_OK
import random
from os.path import exists
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import subprocess
import keras
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import load
from pandas import read_csv
import os
import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

# DiCE imports
import dice_ml

from datetime import datetime

def CereateACTSParamFile(dataset_filepath,acts_system,target_feature,file_suffix,logfile) :
    df = pd.read_csv(dataset_filepath)
    # print(df.head())
    x_trdfain = df.drop(target_feature, axis=1)

    continuous_featuresList = []
    
    ACTS_paramFilePath = "Data/ACTS_param/"+acts_system+"_"+file_suffix+ ".txt"
    
    f= open(ACTS_paramFilePath,"w+")

    f.write("[System]\n")
    f.write("Name : " +acts_system+ "\n\n")
    f.write("[Parameter]\n")

    col_num = len(df.columns) 
    i =0 
    for i in range(col_num - 1):
        target = str(df.columns[col_num-1])
        attributeName = str(df.columns[i])

        if(is_string_dtype(df.iloc[:,i])):
            # merge all column values into a comma seperated string 
            x_str = ",".join(df.iloc[:,i].astype('str').unique())

            x_str =x_str.replace(',?','')
            f.write(str(df.columns[i]) + " (enum) : " + x_str + "\n")
        elif(is_numeric_dtype(df.iloc[:,i])):
            continuous_featuresList.append(str(df.columns[i]))

            int_unique, indices = np.unique(df.iloc[:,i], return_inverse=True)

            logfile.write(" Number of Unique values = " + str(len(int_unique)) + " for Int attribute = "+ attributeName + " "'\n')
            
            # if the Integer unique values are less than 8 , take unique values, else use Discretization
            if(len(int_unique) <= 8):
                logfile.write("\t"+ " writing all unique values to param file")
                x_int = np.array2string(int_unique,separator=',')
                x_int = x_int.replace("[","").replace("]","")
                f.write(str(df.columns[i]) + " (int) : " + x_int + "\n")
            else:
                logfile.write("\t"+ " Get all values using DT for Int attribute "+ attributeName + " "'\n')
                logfile.write("\t"+ " Generating DT without specifiying depth \n")
                isDepth = 0
                maxDepth = 0
                GenerateDecisionTreeBins(dataset_filepath,acts_system,attributeName,target,file_suffix,isDepth,maxDepth)
                file = open("Data/DecisionTreeBins/"+acts_system+"_"+file_suffix+"/"+attributeName+'.csv')
                csvreader = csv.reader(file)
                next(csvreader)
                lines= len(list(csvreader))
                logfile.write("\t"+ " DT length without specifiying depth = "+ str(lines)+"\n")
                x_int = ','.join([str(i) for i in range(lines)])
            
                binSize = lines

                # print("Int Bin Size = ", binSize)
                if(binSize > 8):
                    isDepth = 1
                    maxDepth = 3
                    logfile.write("\t"+ " # Bins  > 8 , So generating DT of Depth = "+ str(maxDepth) +"\n")
                    GenerateDecisionTreeBins(dataset_filepath,acts_system,attributeName,target,file_suffix,isDepth,maxDepth)
                    file = open("Data/DecisionTreeBins/"+acts_system+"_"+file_suffix+"/"+attributeName+'.csv')
                    csvreader = csv.reader(file)
                    next(csvreader)
                    lines= len(list(csvreader))
                    # print(lines)
                    x_int = ','.join([str(i) for i in range(lines)])
                    
                f.write(str(df.columns[i]) + " (int) :" + x_int + "\n")
    f.close()

    logfile.write(str(datetime.now())+ " - " +"ACTS parameter file created"+ '\n')
    return continuous_featuresList

def Get_tWay_abstractTC(dataset_filepath,acts_system,tWay,file_suffix,target_feature,logfile) :
    ACTS_paramFilePath = "Data/ACTS_param/"+acts_system+"_"+file_suffix+ ".txt"
    
    ACTS_jar_path = 'ACTS3/acts_3.0.jar'
    tWayFilePath = "Data/tWay_Abstract_TC/" +acts_system+ "_" +tWay+ "way_abstract_TC_" +file_suffix+ ".csv"

    try:
        subprocess.call(['java','-Dalgo=ipog','-Ddoi='+tWay , '-Doutput=csv', '-jar', ACTS_jar_path , ACTS_paramFilePath, tWayFilePath])
        lines = open(tWayFilePath).readlines()
    
        open("Data/tWay_Concrete_TC/"+acts_system+ "_" +tWay+ "way_concrete_TC_" +file_suffix+ ".csv", 'w').writelines(lines[6:])
    
        logfile.write(str(datetime.now())+ " - " +"abstract test case file created"+ '\n')

        return 1
    
    except OSError:
        print("An exception occurred") 
        logfile.write(str(datetime.now())+ " - " +"Error occured while generating abstract test case file "+ '\n')
        return 0

def Get_tWay_ConcreteTC(concreteTC_filepath,system_name,tWay,file_suffix,logFile):
    print("Getting concrete TC")
    csv_files = glob.glob(os.path.join('Data/DecisionTreeBins/'+system_name+"_"+file_suffix+"/", "*.csv"))
    # print(csv_files)

    # # random.seed(a=2)
  
    for file in csv_files :
        # print(file)
        attribute = file.split('/')[len(file.split("/")) - 1].split('.')[0]
        # print(attribute)
        random.seed(a=2)
  
        df = pd.read_csv(file)
        df = df.iloc[::-1]
        for index, row in df.iterrows():
            # print(index)
            # print(int(row['min']))
            # print(int(row['max']))
            ReplaceAbstractValues_Random(concreteTC_filepath,concreteTC_filepath,attribute,index,int(row['min']),int(row['max']))

    print("Concrete TC file created ")
    #print(concreteTC_filepath)
    logFile.write(str(datetime.now())+ " - " +"Concrete TC file created"+ '\n')


def GetModelPrediction (model_Path,inputFile,outputFile,predColName):
    loaded_model = load(model_Path)
    print("Getting Model prediction ")
    #print(inputFile)
    TestData = read_csv(inputFile)
    #print('Model Execution:')
    pred = loaded_model.predict(TestData) 
    #print(pred)

    export_df = pd.DataFrame(pred)  
    export_df.columns =[predColName]
    export_df = export_df.join(TestData)

    print('Export to CSV')
    export_df.to_csv(outputFile, index=False) 
    print("Done!")


def Get_DiCECounterfactuals(model_path,dataset_filepath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,restrict_attribute,num_CF,compareFile,logFile):
    loaded_model = load(model_path)

    # print(continuous_featuresList)
    # print(target_feature)

    logFile.write(str(datetime.now())+ " - " +"call to DiCE "+ '\n')
    trainingDataset = pd.read_csv(dataset_filepath)
    d = dice_ml.Data(dataframe=trainingDataset, continuous_features=continuous_featuresList, outcome_name=target_feature)

    # Using sklearn backend
    m = dice_ml.Model(model=loaded_model, backend="sklearn")
    # Using method=random for generating CFs
    #exp = dice_ml.Dice(d, m, method="random")
    exp = dice_ml.Dice(d, m)

    df = pd.read_csv(concreteTC_filepath)

    length = df.shape[0] 

    # Run Concrete TC one by one in a for loop 
    test_instance_df = pd.DataFrame()
    cf_instance_df = pd.DataFrame()
    Concrete_TC_num_for_CF_found = []
    count_of_concrete_TC_CF_found = []
    tc_num_list = []

    
    logFile.write(str(datetime.now())+ " - " +"Executing Concrete TC in natural order - "+ '\n')
    tc_num_list = list(range(0, length))

    # ***** ---- IMPORTANT - Remember to comment 
    # logFile.write(str(datetime.now())+ " - " +"Executing Concrete TC in random order - "+ '\n')
    # seedVal = 10
    # random.seed(a=seedVal)
    # logFile.write(str(datetime.now())+ " - " +"seed value =  - "+ str(seedVal) +  '\n')
    # numbers = range(0,length)
    # tc_num_list = random.sample(numbers, length)


    logFile.write(str(datetime.now())+ " - " +"Concrete TC execution sequence - "+ '\n')
    logFile.write("\t\t"+ str(tc_num_list)+ '\n')
    j = 1

    for i in tc_num_list:
        e1 = exp.generate_counterfactuals(df.loc[[i]], total_CFs=num_CF, desired_class="opposite", features_to_vary=restrict_attribute)
        j = j + 1 
        #e1.visualize_as_list()
        #print(e1.cf_examples_list[0].test_instance_df)
        #print(e1.cf_examples_list[0].final_cfs_df)

        if (e1.cf_examples_list[0].final_cfs_df) is not None:
        #     print("list is  empty")
        # else:
            # print("list is not empty")
            Concrete_TC_num_for_CF_found.append(i)
            count_of_concrete_TC_CF_found.append(j)
            test_instance_df = test_instance_df.append(e1.cf_examples_list[0].test_instance_df)
            cf_instance_df = cf_instance_df.append(e1.cf_examples_list[0].final_cfs_df)
            

    print("Concrete TC sequence #  (count) for which Dice generated a CF - ")
    print(count_of_concrete_TC_CF_found) 
    logFile.write(str(datetime.now())+ " - " +"Concrete TC count # for which Dice generated a CF - "+ '\n')
    logFile.write("\t\t"+ str(count_of_concrete_TC_CF_found)+ '\n')

    print("Concrete TC  # for which Dice generated a CF - ")
    print(Concrete_TC_num_for_CF_found) 
    logFile.write(str(datetime.now())+ " - " +"Concrete TC  # for which Dice generated a CF - "+ '\n')
    logFile.write("\t\t"+ str(Concrete_TC_num_for_CF_found)+ '\n')

    cf_instance_df.to_csv(CF_outputFile,index=False)
    logFile.write(str(datetime.now())+ " - " +"CSV file created with DiCE output CF"+ '\n')

    test_instance_df.to_csv(CF_origionalTC_ForCF_File,index=False)
    logFile.write(str(datetime.now())+ " - " +"CSV file created with Concrete TC for which DiCE was able to generate CF"+ '\n')

    # Compare the Concrete TC and the CF to see what was changed  
    ne_stacked = (test_instance_df != cf_instance_df).stack()
    changed = ne_stacked[ne_stacked]
    difference_locations = np.where(test_instance_df != cf_instance_df)
    #print(difference_locations)
    changed_from = test_instance_df.values[difference_locations]
    changed_to = cf_instance_df.values[difference_locations]
    df = pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)
    #print(df)
    df.to_csv(compareFile,index=False)


def Get_DiCECounterfactuals_for_DNN(model_path,dataset_filepath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,restrict_attribute,num_CF,compareFile,logFile):
    #loaded_model = load(model_path)
    loaded_model = model_path

    # print(continuous_featuresList)
    # print(target_feature)

    logFile.write(str(datetime.now())+ " - " +"call to DiCE "+ '\n')
    trainingDataset = pd.read_csv(dataset_filepath)
    d = dice_ml.Data(dataframe=trainingDataset, continuous_features=continuous_featuresList, outcome_name=target_feature)

    backend = 'TF'+tf.__version__[0]  # TF1

    # Using sklearn backend
    m = dice_ml.Model(model_path=loaded_model, backend=backend)
    # Using method=random for generating CFs
    exp = dice_ml.Dice(d, m, method="random")

    df = pd.read_csv(concreteTC_filepath)

    length = df.shape[0] 

    # Run Concrete TC one by one in a for loop 
    test_instance_df = pd.DataFrame()
    cf_instance_df = pd.DataFrame()
    Concrete_TC_num_for_CF_found = []
    count_of_concrete_TC_CF_found = []
    tc_num_list = []

    
    logFile.write(str(datetime.now())+ " - " +"Executing Concrete TC in natural order - "+ '\n')
    tc_num_list = list(range(0, length))

    # ***** ---- IMPORTANT - Remember to comment 
    # logFile.write(str(datetime.now())+ " - " +"Executing Concrete TC in random order - "+ '\n')
    # seedVal = 10
    # random.seed(a=seedVal)
    # logFile.write(str(datetime.now())+ " - " +"seed value =  - "+ str(seedVal) +  '\n')
    # numbers = range(0,length)
    # tc_num_list = random.sample(numbers, length)


    logFile.write(str(datetime.now())+ " - " +"Concrete TC execution sequence - "+ '\n')
    logFile.write("\t\t"+ str(tc_num_list)+ '\n')
    j = 1

    for i in tc_num_list:
        e1 = exp.generate_counterfactuals(df.loc[[i]], total_CFs=num_CF, desired_class="opposite", features_to_vary=restrict_attribute)
        j = j + 1 
        
        if (e1.cf_examples_list[0].final_cfs_df) is not None:
            # print("list is not empty")
            Concrete_TC_num_for_CF_found.append(i)
            count_of_concrete_TC_CF_found.append(j)
            test_instance_df = test_instance_df.append(e1.cf_examples_list[0].test_instance_df)
            cf_instance_df = cf_instance_df.append(e1.cf_examples_list[0].final_cfs_df)
            

    print("Concrete TC sequence #  (count) for which Dice generated a CF - ")
    print(count_of_concrete_TC_CF_found) 
    logFile.write(str(datetime.now())+ " - " +"Concrete TC count # for which Dice generated a CF - "+ '\n')
    logFile.write("\t\t"+ str(count_of_concrete_TC_CF_found)+ '\n')

    print("Concrete TC  # for which Dice generated a CF - ")
    print(Concrete_TC_num_for_CF_found) 
    logFile.write(str(datetime.now())+ " - " +"Concrete TC  # for which Dice generated a CF - "+ '\n')
    logFile.write("\t\t"+ str(Concrete_TC_num_for_CF_found)+ '\n')

    cf_instance_df.to_csv(CF_outputFile,index=False)
    logFile.write(str(datetime.now())+ " - " +"CSV file created with DiCE output CF"+ '\n')

    test_instance_df.to_csv(CF_origionalTC_ForCF_File,index=False)
    logFile.write(str(datetime.now())+ " - " +"CSV file created with Concrete TC for which DiCE was able to generate CF"+ '\n')

def GenerateDecisionTreeBins(dataset_filepath,system_name,attribute_name,target,file_suffix,isDepth,maxDepth):
    # print('Started generating Bins for - ',attribute_name)
    tree_lable = attribute_name + "_tree"
    data = pd.read_csv(dataset_filepath,usecols =[attribute_name,target])

    ## Drop NULL values
    data = data.fillna(method='ffill')
    
    data = data.dropna()
    #print(data.head())

    X_train, X_test, y_train, y_test = train_test_split(data[[attribute_name,target]],data[target],random_state=0)

    if(isDepth) : 
        tree_model = DecisionTreeClassifier(max_depth=maxDepth,splitter='best',random_state=0,criterion='entropy')
    else :
        tree_model = DecisionTreeClassifier(splitter='best',random_state=0,criterion='entropy')
    
    tree_model.fit(X_train[attribute_name].to_frame(), X_train[target])
    X_train[tree_lable]=tree_model.predict_proba(X_train[attribute_name].to_frame())[:,1] 
    
    df = pd.concat( [X_train.groupby([tree_lable])[attribute_name].min(), X_train.groupby([tree_lable])[attribute_name].max()], axis=1)
    df.columns.values[0] = "min"
    df.columns.values[1] = "max"
    sorted_df = df.sort_values(by=['min'], ascending=True)
    os.makedirs("Data/DecisionTreeBins/"+system_name + "_"+file_suffix+"/",exist_ok=True)
    sorted_df.to_csv("Data/DecisionTreeBins/"+system_name+"_"+file_suffix+"/"+attribute_name+'.csv')
    # print('Done')

def GetCategoricalAttributesValuesWithDT(attribute_name,target,dataset_filepath):
    
    #print(attribute_name)
    #print(target)
    tree_lable = attribute_name + "_tree"
    data = pd.read_csv(dataset_filepath,usecols =[attribute_name,target])
    
    #drop empty values
    data = data.dropna()
    
    values = array(data[attribute_name])
    #print(values)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print("integer_encoded")
    #print(integer_encoded)

    data['integer_encoded'] = integer_encoded

    #print(data.head())

    X_train, X_test, y_train, y_test = train_test_split(data[['integer_encoded',target]],data[target],random_state=0)
    #print(X_train)

    tree_model = DecisionTreeClassifier(max_depth=3,splitter='best',random_state=0,criterion='entropy')
    
    tree_model.fit(X_train['integer_encoded'].to_frame(), X_train[target])
    X_train[tree_lable]=tree_model.predict_proba(X_train['integer_encoded'].to_frame())[:,1] 

    # df = pd.concat( [X_train.groupby([tree_lable])['integer_encoded'].min(), X_train.groupby([tree_lable])['integer_encoded'].max()], axis=1)
    # df.columns.values[0] = "min"
    # df.columns.values[1] = "max"
    # sorted_df = df.sort_values(by=['min'], ascending=True)
    # print(sorted_df)

    df = pd.concat( [X_train.groupby([tree_lable])['integer_encoded'].mean().astype(int)], axis=1)
    # print(df)
    sorted_df = df.sort_values(by=[tree_lable], ascending=False)
    # print(sorted_df)

    # # invert first example
    # inverted = label_encoder.inverse_transform(sorted_df['integer_encoded'])
    # print("inverted")
    # print(inverted)

    inverted_df = pd.DataFrame(label_encoder.inverse_transform(sorted_df['integer_encoded']),sorted_df['integer_encoded'])
    # print('inverted_df')
    # print(inverted_df)

    attribute_df = sorted_df.join(inverted_df, on="integer_encoded")
    attribute_df.columns.values[1] = "values"
    # print('attribute_df.columns')
    # print(attribute_df.columns)

    # print(attribute_df.iloc[:,1])
    test_list = list(set(attribute_df.iloc[:,1]))
    # print('test_list')
    # print(test_list)
    
    attribute_string1 = ','.join(attribute_df.iloc[:,1])
    # print('attribute_string1')
    # print(attribute_string1)

    # Not used
    # attribute_string = ','.join(test_list)
    # print(attribute_string)

    return attribute_string1

def ReplaceAbstractValues_Random(inputFile,outputFile,col,val_to_replace,min_val,max_val):
    df = pd.read_csv(inputFile)
    # print(df.head())

    output= []
    random.seed(a=2)
  
    for index, row in df.iterrows():
        if row[col] == val_to_replace:
            random.seed(a=2)
          
            val = random.randint(min_val,max_val)
            df.at[index,col] = val

    df.to_csv(outputFile,index=False)

def ReplaceDatasetValuesByAbstract(dataset_filePath, decisionTree_Bin_path,outputFilePath):
    df = pd.read_csv(dataset_filePath)
    new_df = df
    col_num = len(df.columns) 
    i =0 
    for i in range(col_num - 1):
        #print("Column = ")
        #print(df.iloc[:,i])
        # print(str(df.columns[i]))
        #if(is_string_dtype(df.iloc[:,i])):
            # print("Stirng Attribute - Do nothing")
        if(is_numeric_dtype(df.iloc[:,i])):
            attributeName = str(df.columns[i])
            target = str(df.columns[col_num-1])
            decisionTree_Bins_path = decisionTree_Bin_path + attributeName + '.csv'
            #print(decisionTree_Bins_path)
            if (exists(decisionTree_Bins_path)):
                bin_df = pd.read_csv(decisionTree_Bins_path)
                for index, row in bin_df.iterrows():
                    #print('Index = ' , index)
                    #print('Min = ' , row[1])
                    #print('Max = ' , row[2])
                    new_df.loc[(new_df[attributeName] >= int(row[1])) & (new_df[attributeName] <= int(row[2])) , attributeName] = int(index)
    
    new_df.to_csv(outputFilePath,index=False)
    print('New export csv file saved successfully !')


def GeneratePerturbations_PairWise(cols,perturbFile,inputFile,outputFile):
    df = pd.read_csv(inputFile)
    df2 = df.copy()
    allPerturb_df = pd.DataFrame(columns=df.columns)    

    possiblePerturbDF = pd.read_csv(perturbFile)
    df['combined'] = df[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    possiblePerturbDF['combined'] = possiblePerturbDF[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    # print(possiblePerturbDF.head())

    for index , row in df.iterrows() :
        print('index = ',index)
        #print(df.loc[index])
        dfRowValue = row['combined']
        #print("dfRowValue")
        #print(dfRowValue)
        
        for pIndex in range(len(possiblePerturbDF)) :
            perturbRowValue = possiblePerturbDF.at[pIndex,'combined']
            #print("perturbRowValue")
            #print(perturbRowValue)
           
            if(perturbRowValue == dfRowValue):
                # print('matching at',pIndex)
                rows = df.loc[index, :]
                rows['TC'] = 'OrigionalTC' + str(index) 
                allPerturb_df = allPerturb_df.append(rows, ignore_index=True)
            else :
                # print('Not matching')
                #print(possiblePerturbDF.at[pIndex,'race'] )
                #print(df2.at[index,'race'])
                df2.at[index,'sex'] = possiblePerturbDF.at[pIndex,'sex']
                df2.at[index,'race'] = possiblePerturbDF.at[pIndex,'race']
                rows = df2.loc[index, :]
                rows['TC'] = 'PerturbTC' + str(index) 
                allPerturb_df = allPerturb_df.append(rows, ignore_index=True)

    allPerturb_df = allPerturb_df.drop('combined',axis=1)
    allPerturb_df.to_csv(outputFile,index=False)
    print("Perturbation File generated")


def GeneratePerturbationsForAttribute(attributr_name_to_perturb,valListToPerturb,inputFile,outputFile):
    df = pd.read_csv(inputFile)
    df2 = df.copy()
    allPerturb_df = pd.DataFrame(columns=df.columns)    

    for index , row in df.iterrows() :
        print('index = ',index)
        #print(df.loc[index])
        
        for val in valListToPerturb :
            print('Value to Perturb is =',val)
            print('df Column value is - ', df.at[index,attributr_name_to_perturb])
            print('df 2 Column value is - ', df2.at[index,attributr_name_to_perturb])
            df2 = df.copy()

            if(val != df.loc[index,attributr_name_to_perturb]):
                print('Not matching')
                df2.at[index,attributr_name_to_perturb] = val
                rows = df2.loc[index, :]
                rows['TC'] = 'PerturbTC' + str(index) 
                allPerturb_df = allPerturb_df.append(rows, ignore_index=True)
            else :
                print("same value")
                rows = df.loc[index, :]
                print(rows)
                rows['TC'] = 'OrigionalTC' + str(index) 
                allPerturb_df = allPerturb_df.append(rows, ignore_index=True)

    allPerturb_df.to_csv(outputFile,index=False)
    
def one_hot_encode_data(data,categorical_feature_names):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=categorical_feature_names)

def normalize_data(data_df,continuous_features,outcome_name):
        continuous_feature_names = continuous_features
        continuous_feature_indexes = [data_df.columns.get_loc(name) for name in
                                           continuous_feature_names]
        feature_names = [
            name for name in data_df.columns.tolist() if name != outcome_name]
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = data_df.copy()
        if isinstance(data_df, pd.DataFrame) or isinstance(data_df, dict):
            for feature_name in continuous_feature_names:
                max_value = data_df[feature_name].max()
                min_value = data_df[feature_name].min()
                if min_value == max_value:
                    result[feature_name] = 0
                else:
                    result[feature_name] = (data_df[feature_name] - min_value) / (max_value - min_value)
        else:
            result = result.astype('float')
            for feature_index in continuous_feature_indexes:
                feature_name = feature_names[feature_index]
                max_value = data_df[feature_name].max()
                min_value = data_df[feature_name].min()
                if len(data_df.shape) == 1:
                    if min_value == max_value:
                        value = 0
                    else:
                        value = (data_df[feature_index] - min_value) / (max_value - min_value)
                    result[feature_index] = value
                else:
                    if min_value == max_value:
                        result[:, feature_index] = np.zeros(len(data_df[:, feature_index]))
                    else:
                        result[:, feature_index] = (data_df[:, feature_index] - min_value) / (max_value - min_value)
        return result


def GetModelPredictionForDNN(concreteTCFile,columnstoUse,modelPath,targetAttribute,predColName,outputFile):
    model = keras.models.load_model(modelPath)
    origionalDF = pd.read_csv(concreteTCFile)
    
    df = pd.read_csv(concreteTCFile,usecols=columnstoUse)

    numerical_columns = []
    
    col_num = len(df.columns) 
    i =0 
    for i in range(col_num - 1):
        if(is_numeric_dtype(df.iloc[:,i])):
            numerical_columns.append(str(df.columns[i]))

    categorical = df.columns.difference(numerical_columns)

    oneHotEncoded_DF  = one_hot_encode_data(df,categorical)
    #print(oneHotEncoded_DF)

    normalizedDF = normalize_data(oneHotEncoded_DF,numerical_columns,targetAttribute)
    #print(normalizedDF)

    df = pd.DataFrame(normalizedDF.columns.values)
    df.to_csv('GetModelPredictionForDNN.csv')

    pred_df = pd.DataFrame()
    column_names = [predColName+"_ModelPred", predColName+"_PredValue"]
    pred_df = pd.DataFrame(columns = column_names)

    i =0 
    length = len(normalizedDF)
    pred_lable = 0 

    for i in range(length):
        y_pred = model.predict(normalizedDF.iloc[i:])
        if(y_pred[0] > 0.5):
            pred_lable  = 1
        else :
            pred_lable = 0

        pred_df = pred_df.append({ predColName+"_ModelPred":y_pred[0].tolist(), predColName+"_PredValue":str(pred_lable)},ignore_index=True)
    
    
    export_df = pd.DataFrame()  
    export_df = pred_df
    export_df = export_df.join(origionalDF)

    print('Export to CSV')
    export_df.to_csv(outputFile, index=False) 
    print("Done!")
