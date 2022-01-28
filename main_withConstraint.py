import sys
import os
from datetime import datetime
import pandas as pd
from pandas.api.types import is_numeric_dtype
import Utilities.helper as h
import configparser

def main2():
    startTime = datetime.now()
    logFileName = "Logs/Log_"+str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))+".txt"
    # print(logFileName)
    log = open(logFileName,"w+")

    log.write(str(datetime.now())+ " - " +"Execution Started"+ '\n')
    # Read Command line agrguments, if arguments are not provided - run program with default values
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        #print(choice)
        log.write(str(datetime.now())+ " - " +"default values"+ '\n')
      
        configs = configparser.ConfigParser()

        # --------- -------- --------- Adult Income Dataset - AI360
        # configs.read('Config/adult_AI360_LR.cfg')
        # configs.read('Config/adult_AI360_RF.cfg')
        # configs.read('Config/adult_AI360_SVM.cfg')
        # configs.read('Config/adult_AI360_DNN.cfg')

        # --------- -------- --------- German Credit Dataset - AI360
        #configs.read('Config/germanCredit_AI360_LR.cfg')
        #configs.read('Config/germanCredit_AI360_RF.cfg')
        #configs.read('Config/germanCredit_AI360_SVM.cfg')
        #configs.read('Config/germanCredit_AI360_DNN.cfg')

        # --------- -------- --------- COMPOS Dataset - AI360
        # configs.read('Config/compas_AI360_LR.cfg')
        # configs.read('Config/compas_AI360_RF.cfg')
        configs.read('Config/compas_AI360_SVM.cfg')
        # configs.read('Config/compas_AI360_DNN.cfg')
        
        dataset_filePath = str(configs['values']['dataset_filePath'])
        modlePath = str(configs['values']['modlePath'])
        protectdAttr = str(configs['values']['protectdAttr']).split(",")
        targetAttr = str(configs['values']['targetAttr'])
        model_arch = str(configs['values']['model_arch'])
        isDNN = str(configs['values']['isDNN'])
        isDice = str(configs['values']['isDice'])
        perturbValueFle = str(configs['values']['perturbValueFle'])
        
        # dataSet = "compas"
        # system="AI360"

        # dataSet = "adult_data"
        # system="AI360"

        #dataSet = "GermanCredit"
        #system="AI360"

        df = pd.read_csv(dataset_filePath)

        target_feature = targetAttr
        system_name = dataset_filePath.split("/")[len(dataset_filePath.split("/")) - 1].split(".")[0]
        tWay = '2'

        file_suffix = "with_constraint"

        #file_suffix = "with_constraint"

        if choice == "1":
            print('Generate ACTS Param File') 
            
            log.write(str(datetime.now())+ " - " +"Dataset = "+ dataset_filePath+ '\n')
            log.write(str(datetime.now())+ " - " +"Model = "+ modlePath+'\n')

            # Step 1. Create ACTS parameter file 
            log.write(str(datetime.now())+ " - " +"Create ACTS parameter file"+ '\n')
            continuous_featuresList = h.CereateACTSParamFile(dataset_filePath,system_name,target_feature,file_suffix,log)

            log.write(str(datetime.now())+ " - " +"continuous_featuresList = \n "+ str(continuous_featuresList) +'\n')

        elif choice == "2":
            log.write(str(datetime.now())+ " - " + 'Convert Dataset to Abstract vlaues')
            print("Convert Dataset to Abstract vlaues")

            # log.write("dataSet = "+ dataSet+'\n')
            # log.write("system = "+ system+'\n')

            #dataset_filePath = 'Dataset/' + dataSet +'_' + system+ '_Modified.csv'
            decisionTree_Bin_path= 'Data/DecisionTreeBins/' + system_name + '_Modified_' + file_suffix + '/'
            outputFilePath = 'Data/Abstract_Dataset/' + system_name + '_With_abstract_values.csv'

            h.ReplaceDatasetValuesByAbstract(dataset_filePath, decisionTree_Bin_path,outputFilePath)
            log.write(str(datetime.now())+ " - " + 'Dataset replaced with Abstract vlaues saved at ' + outputFilePath)
            print("Dataset replaced with Abstract vlaues csv export done !")

        elif choice == "3":
            print('Generate t-Way Abstract and Concrete TC')
            
            abstractFileCreated = 0 
            # step 1. Generate abstract test cases using ACTS 
            log.write(str(datetime.now())+ " - " +"Generate "+ str(tWay)+ "Way abstract test cases"+ '\n')
            abstractFileCreated = h.Get_tWay_abstractTC(dataset_filePath,system_name,tWay,file_suffix,target_feature,log)
            print(abstractFileCreated)

            # Step 2. Convert abstract test cases to concrete test cases 
            log.write(str(datetime.now())+ " - " +"Create concrete TC from abstract test cases"+ '\n')
            concreteTC_filepath = "Data/tWay_Concrete_TC/"+system_name+ "_" +tWay+ "way_concrete_TC_" +file_suffix+ ".csv"
            h.Get_tWay_ConcreteTC(concreteTC_filepath,system_name,tWay,file_suffix,log)

        elif choice == "4":
            #file_suffix = "with_constraint_" 
            if (isDice == "1") : 
                print('Get CF from DiCE')
                log.write(str(datetime.now())+ " - " +"generate counterfactuals using DiCE "+ '\n')
                concreteTC_filepath = "Data/tWay_Concrete_TC/"+system_name+ "_" +tWay+ "way_concrete_TC_" +file_suffix+ ".csv"

                continuous_featuresList= []

                col_num = len(df.columns) 
                i =0 
                for i in range(col_num - 1):
                    if(is_numeric_dtype(df.iloc[:,i])):
                        continuous_featuresList.append(str(df.columns[i]))

                number_of_CF = 1    
                CF_outputFile = 'Data/Counterfactuals/' +system_name+ "_" +tWay+ "way_Generated_CF_" + file_suffix + "_" + model_arch+ ".csv"
                CF_origionalTC_ForCF_File = 'Data/Counterfactuals/' +system_name+ "_" +tWay+ "way_concreteTC_ForWhichCFFound_" + file_suffix  + "_" + model_arch+ ".csv"
                CompareFile = 'Data/CompareFile/' +system_name+ "_" +tWay+ "way_compare_concreteTC_CF_" + file_suffix  + "_" + model_arch+ ".csv"
                #h.Get_DiCECounterfactuals(modlePath,dataset_filePath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,protectdAttr,number_of_CF,CompareFile,log)
                if (isDNN == "1"):
                    h.Get_DiCECounterfactuals_for_DNN(modlePath,dataset_filePath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,protectdAttr,number_of_CF,CompareFile,log)
                else : 
                    h.Get_DiCECounterfactuals(modlePath,dataset_filePath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,protectdAttr,number_of_CF,CompareFile,log)


                cTC_df = pd.read_csv(concreteTC_filepath)
                num_of_tway_TC = cTC_df.shape[0]
                print("# Total " +tWay+ "-way TC generated = " + str(num_of_tway_TC)+ '\n')
                
                CF_df = pd.read_csv(CF_outputFile)
                num_of_CF = CF_df.shape[0]
                print("# CF generated = " + str(num_of_CF)+ '\n')
            else:
                print('Run Manual expirement')
                concreteTC_filepath = "Data/tWay_Concrete_TC/"+system_name+ "_" +tWay+ "way_concrete_TC_" +file_suffix+ ".csv"
                
                df = pd.read_csv(concreteTC_filepath)
                colsToUse = df.columns

                # # get model prediction for concrete TC
                print('Getting Model Prediction for concrete test cases')
                concreteTC_modelPred_outputFile = "Data/Model_Output/" +system_name+ "_" +tWay+ "way_concreteTC_ModelPred_" + file_suffix  + "_" + model_arch+  ".csv"
                h.GetModelPredictionForDNN(concreteTC_filepath,colsToUse,modlePath,targetAttr,"Expected",concreteTC_modelPred_outputFile)

                # # generate perturbations 
                print('Generating perturbations')
                perturbValueFle = perturbValueFle
                inputfile_to_generate_perturb = concreteTC_modelPred_outputFile
                outputFile_with_perturbations = 'Data/Perturb_TC/'+system_name+ "_" +tWay+ "way_pairWisePerturb_" + file_suffix + "_" + model_arch+ ".csv"
                # colsForPerturb = ['sex', 'age']
                colsForPerturb = ['sex', 'race']
                h.GeneratePerturbations_PairWise(colsForPerturb,perturbValueFle,inputfile_to_generate_perturb,outputFile_with_perturbations)

                # get model prediction for perturbations TC
                print('Getting Model Prediction for perturbations')
                modelPred_outputFile = "Data/Model_Output/" +system_name+ "_" +tWay+ "way_Perturb_ModelPred_" + file_suffix  + "_" + model_arch+  ".csv"
                h.GetModelPredictionForDNN(outputFile_with_perturbations,colsToUse,modlePath,targetAttr,"Perturb",modelPred_outputFile)

        else: 
            print('Invalid Choice entered')
            print('\n Please select : \n 1. To Generate ACTS Param file \n 2.Convert Dataset to Abstract vlaues \n 3. To generate abstract and concrete TC \n 4. To get CF')
        
    else:
        print('Please provide command line Arg : \n 1. To Generate ACTS Param file \n 2.Convert Dataset to Abstract vlaues \n 3. To generate abstract and concrete TC \n 4. To get CF')

    log.close()

main2()
