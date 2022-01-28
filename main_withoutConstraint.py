import sys
import os
from datetime import datetime
import pandas as pd
import Utilities.helper as h
import configparser

def main():
    startTime = datetime.now()
    logFileName = "Logs/Log_"+str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))+".txt"
    print(logFileName)
    log = open(logFileName,"w+")

    log.write(str(datetime.now())+ " - " +"Execution Started"+ '\n')
    # Read Command line agrguments, if arguments are not provided - run program with default values
    if len(sys.argv) > 4:
        dataset_filePath,modlePath,tWay,protectdAttr,targetAttr = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4].split(","),sys.argv[5].split(",")
        log.write(str(datetime.now())+ " - " +"command line arguments"+ '\n')
    else:
        log.write(str(datetime.now())+ " - " +"default values"+ '\n')
      
        configs = configparser.ConfigParser()

        #configs.read('Config/adult_AI360_LR.cfg')

        #configs.read('Config/germanCredit_LR.cfg')
        #configs.read('Config/germanCredit_AI360_LR.cfg')

        #configs.read('Config/germanCredit_DNN.cfg')
        configs.read('Config/germanCredit_AI360_DNN.cfg')
        
        
        dataset_filePath = str(configs['values']['dataset_filePath'])
        modlePath = str(configs['values']['modlePath'])
        protectdAttr = str(configs['values']['protectdAttr']).split(",")
        targetAttr = str(configs['values']['targetAttr'])
        model_arch = str(configs['values']['model_arch'])
        isDNN = str(configs['values']['isDNN'])
        isDice = str(configs['values']['isDice'])
        perturbValueFle = str(configs['values']['perturbValueFle'])

    if(isDice == "1"):
        print('DiCE')
    else:
        print('manual')

    if(isDNN):
        print('DNN')
    else:
        print('manual')

    df = pd.read_csv(dataset_filePath)
    #target_feature = df.columns[len(df.columns) - 1]
    target_feature = targetAttr
    system_name = dataset_filePath.split("/")[len(dataset_filePath.split("/")) - 1].split(".")[0]
    #file_suffix = str(datetime.now().strftime("%m")+datetime.now().strftime("%d")+"_"+datetime.now().strftime("%H")+datetime.now().strftime("%M"))
    file_suffix = model_arch  + "_" + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    
    log.write(str(datetime.now())+ " - " +"Dataset = "+ dataset_filePath+ '\n')
    log.write(str(datetime.now())+ " - " +"Model = "+ modlePath+'\n')

    # Step 1. Create ACTS parameter file 
    log.write(str(datetime.now())+ " - " +"Create ACTS parameter file"+ '\n')
    continuous_featuresList = h.CereateACTSParamFile(dataset_filePath,system_name,target_feature,file_suffix,log)
        
    for tWayCount in [2]:
        #print (tWayCount)
        tWay = str(tWayCount)
        output = open("Output/Output_"+system_name+"_"+tWay+"Way_" +file_suffix+ ".txt","w+")
        
        log.write(str(datetime.now())+ " - " +"\n\nRunning expirements for tWay = " + str(tWay) + '\n')

        # step 2. Generate abstract test cases using ACTS 
        log.write(str(datetime.now())+ " - " +"Generate "+ str(tWay)+ "Way abstract test cases"+ '\n')
        h.Get_tWay_abstractTC(dataset_filePath,system_name,tWay,file_suffix,target_feature,log)
        
        # Step 3. Convert abstract test cases to concrete test cases 
        log.write(str(datetime.now())+ " - " +"Create concrete TC from abstract test cases"+ '\n')
        concreteTC_filepath = "Data/tWay_Concrete_TC/"+system_name+ "_" +tWay+ "way_concrete_TC_" +file_suffix+ ".csv"
        h.Get_tWay_ConcreteTC(concreteTC_filepath,system_name,tWay,file_suffix,log)

        if (isDNN != "1"):
            # Step 4. Get modle prediction (expected) for concretee test cases 
            log.write(str(datetime.now())+ " - " +"get model prediction for concrete TC "+ '\n')
            concreteTC_modelPred_outputFile = "Data/Model_Output/" +system_name+ "_" +tWay+ "way_concreteTC_ModelPred_" +file_suffix+ ".csv"
            h.GetModelPrediction(modlePath,concreteTC_filepath,concreteTC_modelPred_outputFile,'ModelPrediction_expected')
            #log.write(str(datetime.now())+ " - " +"CSV file created with model prediction for concrete TC "+ '\n')

        # Step 5. Generate counterfactual using DiCE for provided protected attributes 
        if(isDice == "1"):
            log.write(str(datetime.now())+ " - " +"generate counterfactuals using DiCE "+ '\n')
            log.write(str(datetime.now())+ " - " +str(continuous_featuresList)+ '\n')
            number_of_CF = 1    
            CF_outputFile = 'Data/Counterfactuals/' +system_name+ "_" +tWay+ "way_Generated_CF_" +file_suffix+ ".csv"
            CF_origionalTC_ForCF_File = 'Data/Counterfactuals/' +system_name+ "_" +tWay+ "way_concreteTC_ForWhichCFFound_" +file_suffix+ ".csv"
            CompareFile = 'Data/CompareFile/' +system_name+ "_" +tWay+ "way_compare_concreteTC_CF_" +file_suffix+ ".csv"

            if (isDNN == "1"):
                h.Get_DiCECounterfactuals_for_DNN(modlePath,dataset_filePath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,protectdAttr,number_of_CF,CompareFile,log)
            else : 
                h.Get_DiCECounterfactuals(modlePath,dataset_filePath,concreteTC_filepath,continuous_featuresList,target_feature,CF_outputFile,CF_origionalTC_ForCF_File,protectdAttr,number_of_CF,CompareFile,log)

                # # Step 6. Get model prediction for generated CF (verify with expected values)

                # check if CF generated or the file is empty 
                if (os.stat(CF_outputFile).st_size > 1):
                    log.write(str(datetime.now())+ " - " +"get model prediction for generated CF "+ '\n')
                    CF_modelPredFile = "Data/Model_Output/" +system_name+ "_" +tWay+ "way_CF_ModelPred_" +file_suffix+ ".csv"
                    
                    log.write(str(datetime.now())+ '\n\n\n'+ " - " +"Getting Model prediction for  "+ CF_outputFile+'\n\n\n')
                    
                    CFfound = h.GetModelPrediction(modlePath,CF_outputFile,CF_modelPredFile,'ModelPrediction_CF')
                    log.write(str(datetime.now())+ " - " +"CSV file created with model prediction for CF "+ '\n')

                    # Step 7. Verify Model Pred and DiCE CF output
                    log.write(str(datetime.now())+ " - " +"Verify Model Pred with DiCE CF output "+ '\n')
                    df = pd.read_csv(CF_modelPredFile)
                    log.write(str(datetime.now())+ " - " +" Model Pred vs DiCE CF output equals = "+ str(df['ModelPrediction_CF'].equals(df[target_feature])) + '\n')

                    endTime = datetime.now()

                    # Step 8. Output = # CF / # tWat tc
                    
                    output.write("Data set = "+ dataset_filePath + '\n')
                    output.write("Model = "+ modlePath + '\n')

                    output.write("Model Pred vs DiCE CF output equals = "+ str(df['ModelPrediction_CF'].equals(df[target_feature])) + '\n')
                    
                    df = pd.read_csv(concreteTC_modelPred_outputFile)
                    num_of_tway_TC = df.shape[0]
                    output.write("# Total " +tWay+ "-way TC generated = " + str(num_of_tway_TC)+ '\n')
                    print("# Total " +tWay+ "-way TC generated = " + str(num_of_tway_TC)+ '\n')
                    
                    df1 = pd.read_csv(CF_outputFile)
                    num_of_CF = df1.shape[0]
                    output.write("# CF generated = " + str(num_of_CF)+ '\n')
                    print("# CF generated = " + str(num_of_CF)+ '\n')
                    
                    #output.write("Ratio  (CF / Total tWay)= " + str((num_of_CF/num_of_tway_TC)*100)+ '% \n')

                    output.write("Execution Time = " + str(endTime - startTime)+ '\n')
                else:
                    output.write("Data set = "+ dataset_filePath + '\n')
                    output.write("Model = "+ modlePath + '\n')

                    df = pd.read_csv(concreteTC_modelPred_outputFile)
                    num_of_tway_TC = df.shape[0]
                    output.write("# Total " +tWay+ "-way TC generated = " + str(num_of_tway_TC)+ '\n')

                    output.write("NO Counterfactuals found !" + '\n')

        else:
            print('Run Manual expirement')
            concreteTC_filepath = "Data/tWay_Concrete_TC/"+system_name+ "_" +tWay+ "way_concrete_TC_" +file_suffix+ ".csv"
            
            df = pd.read_csv(concreteTC_filepath)
            colsToUse = df.columns

            # # get model prediction for concrete TC
            print('Getting Model Prediction for concrete test cases')
            concreteTC_modelPred_outputFile = "Data/Model_Output/" +system_name+ "_" +tWay+ "way_concreteTC_ModelPred_" +file_suffix+ ".csv"
            h.GetModelPredictionForDNN(concreteTC_filepath,colsToUse,modlePath,targetAttr,"Expected",concreteTC_modelPred_outputFile)

            # # generate perturbations 
            print('Generating perturbations')
            perturbValueFle = perturbValueFle
            inputfile_to_generate_perturb = concreteTC_modelPred_outputFile
            outputFile_with_perturbations = 'Data/Perturb_TC/'+system_name+ "_" +tWay+ "way_pairWisePerturb_" + file_suffix + "_" + model_arch+ ".csv"
            colsForPerturb = ['sex', 'age']
            h.GeneratePerturbations_PairWise(colsForPerturb,perturbValueFle,inputfile_to_generate_perturb,outputFile_with_perturbations)

            # get model prediction for perturbations TC
            print('Getting Model Prediction for perturbations')
            outputFile_with_perturbations = 'Data/Perturb_TC/GermanCredit_AI360_Modified_2way_pairWisePerturb_DNN_2022_01_24_02_04_15_DNN.csv'
            modelPred_outputFile = "Data/Model_Output/" +system_name+ "_" +tWay+ "way_Perturb_ModelPred_" +file_suffix+ ".csv"
            h.GetModelPredictionForDNN(outputFile_with_perturbations,colsToUse,modlePath,targetAttr,"Perturb",modelPred_outputFile)

        output.close()

    log.close()

main()
