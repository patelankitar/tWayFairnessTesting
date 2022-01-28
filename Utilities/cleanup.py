import glob
import os
import shutil

def MoveSpecificFilesToArchive(searchString):

    searchString = searchString
    
    print('__________ Moving Files with string '+ searchString+' to Archived Folder _______________-')

    ##-------- -------- -------- LOGS
    folder = 'LOGS'

    print('Cleaning - ',folder)

    src = "Logs/"
    dest = "Logs/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- Output 
    folder = 'Output'

    print('Cleaning - ',folder)

    src = "Output/"
    dest = "Output/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- CF Files 
    folder = 'Counterfactuals'

    print('Cleaning - ',folder)

    src = "Data/Counterfactuals/"
    dest = "Data/Counterfactuals/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- tWay_Concrete_TC Files 

    folder = 'tWay_Concrete_TC'

    print('Cleaning - ',folder)

    src = "Data/tWay_Concrete_TC/"
    dest = "Data/tWay_Concrete_TC/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- tWay_Abstract_TC Files 

    folder = 'tWay_Abstract_TC'

    print('Cleaning - ',folder)

    src = "Data/tWay_Abstract_TC/"
    dest = "Data/tWay_Abstract_TC/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- Model_Output Files 

    folder = 'Model_Output'

    print('Cleaning - ',folder)

    src = "Data/Model_Output/"
    dest = "Data/Model_Output/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- CompareFile Files 

    folder = 'CompareFile'

    print('Cleaning - ',folder)

    src = "Data/CompareFile/"
    dest = "Data/CompareFile/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- ACTS Param Files 

    folder = 'ACTS params'

    print('Cleaning - ',folder)

    src = "Data/ACTS_param/"
    dest = "Data/ACTS_param/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    print(files)

    for file in files:
        file_name = os.path.basename(file)
        if searchString in file_name:
            print(file_name)
            shutil.move(file, dest + file_name)

    print('Done - ',folder)

# ______________________________________________________________________________________________

def MoveAllFilestoArchive():

    print('__________ Moving Files to Archived Folder _______________-')

    ##-------- -------- -------- DecisionTreeBins
    folder = 'DecisionTreeBins'

    print('Cleaning - ',folder)

    # # Source path 
    src = 'Data/DecisionTreeBins/*'
    
    # # Destination path 
    destination = 'Data/DecisionTreeBins/Archieved'


    allFolders = glob.glob(src)

    # print(glob.glob(allFolders))

    for folder in allFolders:
        print(folder)
        shutil.move(folder,destination)

    print('Done - ',folder)


    ##-------- -------- -------- LOGS
    folder = 'LOGS'

    print('Cleaning - ',folder)

    src = "Logs/"
    dest = "Logs/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- Output 
    folder = 'Output'

    print('Cleaning - ',folder)

    src = "Output/"
    dest = "Output/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- CF Files 
    folder = 'Counterfactuals'

    print('Cleaning - ',folder)

    src = "Data/Counterfactuals/"
    dest = "Data/Counterfactuals/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- tWay_Concrete_TC Files 

    folder = 'tWay_Concrete_TC'

    print('Cleaning - ',folder)

    src = "Data/tWay_Concrete_TC/"
    dest = "Data/tWay_Concrete_TC/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)


    # -------- -------- -------- -------- tWay_Abstract_TC Files 

    folder = 'tWay_Abstract_TC'

    print('Cleaning - ',folder)

    src = "Data/tWay_Abstract_TC/"
    dest = "Data/tWay_Abstract_TC/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- Model_Output Files 

    folder = 'Model_Output'

    print('Cleaning - ',folder)

    src = "Data/Model_Output/"
    dest = "Data/Model_Output/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- CompareFile Files 

    folder = 'CompareFile'

    print('Cleaning - ',folder)

    src = "Data/CompareFile/"
    dest = "Data/CompareFile/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)

    # -------- -------- -------- -------- ACTS Param Files 

    folder = 'ACTS params'

    print('Cleaning - ',folder)

    src = "Data/ACTS_param/"
    dest = "Data/ACTS_param/Archieved/ "
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    #print(files)

    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, dest + file_name)

    print('Done - ',folder)


def DeleteAllFiles():

    print('__________ Moving Files to Archived Folder _______________-')

    ##-------- -------- -------- DecisionTreeBins
    folder = 'DecisionTreeBins'

    print('Cleaning - ',folder)

    # # Source path 
    src = 'Data/DecisionTreeBins/*'
   
    allFolders = glob.glob(src)

    # print(glob.glob(allFolders))

    for folder in allFolders:
        shutil.rmtree(folder)


    print('Done - ',folder)


    ##-------- -------- -------- LOGS
    folder = 'LOGS'

    print('Cleaning - ',folder)

    src = "Logs/"
    dest = "Logs/Archieved/"
    
    pattern = "*.txt"
    
    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)

    # -------- -------- -------- -------- Output 
    folder = 'Output'

    print('Cleaning - ',folder)

    src = "Output/"
    dest = "Output/Archieved/"
  
    # Search files with .txt extension in source directory
    pattern = "*.txt"

    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)


    # -------- -------- -------- -------- CF Files 
    folder = 'Counterfactuals'

    print('Cleaning - ',folder)

    src = "Data/Counterfactuals/"
    dest = "Data/Counterfactuals/Archieved/"
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    
    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)


    # -------- -------- -------- -------- tWay_Concrete_TC Files 

    folder = 'tWay_Concrete_TC'

    print('Cleaning - ',folder)

    src = "Data/tWay_Concrete_TC/"
    dest = "Data/tWay_Concrete_TC/Archieved/"
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)


    # -------- -------- -------- -------- tWay_Abstract_TC Files 

    folder = 'tWay_Abstract_TC'

    print('Cleaning - ',folder)

    src = "Data/tWay_Abstract_TC/"
    dest = "Data/tWay_Abstract_TC/Archieved/"
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)

    # -------- -------- -------- -------- Model_Output Files 

    folder = 'Model_Output'

    print('Cleaning - ',folder)

    src = "Data/Model_Output/"
    dest = "Data/Model_Output/Archieved/"
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)

    # -------- -------- -------- -------- CompareFile Files 

    folder = 'CompareFile'

    print('Cleaning - ',folder)

    src = "Data/CompareFile/"
    dest = "Data/CompareFile/Archieved/"
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.csv"
    files = glob.glob(src + pattern)
    #print(files)

    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)

    # -------- -------- -------- -------- ACTS Param Files 

    folder = 'ACTS params'

    print('Cleaning - ',folder)

    src = "Data/ACTS_param/"
    dest = "Data/ACTS_param/Archieved/"
    #files = os.listdir(path)

    # Search files with .txt extension in source directory
    pattern = "*.txt"
    files = glob.glob(src + pattern)
    #print(files)

    files = glob.glob(src + pattern)
    for file in files:
        os.remove(file)

    files = glob.glob(dest + pattern)
    for file in files:
        os.remove(file)

    print('Done - ',folder)




# MoveSpecificFilesToArchive('compas')

# MoveAllFilestoArchive()

DeleteAllFiles()

