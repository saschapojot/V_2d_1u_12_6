import re
from decimal import Decimal
import json
import sys
from pathlib import Path
import os
import shutil
from distutils.util import strtobool

import numpy as np

#this script initializes the parameters for mc computation by reading summary file
#and also creates/erases necessary folders

invalidValueErrCode=1
mcErrCode=2
pathErrCode=3
numArgErr=4
if (len(sys.argv)!=2):
    print("wrong number of arguments.")
    exit(numArgErr)


jsonDataFromConf=json.loads(sys.argv[1])
#read json
T=float(jsonDataFromConf["T"])
if T<=0:
    print("invalid temperature: "+str(T))
    exit(invalidValueErrCode)


erase_data_if_exist=bool(strtobool(jsonDataFromConf["erase_data_if_exist"]))
search_and_read_summary_file=bool(strtobool(jsonDataFromConf["search_and_read_summary_file"]))
potential_function_name=jsonDataFromConf["potential_function_name"]
effective_data_num_required=int(jsonDataFromConf["effective_data_num_required"])
sweep_to_write=int(jsonDataFromConf["sweep_to_write"])
default_flush_num=int(jsonDataFromConf["default_flush_num"])

confFileName=jsonDataFromConf["confFileName"]
N=int(jsonDataFromConf["unitCellNum"])

#TDirRoot contains everything for a computation
TDirRoot=os.path.dirname(confFileName)

TDirRoot=TDirRoot+"/"

#create directory for raw data of U and dist
U_dist_dataDir=TDirRoot+"/U_dist_dataFiles/"
# print(dataDir)
if erase_data_if_exist==True:
    if os.path.isdir(U_dist_dataDir):
        try:
            shutil.rmtree(U_dist_dataDir)
            print(f'Directory {U_dist_dataDir} and all its contents have been removed successfully')
        except OSError as e:
            print(f'Error: {U_dist_dataDir} : {e.strerror}')
            exit(pathErrCode)
#create dataDir if not exists
Path(U_dist_dataDir).mkdir(exist_ok=True, parents=True)


#parameters to guide mc computation
lag=-1
startingFileInd=-1
startingVecPosition=-1
newDataPointNum=-1
newMcStepNum=sweep_to_write*default_flush_num
newFlushNum=default_flush_num


def create_jsonFromSummary(startingFileIndVal,startingVecPositionVal,newMcStepNumVal,
                           newDataPointNumVal,newFlushNumVal,TDirRootStr,U_dist_dataDirStr):

    """

    :param startingFileIndVal:
    :param startingVecPositionVal:
    :param newMcStepNumVal:
    :param newDataPointNumVal:
    :param newFlushNumVal:
    :param TDirRootStr:
    :param U_dist_dataDirStr:
    :return: jsonFromSummary as string
    """
    outDict={
        "startingFileInd":str(startingFileIndVal),
        "startingVecPosition":str(startingVecPositionVal),
        "newMcStepNum":str(newMcStepNumVal),
        "newDataPointNum":str(newDataPointNumVal),
        "newFlushNum": str(newFlushNumVal),
        "TDirRoot":str(TDirRootStr),
        "U_dist_dataDir": str(U_dist_dataDirStr),
    }

    return json.dumps(outDict)



#if observable_name not found, return -1,-1, sweep_to_write*default_flush_num, then exit with code 0
if "observable_name" not in jsonDataFromConf:
    jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,startingVecPosition,newMcStepNum,
                                              newDataPointNum,newFlushNum,TDirRoot,U_dist_dataDir)
    jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
    print(jsonFromSummary_stdout)
    exit(0)

obs_name=jsonDataFromConf["observable_name"]
summaryFileName=TDirRoot+"/summary_"+obs_name+".txt"
summaryFileExists= os.path.isfile(summaryFileName)


#if summary file does not exist, return -1,-1, sweep_to_write*default_flush_num, then exit with code 0
if summaryFileExists==False:
    jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,startingVecPosition,newMcStepNum,
                                              newDataPointNum,newFlushNum,TDirRoot,U_dist_dataDir)
    jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
    print(jsonFromSummary_stdout)
    exit(0)


#parse summary file
with open(summaryFileName,"r") as fptr:
    linesInSummaryFile= fptr.readlines()


for oneLine in linesInSummaryFile:
    matchErr=re.search(r"error",oneLine)
    #if "error" is matched
    if matchErr:
        print("error in previous computation, please re-run.")
        exit(mcErrCode)

    #if "continue" is matched
    matchContinue=re.search(r"continue",oneLine)
    if matchContinue:
        jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,startingVecPosition,newMcStepNum,
                                                  newDataPointNum,newFlushNum,TDirRoot,U_dist_dataDir)
        jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
        print(jsonFromSummary_stdout)
        exit(0)
    #if "high" is matched
    matchHigh=re.search(r"high",oneLine)
    if matchHigh:
        jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,startingVecPosition,newMcStepNum,
                                                  newDataPointNum,newFlushNum,TDirRoot,U_dist_dataDir)
        jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
        print(jsonFromSummary_stdout)
        exit(0)
    #the rest of the cases is "equilibrium"
    matchEq=re.search(r"equilibrium",oneLine)
    if matchEq:
        continue
    #match lag
    matchLag=re.match(r"lag\s*=\s*(\d+)",oneLine)
    if matchLag:
        lag=int(matchLag.group(1))

    #match newDataPointNum
    matchNew=re.match(r"newDataPointNum\s*=\s*(\d+)",oneLine)
    if matchNew:
        newDataPointNum=int(matchNew.group(1))

    #match startingFileInd
    matchStartingFileInd=re.match(r"startingFileInd\s*=\s*(\d+)",oneLine)
    if matchStartingFileInd:
        startingFileInd=int(matchStartingFileInd.group(1))

    #match startingVecPosition
    matchStartingVecPosition=re.match(r"startingVecPosition=(\d+)",oneLine)
    if matchStartingVecPosition:
        startingVecPosition=int(matchStartingVecPosition.group(1))


newMcStepNum=lag*newDataPointNum
# print(newMcStepNum)
# print(sweep_to_write)
newFlushNum=int(np.ceil(newMcStepNum/(sweep_to_write)))
jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,startingVecPosition,newMcStepNum,
                                          newDataPointNum,newFlushNum,TDirRoot,U_dist_dataDir)
jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
print(jsonFromSummary_stdout)
exit(0)