import numpy as np
from datetime import datetime
import sys
import re
import glob
import os
import json
from pathlib import Path
import pandas as pd
import pickle
#this script extracts effective data from pkl files

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    exit()


rowNum=0
unitCellNum=int(sys.argv[1])
rowDirRoot="../dataAll/dataAllUnitCell"+str(unitCellNum)+"/row"+str(rowNum)+"/"
obs_U_dist="U_dist"


#search directory
TVals=[]
TFileNames=[]
TStrings=[]
for TFile in glob.glob(rowDirRoot+"/T*"):
    # print(TFile)
    matchT=re.search(r"T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)",TFile)
    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))
        TStrings.append("T"+matchT.group(1))


#sort T values
sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]
sortedTStrings=[TStrings[ind] for ind in sortedInds]


def parseSummary(oneTFolder,obs_name):

    startingFileInd=-1
    startingVecPosition=-1
    lag=-1
    smrFile=oneTFolder+"/summary_"+obs_name+".txt"
    summaryFileExists=os.path.isfile(smrFile)
    if summaryFileExists==False:
        return startingFileInd,startingVecPosition,-1

    with open(smrFile,"r") as fptr:
        lines=fptr.readlines()
    for oneLine in lines:
        #match startingFileInd
        matchStartingFileInd=re.search(r"startingFileInd=(\d+)",oneLine)
        if matchStartingFileInd:
            startingFileInd=int(matchStartingFileInd.group(1))
        #match startingVecPosition
        matchStartingVecPosition=re.search(r"startingVecPosition=(\d+)",oneLine)
        if matchStartingVecPosition:
            startingVecPosition=int(matchStartingVecPosition.group(1))

        #match lag
        matchLag=re.search(r"lag=(\d+)",oneLine)
        if matchLag:
            lag=int(matchLag.group(1))
    return startingFileInd, startingVecPosition,lag



def sort_data_files_by_swEnd(oneTFolder,obs_name,varName):
    """

    :param oneTFolder: Txxx
    :param obs_name: data files sorted by sweepEnd
    :return:
    """

    dataFolderName=oneTFolder+"/"+obs_name+"_dataFiles/"+varName+"/"
    dataFilesAll=[]
    sweepEndAll=[]

    for oneDataFile in glob.glob(dataFolderName+"/*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"sweepEnd(\d+)",oneDataFile)
        if matchEnd:
            sweepEndAll.append(int(matchEnd.group(1)))


    endInds=np.argsort(sweepEndAll)
    # sweepStartSorted=[sweepStartAll[i] for i in startInds]
    sortedDataFiles=[dataFilesAll[i] for i in endInds]

    return sortedDataFiles


def U_dist_data2csvForOneT(oneTFolder,oneTStr,startingFileInd,startingVecPosition,lag):
    TRoot=oneTFolder
    sortedUDataFilesToRead=sort_data_files_by_swEnd(TRoot,obs_U_dist,"U")
    # print(sortedUDataFilesToRead)
    startingUFileName=sortedUDataFilesToRead[startingFileInd]

    with open(startingUFileName,"rb") as fptr:
        inUStart=pickle.load(fptr)

    UVec=inUStart[startingVecPosition:]
    for pkl_file in sortedUDataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            # print(pkl_file)
            in_UArr=pickle.load(fptr)
            UVec=np.append(UVec,in_UArr)

    UVecSelected=UVec[::lag]

    dataArraySelected=UVecSelected

    #lattice array
    sorted_latticeFilesToRead=sort_data_files_by_swEnd(TRoot,obs_U_dist,"converted_data")

    startingLatticeFileName=sorted_latticeFilesToRead[startingFileInd]


    with open(startingLatticeFileName,"rb") as fptr:
        in_lattice_start=pickle.load(fptr)

    lattice_arr=in_lattice_start[startingVecPosition:,:]

    for pkl_file in sorted_latticeFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            in_lat_arr=pickle.load(fptr)
        lattice_arr=np.r_[lattice_arr,in_lat_arr]

    lattice_selected=lattice_arr[::lag,:]


    dataArraySelected=np.c_[dataArraySelected,lattice_selected]

    colNamesAll=["U","x00","y00","x01","y01","x10","y10","x11","y11"]
    outCsvDataRoot=rowDirRoot+"/csvOutAll/"
    outCsvFolder=outCsvDataRoot+"/"+oneTStr+"/"+obs_U_dist+"/"
    Path(outCsvFolder).mkdir(parents=True, exist_ok=True)
    outCsvFile=outCsvFolder+"/"+obs_U_dist+"Data.csv"
    dfToSave=pd.DataFrame(dataArraySelected,columns=colNamesAll)
    dfToSave.to_csv(outCsvFile,index=False)


for k in range(0,len(sortedTFiles)):
    tStart=datetime.now()
    oneTFolder=sortedTFiles[k]
    oneTStr=sortedTStrings[k]

    startingfileIndTmp,startingVecIndTmp,lagTmp=parseSummary(oneTFolder,obs_U_dist)
    if startingfileIndTmp<0:
        print("summary file does not exist for "+oneTStr+" "+obs_U_dist)
        continue

    U_dist_data2csvForOneT(oneTFolder,oneTStr,startingfileIndTmp,startingVecIndTmp,lagTmp)
    tEnd=datetime.now()
    print("processed T="+str(sortedTVals[k])+": ",tEnd-tStart)


