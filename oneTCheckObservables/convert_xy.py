import pickle
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
import statsmodels.api as sm
import sys
import re
import warnings
from scipy.stats import ks_2samp
import glob
from pathlib import Path
import os
import json
from decimal import Decimal
import matplotlib.pyplot as plt

#this script converts x, y values to dist and eliminate translation and rotation

def format_using_decimal(value):
    # Convert the float to a Decimal using string conversion to avoid precision issues
    decimal_value = Decimal(str(value))
    # Remove trailing zeros and ensure fixed-point notation
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

argErrCode=2
missingErrCode=4

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    exit(argErrCode)

T=float(sys.argv[1])
TStr=format_using_decimal(T)

rowNum=0
dataFileRoot="./dataAll/dataAllUnitCell1/row"+str(rowNum)+"/T"+TStr+"/U_dist_dataFiles/"

folder_x00=dataFileRoot+"/x00/"
folder_x01=dataFileRoot+"/x01/"
folder_x10=dataFileRoot+"/x10/"
folder_x11=dataFileRoot+"/x11/"

folder_y00=dataFileRoot+"/y00/"
folder_y01=dataFileRoot+"/y01/"
folder_y10=dataFileRoot+"/y10/"
folder_y11=dataFileRoot+"/y11/"

folderVec_x=[folder_x00,folder_x01,folder_x10,folder_x11]
folderVec_y=[folder_y00,folder_y01,folder_y10,folder_y11]
def sort_data_files_by_sweepEnd(oneDir):
    dataFilesAll=[]
    sweepEndAll=[]
    # print("entering sort")
    for oneDataFile in glob.glob(oneDir+"/*.pkl"):
        # print(oneDataFile)
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"sweepEnd(\d+)",oneDataFile)
        if matchEnd:
            indTmp=int(matchEnd.group(1))
            sweepEndAll.append(indTmp)
    endInds=np.argsort(sweepEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]
    return sortedDataFiles


def rotationMatrix(x00,y00,x01,y01):
    """

    :param x00:
    :param y00:
    :param x01:
    :param y01:
    :return:
    """

    r=np.sqrt((x01-x00)**2+(y01-y00)**2)

    sin_theta=(y01-y00)/r
    cos_theta=(x01-x00)/r

    R=np.array([
        [cos_theta,sin_theta],
        [-sin_theta,cos_theta]
    ],dtype=float)

    return R


def convert_row(arr_row):
    """

    :param arr_row: [x00, y00, x01, y01, x10, y10, x11, y11]
    :return:
    """
    x00, y00, x01, y01, x10, y10, x11, y11=arr_row

    R=rotationMatrix(x00,y00,x01,y01)

    A01Tilde=np.array([x00,y00])+R@np.array([x01-x00,y01-y00])



    A10Tilde=np.array([x00,y00])+R@np.array([x10-x00,y10-y00])

    A11Tilde=np.array([x00,y00])+R@np.array([x11-x00,y11-y00])

    arr_row_new=np.array([x00,y00,A01Tilde[0],A01Tilde[1],A10Tilde[0],A10Tilde[1],A11Tilde[0],A11Tilde[1]])

    return arr_row_new

#x00,x01,x10,x11
filesVec_x=[sort_data_files_by_sweepEnd(fld) for fld in folderVec_x]


#y00,y01,y10,y11
filesVec_y=[sort_data_files_by_sweepEnd(fld) for fld in folderVec_y]


lengths_x=[len(item) for item in filesVec_x]

lengths_y=[len(item) for item in filesVec_y]

lengthsAll=np.append(lengths_x,lengths_y)

all_equal = np.all(lengthsAll == lengthsAll[0])

if all_equal==False:
    print("data missing.")
    exit(missingErrCode)

def oneArray(n):
    """

    :param n: ind of files
    :return:
    """
    x00File=filesVec_x[0][n]
    x01File=filesVec_x[1][n]
    x10File=filesVec_x[2][n]
    x11File=filesVec_x[3][n]

    y00File=filesVec_y[0][n]
    y01File=filesVec_y[1][n]
    y10File=filesVec_y[2][n]
    y11File=filesVec_y[3][n]

    with open(x00File,"rb") as fptr:
        x00_arr=pickle.load(fptr)

    with open(x01File,"rb") as fptr:
        x01_arr=pickle.load(fptr)

    with open(x10File,"rb") as fptr:
        x10_arr=pickle.load(fptr)

    with open(x11File,"rb") as fptr:
        x11_arr=pickle.load(fptr)

    with open(y00File,"rb") as fptr:
        y00_arr=pickle.load(fptr)


    with open(y01File,"rb") as fptr:
        y01_arr=pickle.load(fptr)

    with open(y10File,"rb") as fptr:
        y10_arr=pickle.load(fptr)

    with open(y11File,"rb") as fptr:
        y11_arr=pickle.load(fptr)

    arrCombined=np.array([
        x00_arr,y00_arr,x01_arr,y01_arr,x10_arr,y10_arr,x11_arr,y11_arr
    ]).T

    return arrCombined
tConvertStart=datetime.now()
converted_dataDir=dataFileRoot+"/converted_data/"
Path(converted_dataDir).mkdir(parents=True,exist_ok=True)

for n in range(0,lengths_x[0]):
    arrTmp=oneArray(n)

    outFile=converted_dataDir+"/latticeFile"+str(n)+".pkl"
    with open(outFile,"bw+") as fptr:
        pickle.dump(arrTmp,fptr)
tConvertEnd=datetime.now()

print("converting time: ",tConvertEnd-tConvertStart)