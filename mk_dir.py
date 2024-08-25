from pathlib import Path
from decimal import Decimal

import numpy as np
import pandas as pd


#This script creates directories and conf files for mc



rowNum=0
inParamFileName="./V_inv_12_6Params.csv"



inDf=pd.read_csv(inParamFileName)
oneRow=inDf.iloc[rowNum,:]
a1=float(oneRow.loc["a1"])
b1=float(oneRow.loc["b1"])
# a2=float(oneRow.loc["a2"])
# b2=float(oneRow.loc["b2"])



def format_using_decimal(value):
    # Convert the float to a Decimal
    value=np.round(value,4)
    decimal_value = Decimal(value)
    # Remove trailing zeros and ensure fixed-point notation
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)
TVals=[1,2,3]
unitCellNum=1
dataRoot="./dataAll/"
dataOutDir=dataRoot+"/dataAllUnitCell"+str(unitCellNum)+"/row"+str(rowNum)+"/"

TDirsAll=[]
TStrAll=[]
# print(TDirsAll)
for k in range(0,len(TVals)):
    T=TVals[k]
    # print(T)

    TStr=str(T)#format_using_decimal(T)
    TStrAll.append(TStr)
    TDir=dataOutDir+"/T"+TStr+"/"
    TDirsAll.append(TDir)
    Path(TDir).mkdir(exist_ok=True,parents=True)



def contents_to_conf(k):
    """

    :param k: index of T
    :return:
    """

    contents=[
        "#This is the configuration file for mc computations\n",
        "\n"
        "potential_function_name=V_inv_12_6\n",
        "\n" ,
        "#parameters of coefficients\n",
        "#parameter_row=row0\n",
        "#the following row is the values of a1, b1\n"
        "coefs=["+format_using_decimal(a1)+","+format_using_decimal(b1)+"]\n",
        "\n",
        "#Temperature\n",
        "T="+TStrAll[k]+"\n",
        "\n",
        "#unit cell number\n",
        "unitCellNum="+str(unitCellNum)+"\n",
        "\n",
        "erase_data_if_exist=False\n",
        "\n",
        "search_and_read_summary_file=True\n"
        "\n",
        "#For the observable name, only digits 0-9, letters a-zA-Z, underscore _ are allowed\n",
        "\n",
        "observable_name=U_dist\n",
        "\n",
        "effective_data_num_required=1000\n",
        "\n",
        "sweep_to_write=100000\n",
        "\n",
        "#within each flush,  sweep_to_write mc computations are executed\n",
        "\n",
        "default_flush_num=15\n",
        "\n",
        "h=5e-2\n"



    ]

    outConfName=TDirsAll[k]+"/run_T"+TStrAll[k]+".mc.conf"
    with open(outConfName,"w+") as fptr:
        fptr.writelines(contents)



for k in range(0,len(TDirsAll)):
    contents_to_conf(k)