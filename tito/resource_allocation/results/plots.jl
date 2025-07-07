using JLD2,JLD,DataFrames,Plots,PyCall, StatsPlots
#Since we tested our results with 3 data models (i.e p= 0.5, 1.0 and 2.0)
#We plot just 3 results

df05 = JLD2.load("df_results_0.5.jld2","df")
df1 = JLD2.load("df_results_1.0.jld2","df")
df2 = JLD2.load("df_results_2.0.jld2","df")


dict05 = Dict(pairs(eachcol(df05)))
dict1 = Dict(pairs(eachcol(df1)))
dict2 = Dict(pairs(eachcol(df2)))



py"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


print("hola")
def plots(dict05,dict1,dict2):
    df05_pd = pd.DataFrame(dict05)
    df1_pd = pd.DataFrame(dict1)
    df2_pd = pd.DataFrame(dict2)

    df05_pd.to_csv("df05.csv" ,encoding='utf-8', index=False,header=True)
    df1_pd.to_csv("df1.csv", encoding='utf-8', index=False,header=True)
    df2_pd.to_csv("df2.csv", encoding='utf-8', index=False, header=True)
"""

py"plots"(dict05,dict1,dict2)
