#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import csv
import numpy as np
import datetime
import math
from numpy import linalg as LA
from scipy import optimize
import matplotlib.pyplot as plt 
import argparse
import sys


# # Question 1 loss function

# In[39]:


def fn(b):
    global df_filtered
    global zo
    zo_temp=zo
    temp=df_filtered.copy()
    temp["exp"]=1-np.exp(-1*b*temp["OversRemaining"])
    temp["loss"]=(temp["Runs.Remaining"]-(zo_temp*temp["exp"]))**2
    loss=temp["loss"].sum()    
    del temp
    return loss


# # b(w) using gradient descent algorithm

# In[40]:


"""
def grad(df_filtered,ZO,b):
    temp=df_filtered.copy()
    temp["exp"]=np.exp(-1*b*temp["OversRemaining"])
    temp["loss"]=2*(temp["Runs.Remaining"]-(ZO*(1-temp["exp"])))*ZO*temp["OversRemaining"]*(temp["exp"])
    loss=(-1*temp["loss"]).sum()    
    return loss
"""


# # Calculating Z0(w)

# In[41]:


def func_zo_w(w):
    global df_in1_all
    if w==10:
        temp=df_in1_all.copy()
        zo_w=np.mean(temp.groupby(['Match'])['Innings.Total.Runs'].max())
        del temp
    else:
        temp=df_in1_all[df_in1_all["Wickets.in.Hand"]<=w].copy()
        zo_w=np.mean(temp.groupby(['Match'])['Runs.Remaining'].max())
        del temp
    return zo_w


# # Question 2 loss function

# In[42]:


def fn_ques2(L):
    global df_ques2
    global Z0
    loss=0
    temp=df_ques2.copy()
    temp2=temp[temp["Wickets.in.Hand"]!=0].reset_index()
    temp2["exp"]=1-np.exp((-1*L*temp2["OversRemaining"])/temp2["Z_values"])
    temp2["loss"]=(temp2["Runs.Remaining"]-(temp2["Z_values"]*temp2["exp"]))**2
    loss=temp2["loss"].sum()   
    #print(loss)
    del temp2
    del temp
    return loss


# # Plot function

# In[43]:


def plot(y_per):
    # defining the x-axis
    x=np.arange(0,51,1)    
    plt.figure()
    color_list = ['g','r', 'b', 'y', 'c', 'm', 'k', '#234a21', '#876e34', '#a21342','mediumseagreen','g']
    for i in range(1,12,1):
        plt.plot(x, y_per[i], c=color_list[i],label='Z['+str(i)+']')
        plt.legend()
    plt.xlabel("Overs Remaining")
    plt.ylabel("Percentage of resources available")
    plt.show()


# # Common Preprocessing

# In[44]:


def preprocessing():
    global df_in1_all
    #Filtering out 1st Innings data for first question
    df_in1_all=df_in1_all[df_in1_all["Innings"]==1].copy()    
    #Adding 50 overs remaining in the dataset
    df_in1_all["OversRemaining"]=df_in1_all["Total.Overs"]-df_in1_all["Over"]    
    # For 0-49 Overs remaining
    temp0_49=df_in1_all[["Match","Runs.Remaining","Innings.Total.Runs","OversRemaining","Wickets.in.Hand"]].copy()    
    #For 50 overs remaining
    df_in1_all["Innings.Total.Runs.Copy"]=df_in1_all["Innings.Total.Runs"]
    temp50=df_in1_all.groupby(["Match"])["Innings.Total.Runs.Copy","Innings.Total.Runs","OversRemaining"].max().reset_index()
    temp50["OversRemaining"]=50
    temp50["Wickets.in.Hand"]=10
    temp50.columns=["Match","Runs.Remaining","Innings.Total.Runs","OversRemaining","Wickets.in.Hand"]

    #Final innings1 dataframe containing all data
    df_in1_all=temp0_49.append(temp50).reset_index()  
    del temp0_49
    del temp50
    df_in1_all=df_in1_all[['Match', 'Runs.Remaining', 'Innings.Total.Runs',
       'OversRemaining', 'Wickets.in.Hand']]



# # Question 1 - DuckworthLewis20Params

# In[67]:


def DuckworthLewis20Params(file:str):
    global df_in1_all
    global df_filtered
    global zo    
    input_df=pd.read_csv(file)
    df_in1_all=input_df.copy()
    # Calling basic preprocessing function
    preprocessing()
    
    #Saving Z0_W values in a list, zo_w[0] is when w(wickets in hand)=0 and zo_w[10] is when w=10 and so on
    zo_w=list()
    for w in range(0,11):
        zo_w.append(func_zo_w(w))
        
        
    # Using Gradient descent approach instead of library.
    # Commenting this code out because this takes a lot of time
    """
    b_w=[]
    for w in range(0,11,1):
        # Initial Parameters defined
        leaning_rate=0.00000000001
        b=0.035
        ZO=zo_w[w]
        # To filter out dataframe by wickets remaining in hand
        df_filtered=df_in1_all[df_in1_all["Wickets.in.Hand"]==w].copy()
        # Applying gradient descnet
        while(1):   
            grad_loss=grad(df_filtered,ZO,b)
            #print("Gradient-",grad_loss,b)
            b=b-leaning_rate*grad_loss
            if LA.norm(grad_loss)<0.001:
                break
            else:
                continue 
        print("For w-",w,"optimal b(w)-",b)
        b_w.append(b)
    """     
        
    # Using scipy optimize function to calculate optimal b(w)'s
    b_w_scipy=[]
    for w in range(0,11,1):    
        zo=zo_w[w]
        df_filtered=df_in1_all[df_in1_all["Wickets.in.Hand"]==w].copy()
        # Initial Parameters defined
        b=0.035
        result=optimize.minimize(fn,b,method='L-BFGS-B')
        b_w_scipy=b_w_scipy+(result.x.tolist())

    # Plotting
    y=[]
    for i in range(0,11,1):
        zo_ques1=zo_w[i]
        b=b_w_scipy[i]
        y_output=[]
        for u in range(0,51):
            y_output.append(zo_ques1*(1-math.exp(-1*b*u)))
        y.append(y_output)    
    #Converting in Percentage of resources available format
    y_per=((y/y[10][50])*100).tolist()
    st_line = np.array([(i/50)*100 for i in np.arange(0,51,1)]).tolist()
    y_per.append(st_line)
    plot(y_per)
    
    
    Z0=zo_w
    #Calculating MSE for question 1
    loss_mse=0
    for w in range(1,11,1):    
        zo=Z0[w]
        df_filtered=df_in1_all[df_in1_all["Wickets.in.Hand"]==w].copy()
        loss_mse=loss_mse+fn(b_w_scipy[w])

    loss_mse=loss_mse/len(df_in1_all)
    print("Loss MSE Question 1=",loss_mse)

    # Checking slopes for all w's at u=0 => Z0*b
    slopes_ques1=[]
    for w in range(1,11,1):
        slopes_ques1.append(zo_w[w]*b_w_scipy[w])
    print("\nslope question 1 at u=0=",slopes_ques1)
    print("\nZ0(w)=",zo_w)
    print("\nb(w)=",b_w_scipy)
      
      
    return zo_w,b_w_scipy


# # Question 2 - DuckworthLewis11Params

# In[68]:


def DuckworthLewis11Params(file:str): 
    global df_in1_all
    global df_ques2
    global zo
    
    input_df_2=pd.read_csv(file)
    df_in1_all=input_df_2.copy()    
    
    # Calling basic preprocessing function
    preprocessing()

    #Saving Z0_W values in a list, zo_w[0] is when w(wickets in hand)=0 and zo_w[10] is when w=10 and so on
    zo_w=list()
    for w in range(0,11):
        zo_w.append(func_zo_w(w))

    Z0=zo_w
    #File for question 2
    df_ques2=df_in1_all.copy()
    
    #Extra Preprocessing for question 2
    df_ques2["Z_values"]=0
    for i in range(0,len(df_ques2)):
        if df_ques2["Wickets.in.Hand"][i] == 0 :
            df_ques2["Z_values"][i]=Z0[0]
        elif df_ques2["Wickets.in.Hand"][i] == 1 :
            df_ques2["Z_values"][i]=Z0[1]
        elif df_ques2["Wickets.in.Hand"][i] == 2 :
            df_ques2["Z_values"][i]=Z0[2]
        elif df_ques2["Wickets.in.Hand"][i] == 3 :
            df_ques2["Z_values"][i]=Z0[3]
        elif df_ques2["Wickets.in.Hand"][i] == 4 :
            df_ques2["Z_values"][i]=Z0[4]
        elif df_ques2["Wickets.in.Hand"][i] == 5 :
            df_ques2["Z_values"][i]=Z0[5]
        elif df_ques2["Wickets.in.Hand"][i] == 6 :
            df_ques2["Z_values"][i]=Z0[6]
        elif df_ques2["Wickets.in.Hand"][i] == 7 :
            df_ques2["Z_values"][i]=Z0[7]
        elif df_ques2["Wickets.in.Hand"][i] == 8 :
            df_ques2["Z_values"][i]=Z0[8]
        elif df_ques2["Wickets.in.Hand"][i] == 9 :
            df_ques2["Z_values"][i]=Z0[9]
        else:
            df_ques2["Z_values"][i]=Z0[10]

    Z0_2=Z0
    # Initial Parameters defined
    L=0.035
    # Finding optimal L values
    result=optimize.minimize(fn_ques2,L,method='L-BFGS-B')
    L=(result.x)
    #Plotting
    y=[]
    L=float(L)
    for i in range(0,11,1):
        z_ques2=Z0[i]
        y_output=[]
        for u in range(0,51):
            y_output.append(z_ques2*(1-math.exp((-1*L*u)/z_ques2)))
        y.append(y_output)

    #Converting in Percentage of resources available format
    y_per=((y/y[10][50])*100).tolist()
    st_line = np.array([(i/50)*100 for i in np.arange(0,51,1)]).tolist()
    y_per.append(st_line)
    plot(y_per) 
    
    # Calculating MSE for question 2
    loss_mse_q2=fn_ques2(L)
    loss_mse_q2=loss_mse_q2/len(df_ques2)
    print("Loss MSE Question 1=",loss_mse_q2)

    # Checking slopes for all w's at u=0 => L
    slopes_ques2=[]
    for w in range(1,11,1):
        slopes_ques2.append(L)
    print("\nslope question 2 at u=0=",slopes_ques2)
    print("\nZ0(w)=",Z0_2)
    print("\nL=",L)
    
    
    return Z0_2,L 


# # Main Function

# In[69]:


# Declaring some Global variables
df_filtered=pd.DataFrame()
df_ques2=pd.DataFrame()
df_in1_all=pd.DataFrame()
zo=0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




