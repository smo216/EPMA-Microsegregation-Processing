# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:04:09 2019

@author: EMG
EPMA Microsegregation Analysis
"""

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# curve-fit() function imported from scipy 
from scipy.optimize import curve_fit 
import statsmodels.api as sm
from matplotlib.pyplot import figure
import math
# %% Import EPMA data
""" Paste Filename Here"""
filename='10_1_19 DataBase W Mo Ti disabled new filament _Un   29  Duraloy M10 CCnew Mn.xlsx'
#filename='10_1_19 DataBase W Mo Ti disabled new filament _Un   33  Duraloy M10 DTA.xlsx'
#filename='9_27_19 DataBase W Ti disabled_Un   25  MT418 DTA Grid.xlsx'
#filename='9_24_19 DataBase W Ti disabled_Un   20  418 DTA Linetrace Core.xlsx'
#filename='10_1_19 DataBase W Mo Ti disabled new filament _Un   32  HP-2 DTA.xlsx'
data = pd.read_excel(filename)
data.head
[Len,Wid]=data.shape
data.info
data.describe()


# %% Pull EPMA data
Total=data['Elemental Totals']
Si=data['Si Elemental Percents']
Cr=data["Cr Elemental Percents"]
Fe=data["Fe Elemental Percents"]
Ni=data["Ni Elemental Percents"]
W=data["W  Elemental Percents"]
Nb=data["Nb Elemental Percents"]
Mo=data["Mo Elemental Percents"]
Mn=data["Mn Elemental Percents"]
Ti=data["Ti Elemental Percents"]
#data['Total'] = data.sum(axis=1)-distance
#Total=data.sum(axis=1)-distance #totals the composition

# errprs
Si_er=data['Si Percent Errors']
Cr_er=data["Cr Percent Errors"]
Fe_er=data["Fe Percent Errors"]
Ni_er=data["Ni Percent Errors"]
W_er=data["W  Percent Errors"]
Nb_er=data["Nb Percent Errors"]
Mo_er=data["Mo Percent Errors"]
Mn_er=data["Mn Percent Errors"]
Ti_er=data["Ti Percent Errors"]

# %%Lets get Plotting
# make subplots?
plt.scatter(Fe,Si,label="Si")
plt.scatter(Fe,Cr,label="Cr")
#plt.plot(Fe,Fe,label="Fe")
plt.scatter(Fe,Ni,label="Ni")
#plt.scatter(Fe,W,label="W")
plt.scatter(Fe,Nb,label="Nb")
#plt.scatter(Fe,Mo,label="Mo")
plt.scatter(Fe,W,label="Mn")
#plt.plot(Fe,Ti,label="Ti")
#
#
#plt.xlabel('Distance (um)')
plt.xlabel('Concentration Fe (wt.%)')
#
plt.ylabel('Concentration (wt.%)')
#
plt.title("Concentration of Elements")
#
plt.legend()
#plt.xlim(30,35)
#plt.ylim(0,40)
#
plt.show()
# %%Lets get Plotting Function of Cr
# make subplots?
plt.scatter(Cr,Si,label="Si")
plt.scatter(Cr,Fe,label="Fe")
#plt.plot(Fe,Fe,label="Fe")
plt.scatter(Cr,Ni,label="Ni")
#plt.scatter(Fe,W,label="W")
plt.scatter(Cr,Nb,label="Nb")
#plt.scatter(Fe,Mo,label="Mo")
plt.scatter(Cr,W,label="Mn")
#plt.plot(Fe,Ti,label="Ti")
#
#
#plt.xlabel('Distance (um)')
plt.xlabel('Concentration Cr (wt.%)')
#
plt.ylabel('Concentration (wt.%)')
#
plt.title("Concentration of Elements")
#
plt.legend()
#plt.xlim(30,35)
#plt.ylim(0,40)
#
plt.show()


# %% Subplots

fig, axs = plt.subplots(6, sharex=True)
fig.suptitle('Concentration wt% as a function of Fe')
axs[0].scatter(Fe,Si)
#axs[0].set_title('Si')
axs[1].scatter(Fe,Cr)
#axs[1].set_title('Cr')
axs[2].scatter(Fe,Ni)
axs[3].scatter(Fe,Nb)
axs[4].scatter(Fe,Mo)
axs[5].scatter(Fe,Mn)
#plt.legend()
#plt.xlim(30,35)
#plt.ylim(0,40)

# %% Filter for Carbides
totalwtcarbide = 95 #max comp for filtering for carbides
M7_filter = (data['Elemental Totals']<totalwtcarbide) & (data["Cr Elemental Percents"] > 70)
M7_comp=data[M7_filter]
AM7_comp=M7_comp.loc[:,"Si Elemental Percents":"V  Elemental Percents"].mean(axis=0)
print(AM7_comp)

#M7_comp1=M7_comp.loc['Element Totals':'V Elemental Percents']#.mean(axis=1)
#print(M7_comp1)


MC_filter = (data['Elemental Totals']<totalwtcarbide) & (data["Nb Elemental Percents"] > 80)
MC_comp=data[MC_filter]
AMC_comp=MC_comp.loc[:,"Si Elemental Percents":"V  Elemental Percents"].mean(axis=0)
print(AMC_comp)

Avg_comp=data.loc[:,"Si Elemental Percents":"V  Elemental Percents"].mean(axis=0)
print(Avg_comp)
# %% WIRS
#filter dataset to remove interdendritic regions
totalwtlow=97 #threshold for filtering interdendritic regions may need to be tweaked
totalwthigh=103
crmax=26
nbmax=1
nimin=30
nimax=36.5

""" This value will influence the kline"""
maxfs=1#0.96#0.899#82.19485515/100 HP-2 #0.96 for M10 #0.899 for HP


max_filter = (data['Elemental Totals']>totalwtlow) & (data["Cr Elemental Percents"] < crmax) & (data["Nb Elemental Percents"] < nbmax) & (data['Elemental Totals']<totalwthigh) & (data["Ni Elemental Percents"] > nimin) & (data["Ni Elemental Percents"] < nimax)
primary_y=data#[max_filter]
#print(primary_y)

#plt.plot(primary_y['Relative Microns'],primary_y['          "Si Elemental Percents"'],label="Si")
#plt.plot(primary_y['Relative Microns'],primary_y["Cr Elemental Percents"],label="Cr")
#plt.show()

#for negatively segregating elements
primary_y['Si_bar']=(primary_y['Si Elemental Percents'] - primary_y['Si Elemental Percents'].min())/primary_y['Si Percent Errors']
primary_y['Cr_bar']=(primary_y["Cr Elemental Percents"] - primary_y["Cr Elemental Percents"].min())/primary_y["Cr Percent Errors"]
primary_y['Ni_bar']=(primary_y["Ni Elemental Percents"] - primary_y["Ni Elemental Percents"].min())/primary_y["Ni Percent Errors"] #if Ni negatively segregates
primary_y['Nb_bar']=(primary_y["Nb Elemental Percents"] - primary_y["Nb Elemental Percents"].min())/primary_y["Nb Percent Errors"]
#primary_y['Mo_bar']=(primary_y["Mo Elemental Percents"] - primary_y["Mo Elemental Percents"].min())/primary_y["Mo Percent Errors"]
primary_y['Mn_bar']=(primary_y["Mn Elemental Percents"] - primary_y["Mn Elemental Percents"].min())/primary_y["Mn Percent Errors"]
#W_bar=(primary_y["W  Elemental Percents"] - primary_y["W  Elemental Percents"].min())/primary_y["W  Percent Errors"]


#for positively segregating elements
primary_y['Fe_bar']=(primary_y["Fe Elemental Percents"].max() - primary_y["Fe Elemental Percents"])/primary_y["Fe Percent Errors"]
#primary_y['Ni_bar']=(primary_y["Ni Elemental Percents"].max() - primary_y["Ni Elemental Percents"])/primary_y["Ni Percent Errors"]
#Ti_bar=(primary_y["Ti Elemental Percents"].max() - primary_y["Ti Elemental Percents"])/primary_y["Ti Percent Errors"]

#Aggregate Values into a new dataframe
#Cbar=pd.DataFrame(data=[Si_bar,Cr_bar,Fe_bar,Ni_bar,W_bar,Nb_bar,Mo_bar,Ti_bar]).T
#Cbar.columns=['Sibar', 'Crbar', 'Febar', 'Nibar', 'Wbar', 'Nbbar', 'Mobar', 'Tibar']
#Cbar=pd.DataFrame(data=[Si_bar,Cr_bar,Fe_bar,Ni_bar,Nb_bar,Mo_bar]).T
#Cbar.columns=['Sibar', 'Crbar', 'Febar', 'Nibar', 'Nbbar', 'Mobar']
#primary_y['Avgbar'] = primary_y[['Si_bar', 'Cr_bar', 'Fe_bar', 'Ni_bar', 'Nb_bar', 'Mo_bar', 'Mn_bar']].mean(axis=1)
primary_y['Avgbar'] = primary_y[['Si_bar', 'Cr_bar', 'Fe_bar', 'Ni_bar', 'Nb_bar', 'Mn_bar']].mean(axis=1)
#print(primary_y)

#sort according to Cbar min to max
primary_y_sort=primary_y.sort_values(by=['Avgbar'])
#print(primary_y_sort)
primary_y_sort['Rank'] = primary_y_sort['Avgbar'].rank(ascending = 1)
#print(primary_y_sort)
primary_y_sort['Fsolid']=(primary_y_sort['Rank'] - 0.5)/primary_y_sort['Rank'].max()*maxfs
#print(primary_y_sort['Fsolid'])
f_solid=primary_y_sort['Fsolid']

# %%Lets get Plotting
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Si Elemental Percents'],label="Si")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Cr Elemental Percents'],label="Cr")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Fe Elemental Percents'],label="Fe")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Ni Elemental Percents'],label="Ni")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Nb Elemental Percents'],label="Nb")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mo Elemental Percents'],label="Mo")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mn Elemental Percents'],label="Mn")
#plt.plot(Csort['Fsolid'],Csort['Fe'],label="Fe")
#plt.plot(Csort['Fsolid'],Csort['Ni'],label="Ni")
##plt.plot(Csort['Fsolid'],Csort['W'],label="W")
#plt.plot(Csort['Fsolid'],Csort['Nb'],label="Nb")
#plt.plot(Csort['Fsolid'],Csort['Mo'],label="Mo")
#plt.plot(Csort['Fsolid'],Csort['Ti'],label="Ti")


plt.xlabel('Fraction Solid')
plt.ylabel('Concentration (wt.%)')
plt.title("Concentration of Elements")

#plt.legend()
#loc='best'

plt.show()

# %% Calculate k from Core
#Nominal Composition
C0={'Si':1.929,'Cr':24.571,'Fe':37.695,'Ni':33.206,'Nb':1.28,'Mn':0.837,'Mo':0.07} #M10
#C0={'Si':1.14,'Cr':25.2,'Fe':36.66,'Ni':35,'Nb':0.418,'Mn':0.899,'Mo':0.06} #HP-2
#C0={'Si':1.929,'Cr':24.571,'Fe':37.695,'Ni':33.206,'Nb':1.28,'Mn':0.837,'Mo':0.07}

acore=10 #points to average from start of sorted data

#pull C0 estimates from grid
C0Si=data['Si Elemental Percents'].mean()
C0Cr=data["Cr Elemental Percents"].mean()
C0Fe=data["Fe Elemental Percents"].mean()
C0Ni=primary_y_sort["Ni Elemental Percents"].mean()
C0W=data["W  Elemental Percents"].mean()
C0Nb=data["Nb Elemental Percents"].mean()
#C0Mo=data["Mo Elemental Percents"].mean()
C0Mn=data["Mn Elemental Percents"].mean()
C0Ti=data["Ti Elemental Percents"].mean()

#Average of the first 6 points to solidify to the total average composition
KSi=primary_y_sort['Si Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Si
#primary_y_sort['Si Elemental Percents'].iloc[0:5].mean(axis=0)
print(KSi)
KSic0=primary_y_sort['Si Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Si']
#primary_y_sort['Si Elemental Percents'].iloc[0:5].mean(axis=0)
print(KSic0)

KCr=primary_y_sort['Cr Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Cr
print(KCr)
KCrc0=primary_y_sort['Cr Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Cr']
print(KCrc0)

KFe=primary_y_sort['Fe Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Fe
print(KFe)
KFec0=primary_y_sort['Fe Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Fe']
print(KFec0)

KNi=primary_y_sort['Ni Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Ni
print(KNi)
KNic0=primary_y_sort['Ni Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Ni']
print(KNic0)


#KW=primary_y_sort['W  Elemental Percents'].iloc[0:5].mean(axis=0) / data["W  Elemental Percents"].mean()
KNb=primary_y_sort['Nb Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Nb
print(KNb)
KNbc0=primary_y_sort['Nb Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Nb']
print(KNbc0)

#KMo=primary_y_sort['Mo Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Mo
#print(KMo)
#KMoc0=primary_y_sort['Mo Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Mo']
#print(KMoc0)

KMn=primary_y_sort['Mn Elemental Percents'].iloc[0:acore].mean(axis=0) / C0Mn
print(KMn)
KMnc0=primary_y_sort['Mn Elemental Percents'].iloc[0:acore].mean(axis=0) / C0['Mn']
print(KMnc0)

#KTi=primary_y_sort['Ti Elemental Percents'].iloc[0:5].mean(axis=0) / data["Ti Elemental Percents"].mean()

# %% Calc from Curve

#CsSi=primary_y_sort['Si Elemental Percents'].div(C0['Si'])
#print(CsSi)
#lnCsSi=CsSi.div(C0['Si'])
#print(lnCsSi)

#lnCsNi=np.log(primary_y_sort['Ni Elemental Percents'].div(C0['Ni']))
lnCsSi=np.log(primary_y_sort['Si Elemental Percents'].div(data["Si Elemental Percents"].mean()))
lnCsCr=np.log(primary_y_sort['Cr Elemental Percents'].div(data["Cr Elemental Percents"].mean()))
lnCsFe=np.log(primary_y_sort['Fe Elemental Percents'].div(data["Fe Elemental Percents"].mean()))
lnCsNi=np.log(primary_y_sort['Ni Elemental Percents'].div(data["Ni Elemental Percents"].mean()))
lnCsNb=np.log(primary_y_sort['Nb Elemental Percents'].div(data["Nb Elemental Percents"].mean()))
#lnCsMo=np.log(primary_y_sort['Mo Elemental Percents'].div(data["Mo Elemental Percents"].mean()))
lnCsMn=np.log(primary_y_sort['Mn Elemental Percents'].div(data["Mn Elemental Percents"].mean()))

lnFL=np.log(1-primary_y_sort['Fsolid'])
FL=1-primary_y_sort['Fsolid']
FS=primary_y_sort['Fsolid']

#loglog(lnFL,lnCsSi)

plt.plot(lnFL,lnCsNi,label="Ni")

plt.xlabel('Ln(Fraction Liquid)')

plt.ylabel('Ln(Cs/C0)')

plt.title("Concentration of Elements")

plt.show()

def test(F,a,b): 
#    return math.log(a)+(1-a)*F+b
    return (a-1)*np.log(1-F)+(b)
Siparam, Siparam_cov = curve_fit(test, FS, lnCsSi) 
Crparam, Crparam_cov = curve_fit(test, FS, lnCsCr) 
Feparam, Feparam_cov = curve_fit(test, FS, lnCsFe) 
Niparam, Niparam_cov = curve_fit(test, FS, lnCsNi) 
Nbparam, Nbparam_cov = curve_fit(test, FS, lnCsNb) 
#Moparam, Moparam_cov = curve_fit(test, FS, lnCsMo) 
Mnparam, Mnparam_cov = curve_fit(test, FS, lnCsMn)   

  
print("Sine funcion coefficients:") 
print(Niparam) 
print("Covariance of coefficients:") 
print(Niparam_cov) 

# ans stores the new y-data according to  
# the coefficients given by curve-fit() function 
ansCr = test(FS,Crparam[0],Crparam[1])#((Crparam[0]-1)*lnFL+Crparam[1]) 
ansSi = ((Siparam[0]-1)*lnFL+Siparam[1])
ansNi = ((Niparam[0]-1)*lnFL+Niparam[1]) 
ansFe = ((Feparam[0]-1)*lnFL+Feparam[1]) 
ansNb = ((Nbparam[0]-1)*lnFL+Nbparam[1])
#ansMo = ((Moparam[0]-1)*lnFL+Moparam[1]) 
ansMn = ((Mnparam[0]-1)*lnFL+Mnparam[1])   
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
  
plt.plot(lnFL, lnCsCr, 'o', color ='red', label ="data") 
plt.plot(lnFL, ansCr, '--', color ='blue', label ="optimized data") 
plt.legend()
plt.title("Cr")
plt.xlabel('Ln(Fraction Solid)')
plt.ylabel('Ln(Cs/C0)') 
plt.show()

plt.plot(lnFL, lnCsSi, 'o', color ='red', label ="data") 
plt.plot(lnFL, ansSi, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.title("Si")
plt.xlabel('Ln(Fraction Solid)')
plt.ylabel('Ln(Cs/C0)') 
plt.show()

plt.plot(lnFL, lnCsNi, 'o', color ='red', label ="data") 
plt.plot(lnFL, ansNi, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.title("Ni")
plt.xlabel('Ln(Fraction Solid)')
plt.ylabel('Ln(Cs/C0)') 
plt.show()

plt.plot(lnFL, lnCsNb, 'o', color ='red', label ="data") 
plt.plot(lnFL, ansNb, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.title("Nb")
plt.xlabel('Ln(Fraction Solid)')
plt.ylabel('Ln(Cs/C0)') 
plt.show()

plt.plot(lnFL, lnCsFe, 'o', color ='red', label ="data") 
plt.plot(lnFL, ansFe, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.title("Fe")
plt.xlabel('Ln(Fraction Solid)')
plt.ylabel('Ln(Cs/C0)') 
plt.show()

#define new k values
#K["Si"]=??
KSi_line=Siparam[0] #abs(1-Siparam[0])
print(KSi_line)
KCr_line=Crparam[0] #abs(Crparam[0]-2) #Crparam[0]
print(KCr_line)
KFe_line=Feparam[0] #klineFe=abs(Feparam[0]-2)
print(KFe_line) 
KNi_line=Niparam[0] #abs(Niparam[0]-2) #Niparam[0]
print(KNi_line)
KNb_line=Nbparam[0] #abs(Nbparam[0]-2) #Nbparam[0]
print(KNb_line)
#KMo_line=Moparam[0] #abs(Moparam[0]-2) #Moparam[0]
#print(KMo_line)
KMn_line=Mnparam[0] #abs(Mnparam[0]-2) #Mnparam[0]
print(KMn_line)
# %%K fit with linear regression
X=lnFL
X=sm.add_constant(X)
Simodel=sm.OLS(lnCsSi,X).fit()
Simodel.summary()
klSi=1-0.3637
print(klSi)
Crmodel=sm.OLS(lnCsCr,X).fit()
Crmodel.summary()
Nbmodel=sm.OLS(lnCsNb,X).fit()
Nbmodel.summary()
Mnmodel=sm.OLS(lnCsMn,X).fit()
Mnmodel.summary()
Nimodel=sm.OLS(lnCsNi,X).fit()
Nimodel.summary()
Femodel=sm.OLS(lnCsFe,X).fit()
Femodel.summary()
# %% Scheil Calculation
def scheil(k,Cnom,fs):
       return k*Cnom*(1-fs)**(k-1)
#from dendrite core k values
NEQ_Si=scheil(KSi,C0Si,f_solid)
NEQ_Cr=scheil(KCr,C0Cr,f_solid)
NEQ_Fe=scheil(KFe,C0Fe,f_solid)
NEQ_Ni=scheil(KNi,C0Ni,f_solid)
NEQ_Mn=scheil(KMn,C0Mn,f_solid)
NEQ_Nb=scheil(KNb,C0Nb,f_solid)
#NEQ_Mo=scheil(KMo,C0Mo,f_solid)

# %% Equlibrium Calculation
def equil(k,Cnom,fs):
       return k*Cnom/((1-fs)+k*fs)

EQ_Si=equil(KSi,C0Si,f_solid)
EQ_Cr=equil(KCr,C0Cr,f_solid)
EQ_Fe=equil(KFe,C0Fe,f_solid)
EQ_Ni=equil(KNi,C0Ni,f_solid)
EQ_Mn=equil(KMn,C0Mn,f_solid)
EQ_Nb=equil(KNb,C0Nb,f_solid)
#EQ_Mo=equil(KMo,C0Mo,f_solid)

# %% Brody Flemings Calculation-work in progress
def BF(k,Cnom,fs,alpha):
       return k*Cnom*(1-(1-2*alpha*k)*fs)**((k-1)/(1-2*alpha*k))
   
# %% Plot solidification path     
figure(num=None, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Si Elemental Percents'],label="Si", color='blue')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Cr Elemental Percents'],label="Cr", color='green')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Fe Elemental Percents'],label="Fe", color='red')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Ni Elemental Percents'],label="Ni", color='magenta')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Nb Elemental Percents'],label="Nb", color='cyan')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mo Elemental Percents'],label="Mo")
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mn Elemental Percents'],label="Mn", color='black')


#plt.plot(primary_y_sort['Fsolid'],NEQ_Si,label="NESi", color='blue')
plt.plot(primary_y_sort['Fsolid'],NEQ_Cr,label="NECr", color='green')
plt.plot(primary_y_sort['Fsolid'],NEQ_Fe,label="NEFe", color='red')
plt.plot(primary_y_sort['Fsolid'],NEQ_Ni,label="NENi", color='magenta')
#plt.plot(primary_y_sort['Fsolid'],NEQ_Nb,label="NENb", color='cyan')
#plt.plot(primary_y_sort['Fsolid'],NEQ_Mo,label="NEMo")
#plt.plot(primary_y_sort['Fsolid'],NEQ_Mn,label="NEMn", color='black')

#plt.plot(primary_y_sort['Fsolid'],EQ_Si,label="ESi", color='blue', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Cr,label="ECr", color='green', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Fe,label="EFe", color='red', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Ni,label="ENi", color='magenta', linestyle='dashed')
#plt.plot(primary_y_sort['Fsolid'],EQ_Nb,label="ENb", color='cyan', linestyle='dashed')
#plt.plot(primary_y_sort['Fsolid'],EQ_Mo,label="EMo")
#plt.plot(primary_y_sort['Fsolid'],EQ_Mn,label="EMn", color='black', linestyle='dashed')


plt.xlabel('Fraction Solid')

plt.ylabel('Concentration (wt.%)')

#plt.title("Solidification Path Solidification")
plt.xlim(0,1.0)
plt.ylim(20,45)
#plt.legend()
#loc='best'

plt.show()

# %% Plot solidification path     Major Elements
figure(num=None, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Si Elemental Percents'],label="Si", color='blue')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Cr Elemental Percents'],label="Cr", color='green')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Fe Elemental Percents'],label="Fe", color='red')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Ni Elemental Percents'],label="Ni", color='magenta')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Nb Elemental Percents'],label="Nb", color='cyan')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mo Elemental Percents'],label="Mo")
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mn Elemental Percents'],label="Mn", color='black')


#plt.plot(primary_y_sort['Fsolid'],NEQ_Si,label="NESi", color='blue')
plt.plot(primary_y_sort['Fsolid'],NEQ_Cr,label="NECr", color='green')
plt.plot(primary_y_sort['Fsolid'],NEQ_Fe,label="NEFe", color='red')
plt.plot(primary_y_sort['Fsolid'],NEQ_Ni,label="NENi", color='magenta')
#plt.plot(primary_y_sort['Fsolid'],NEQ_Nb,label="NENb", color='cyan')
#plt.plot(primary_y_sort['Fsolid'],NEQ_Mo,label="NEMo")
#plt.plot(primary_y_sort['Fsolid'],NEQ_Mn,label="NEMn", color='black')

#plt.plot(primary_y_sort['Fsolid'],EQ_Si,label="ESi", color='blue', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Cr,label="ECr", color='green', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Fe,label="EFe", color='red', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Ni,label="ENi", color='magenta', linestyle='dashed')
#plt.plot(primary_y_sort['Fsolid'],EQ_Nb,label="ENb", color='cyan', linestyle='dashed')
#plt.plot(primary_y_sort['Fsolid'],EQ_Mo,label="EMo")
#plt.plot(primary_y_sort['Fsolid'],EQ_Mn,label="EMn", color='black', linestyle='dashed')


plt.xlabel('Fraction Solid')

plt.ylabel('Concentration (wt.%)')

#plt.title("Solidification Path Solidification")
plt.xlim(0,1.0)
plt.ylim(20,45)
#plt.legend()
#loc='best'

plt.show()

# %% Minor Elements
figure(num=None, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Si Elemental Percents'],label="Si", color='blue')
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Nb Elemental Percents'],label="Nb", color='cyan')
#plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mo Elemental Percents'],label="Mo")
plt.plot(primary_y_sort['Fsolid'],primary_y_sort['Mn Elemental Percents'],label="Mn", color='black')


plt.plot(primary_y_sort['Fsolid'],NEQ_Si,label="NESi", color='blue')
plt.plot(primary_y_sort['Fsolid'],NEQ_Nb,label="NENb", color='cyan')
#plt.plot(primary_y_sort['Fsolid'],NEQ_Mo,label="NEMo")
plt.plot(primary_y_sort['Fsolid'],NEQ_Mn,label="NEMn", color='black')

plt.plot(primary_y_sort['Fsolid'],EQ_Si,label="ESi", color='blue', linestyle='dashed')
plt.plot(primary_y_sort['Fsolid'],EQ_Nb,label="ENb", color='cyan', linestyle='dashed')
#plt.plot(primary_y_sort['Fsolid'],EQ_Mo,label="EMo")
plt.plot(primary_y_sort['Fsolid'],EQ_Mn,label="EMn", color='black', linestyle='dashed')


plt.xlabel('Fraction Solid')

plt.ylabel('Concentration (wt.%)')

#plt.title("Solidification Path Solidification")
plt.xlim(0,1.0)
#plt.ylim(20,45)
#plt.legend()
#loc='best'

plt.show()


# %% ??????
#print(primary_y_sort['Si Elemental Percents'])
#print(lnCs)
#C=data['C']
#Nb=data['Nb']
#Si=data['Si']
#Ti=data['Ti']
#W=data['W']
#F_Y_MC=data['Fraction Y E1']
#F_MC=data['Fraction MC E1']
#F_Y_M7=data['Fraction Y E2']
#F_M7=data['Fraction M7C3 E2']
#
##print(F_Y_MC)
##print(F_MC)
##print(F_Y_M7)
##print(F_M7)
#
##set x and y
#y=F_MC
#X=Si
#X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
#
#
## To get statistics of the dataset
## Note the difference in argument order
#model = sm.OLS(y, X).fit()
#predictions = model.predict(X) # make the predictions by the model
#
## Print out the statistics
#model.summary()
# %% Output table?
from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
#['Si', 'Cr', 'Fe', 'Ni', 'Nb', 'Mn']
x.add_row(["Adelaide", 1295, 1158259, 600.5])
x.add_row(["Brisbane", 5905, 1857594, 1146.4])
x.add_row(["Darwin", 112, 120900, 1714.7])
x.add_row(["Hobart", 1357, 205556, 619.5])
x.add_row(["Sydney", 2058, 4336374, 1214.8])
x.add_row(["Melbourne", 1566, 3806092, 646.9])
x.add_row(["Perth", 5386, 1554769, 869.4])

print(x)
# %% Bar Charts
# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [KSi, KCr, KFe, KNi, KNb, KMn]
bars2 = [KSic0, KCrc0, KFec0, KNic0, KNbc0, KMnc0]
bars3 = [KSi_line, KCr_line, KFe_line, KNi_line, KNb_line, KMn_line]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Kmean')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='KC0')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='K_line')
 
# Add xticks on the middle of the group bars
plt.xlabel('Element', fontweight='bold')
plt.ylabel('Partition Coefficient (k)', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Si', 'Cr', 'Fe', 'Ni', 'Nb', 'Mn'])
plt.axhline(1, color="black")#.plot(["Si","Mn"], [1,1], "k--")
# Create legend & Show graphic
plt.ylim(0,1.2)
plt.legend()
plt.show()
#plt.savefig('M10 DTA K-values.png')
