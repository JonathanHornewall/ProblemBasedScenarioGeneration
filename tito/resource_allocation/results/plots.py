import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#---------- p = 05
df05 = pd.read_csv("df05.csv")
df05 = df05.sort_values(by=['method'])
#not LS, CART y SAA
df05 = df05[~df05["method"].isin(['LS', 'CART', 'SAA'])]

fig, axes = plt.subplots(3, 3,figsize=(15, 15))
#AD,ER-SAA, KNN, M5+AD
#tab:blue , tab:green ,tab:red , tab:brown
ax = sns.boxplot(y="OoS", hue="method", data= df05[df05["T"]==100],ax = axes[0,0],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove() 
ax.set(ylim=(0, 5))
ax = sns.boxplot(y="OoS", hue="method", data= df05[df05["T"]==1000],ax = axes[0,1],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 5)) 
ax = sns.boxplot(y="OoS", hue="method", data= df05[df05["T"]==10000],ax = axes[0,2],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 5))

plt.setp(axes[0,0], ylabel='p = 0.5')
plt.setp(axes[0,1], ylabel='')
plt.setp(axes[0,2], ylabel='')

plt.setp(axes[0,0], xlabel='N = 10^2')
plt.setp(axes[0,1], xlabel='N = 10^3')
plt.setp(axes[0,2], xlabel='N = 10^4')

#---------- p = 1
df1 = pd.read_csv("df1.csv")
df1 = df1.sort_values(by=['method'])
df1 = df1[~df1["method"].isin(['LS', 'CART', 'SAA'])]


ax = sns.boxplot(y="OoS", hue="method", data= df1[df1["T"]==100],ax = axes[1,0],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove() 
ax.set(ylim=(0, 5))
ax = sns.boxplot(y="OoS", hue="method", data= df1[df1["T"]==1000],ax = axes[1,1],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 5)) 
ax = sns.boxplot(y="OoS", hue="method", data= df1[df1["T"]==10000],ax = axes[1,2],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 5))

plt.setp(axes[1,0], ylabel='p = 1.0')
plt.setp(axes[1,1], ylabel='')
plt.setp(axes[1,2], ylabel='')

plt.setp(axes[1,0], xlabel='N = 10^2')
plt.setp(axes[1,1], xlabel='N = 10^3')
plt.setp(axes[1,2], xlabel='N = 10^4')


#---------- p = 2
df2 = pd.read_csv("df2.csv")
df2 = df2.sort_values(by=['method'])
df2 = df2[~df2["method"].isin(['LS', 'CART', 'SAA'])]


ax = sns.boxplot(y="OoS", hue="method", data= df2[df2["T"]==100],ax = axes[2,0],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove() 
ax.set(ylim=(0, 10))
ax = sns.boxplot(y="OoS", hue="method", data= df2[df2["T"]==1000],ax = axes[2,1],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 10)) 
ax = sns.boxplot(y="OoS", hue="method", data= df2[df2["T"]==10000],ax = axes[2,2],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 10))

plt.setp(axes[2,0], ylabel='p = 2.0')
plt.setp(axes[2,1], ylabel='')
plt.setp(axes[2,2], ylabel='')

plt.setp(axes[2,0], xlabel='N = 10^2')
plt.setp(axes[2,1], xlabel='N = 10^3')
plt.setp(axes[2,2], xlabel='N = 10^4')



ax.legend(loc='upper center', bbox_to_anchor=(-0.75, -0.1), fancybox=True, shadow=True, ncol=7)

#plt.savefig("plots_results_allocation.pdf")
plt.savefig("tito_plots_results_allocation.pdf")
