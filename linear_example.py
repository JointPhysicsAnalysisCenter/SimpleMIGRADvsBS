# -*- coding: utf-8 -*-
"""

@author: cesar fernandez-ramirez

version: 2022-04-28

Bootstrap and statistics for Joint Physics Analysis Center review
"""
#%%

###########################################################
#   Python libraries
###########################################################

import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

###########################################################
#   JPAC color style
###########################################################

jpac_blue   = "#1F77B4"; jpac_red    = "#D61D28";
jpac_green  = "#2CA02C"; jpac_orange = "#FF7F0E";
jpac_purple = "#9467BD"; jpac_brown  = "#8C564B";
jpac_pink   = "#E377C2"; jpac_gold   = "#BCBD22";
jpac_aqua   = "#17BECF"; jpac_grey   = "#7F7F7F";

dashes = 60*'_'; label_size = 12

#%%

###########################################################
#   Data generation
###########################################################

def curve_generator(x,a): p = np.poly1d(a); return p(x)

def errores_sym(error_input):
    npoints = len(error_input)
    points  = np.arange(npoints)
    nxypoints = (2,npoints)
    uncertainty = np.zeros(nxypoints)
    for i in points:
        uncertainty[0][i] = error_input[i]
        uncertainty[1][i] = error_input[i]
    return uncertainty

def pseudodataset(ydata,y_error):
    pseudodata = [ np.random.normal(ydata[i],y_error[i]) for i in np.arange(y_error.size)]
    return pseudodata

def dataset(xdata,a,errorsize):
    y = curve_generator(xdata, a)
    y_error = np.abs(np.random.normal(0.,errorsize,size=xdata.size))
    y_noise = [ y_error[i]*np.random.normal() for i in np.arange(y_error.size) ]
    yerrors = errores_sym(y_error)
    ydata = y + y_noise
    return ydata, y_error, yerrors

#%%

###########################################################
#   Minuit fit  and errors for the linear example
###########################################################

np.random.seed(1729)
ndatapoints = 40; a = [0.5, 2.,]; errorsize = 1.5;
xdata = np.linspace(0, 20, ndatapoints)
ydata, y_error, yerrors = dataset(xdata,a,errorsize)

def line(x, a, b): return a + x * b

def chisquare(yth,ydata,yerr):
    chisq = [ (yth[i] - ydata[i])/yerr[i] for i in np.arange(yth.size)]
    return np.sum(np.multiply(chisq,chisq)), chisq

data_x = xdata; data_y = ydata; data_yerr = y_error;
least_squares = LeastSquares(data_x, data_y, data_yerr, line)
m = Minuit(least_squares, a=0, b=0)
m.migrad(); m.hesse(); m.minos() 
print(m.params)
print(m.covariance)
print(m.covariance.correlation())
print(dashes); print(r'chi2=',m.fval)
print(r'chi2/dof=',m.fval/(len(data_y)-len(m.values)))
print(dashes)

fig = plt.figure()
plt.xlim((-2.,22.)); plt.ylim((-5.,20.))
plt.xlabel(r'$x$',fontsize=label_size); plt.ylabel(r'$y$',fontsize=label_size)
props = dict(boxstyle='round', facecolor=jpac_blue, edgecolor=jpac_grey, lw=2., alpha=0.15)
plt.text(0.0,17.0,r'$y=\theta_1 + \theta_2\, x$',size=15.,rotation=0.,color=jpac_orange,bbox=props)
plt.errorbar(xdata, ydata, yerrors, fmt="o", markersize=3,capsize=5., c=jpac_blue, alpha=1)
plt.plot(data_x, line(data_x, *m.values), label="fit",c=jpac_orange)
plt.tick_params(direction='in', top=False, right=False,left=True,size=5,labelleft=True,labelbottom=True,labelsize=label_size)
plt.show()
yth = line(data_x, *m.values)
c0bff, c2bff = chisquare(yth,ydata,y_error)
valores = m.values
fig.savefig("linear_fit.pdf", bbox_inches='tight')
fig.savefig("linear_fit.png", bbox_inches='tight')
#%%

###########################################################
#   Bootstrap fit and errors
###########################################################

nbs = 10000; ypseudodata = []
for i in range(nbs): ypseudodata.append(pseudodataset(ydata,y_error))

a_bs = []; b_bs = []; chisq = [];
for i in range(nbs):
    least_squares = LeastSquares(data_x, ypseudodata[i], data_yerr, line)
    m = Minuit(least_squares, a=0, b=0)
    m.migrad()
    a_bs.append(m.values[0]); b_bs.append(m.values[1])
    yth = line(data_x,a_bs[i],b_bs[i])
    c1, c2 = chisquare(yth,ypseudodata[i],y_error)
    chisq.append(c1)

df_parameters = pd.DataFrame({"p0":a_bs,"p1":b_bs});

print(dashes); print('Fit parameters')
print('a=',np.mean(a_bs),' -',np.abs(np.mean(a_bs)-np.quantile(a_bs,0.16)),' +',np.abs(np.quantile(a_bs,0.84)-np.mean(a_bs)))
print('b=',np.mean(b_bs),' -',np.abs(np.mean(b_bs)-np.quantile(b_bs,0.16)),' +',np.abs(np.quantile(b_bs,0.84)-np.mean(b_bs)))
print(dashes); print('covariance matrix')
print(df_parameters.cov()); 
print(dashes); print('correlation matrix')
print(df_parameters.corr())
print(dashes)

#%%

###########################################################
#   Noncentral chi^2
###########################################################

bins1 = 100; alphachoice = 0.5; npoints = 1000
xdw, xup =  1., 160.
x = np.linspace(xdw,xup, npoints)
fig = plt.figure(figsize=(6,4))
plt.ylim((0.0,0.035))
plt.xlabel(r'$\chi^2_{BS}$',size=label_size)
plt.plot(x, stat.ncx2.pdf(x,c0bff,len(xdata)-2),'-', lw=2,c=jpac_blue,zorder=2, label='Noncentral $\chi^2$ distribution')
plt.tick_params(direction='in', top=False, right=False,left=True,size=5,labelleft=True,labelbottom=True,labelsize=label_size)
plt.hist(chisq, bins=bins1, density=True,color=jpac_orange,alpha=alphachoice, label='Bootstrap')
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right',ncol=1,frameon=True,fontsize=label_size-3)
plt.show()
fig.savefig("noncentral_chi2.pdf", bbox_inches='tight')
fig.savefig("noncentral_chi2.png", bbox_inches='tight')

#%%

###########################################################
#   Parameter distribution
###########################################################

for name in ["p0","p1"]:
    fig = plt.figure(figsize=(6,4))
    if name=="p0":
        i = 0; err = 0.07; xdw, xup =  1.7, 2.5
        namefile = 'theta2.pdf'
        nombre = r'$\theta_2$'
        plt.ylim((0.0,6.))
    else:
        i = 1; err = 0.004; xdw, xup =  0.47, 0.52
        namefile = 'theta1.pdf'
        nombre = r'$\theta_1$'
        plt.ylim((0.0,110.))

    x = np.linspace(xdw,xup, npoints)
    plt.xlabel(nombre,size=label_size)
    plt.hist(df_parameters[name], bins=bins1, density=True,color=jpac_orange,alpha=alphachoice,label='Bootstrap')
    plt.plot(x, stat.norm.pdf(x,valores[i],err),'-', lw=2,c=jpac_blue,zorder=2,label='Gaussian distribution')
    plt.tick_params(direction='in', top=False, right=False,left=True,size=5,labelleft=True,labelbottom=True,labelsize=label_size)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right',ncol=1,frameon=True,fontsize=label_size-3)
    plt.show()
    fig.savefig(namefile, bbox_inches='tight')
