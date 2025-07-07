

function LS(Y,X,J,L)
    θₗₛ = ones(J,L+1)*10000
    for j in 1:J
        yⱼ = Y[j,:]
        β = coef(linregress(X,yⱼ)) 
        θₗₛ[j,:] = β
    end
    return θₗₛ
end

#=
py"""
import numpy as np
from sklearn.linear_model import LinearRegression

def LR(X,y):
    reg = LinearRegression().fit(X, y)
    return reg.coef_,reg.intercept_
"""

#py"LR"(rand(3,3),rand(3))
=#