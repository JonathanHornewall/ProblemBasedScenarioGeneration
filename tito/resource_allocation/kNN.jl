
function generateDemandKNN(Xs,Yj_aux,k,Xfresh) #Obtengo nuevos features)
    kdtree = KDTree(transpose(Xs))
    #@show kdtree
    idxs, dists = knn(kdtree, transpose(Xfresh), k, true)
    Yj = Yj_aux[:,idxs[1]] #Obtengo la demanda estimada gracias a KNN
    #@show Yj
    return Yj
end