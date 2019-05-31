import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datos=pd.read_excel('Vectores.xlsx')
df=pd.DataFrame(datos)
x=df['Presupues'].values
y=df['Calif. Tomatoes'].values

print("Valor promedio del presupuesto:   ", df['Presupues'].max())
info=df[['Presupues','Calif. Tomatoes']].as_matrix()
print(info)

X=np.array(list(zip(x,y)))
print(X)

kmeans=KMeans(n_clusters=5)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_
colors=["m.","r.","c.","y.","b."]

for i in range(len(X)):
    print("Coordenada: ", [X])
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidhts=5,zorder=10)
plt.show()