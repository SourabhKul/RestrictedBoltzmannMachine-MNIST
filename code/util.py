import numpy as np
import matplotlib.pyplot as plt

def plot_im_array2(X,S,N,A,title):  
    #Display a set of N row vectors of size S^2 as an AxA array of S^S images.
    #add the title "title" to the plot.  
    I = np.ones(((A*(1+S),A*(1+S))))*max(X.flatten())
    k=0
    for i in range(A):
        for j in range(A):
            I[i*(S)+i:(i+1)*S+i,j*S+j:(j+1)*S+j] = X[k,:].reshape((S, S))
            k=k+1
            if(k==N): break
        if(k==N): break
            
    plt.imshow(I, cmap=plt.cm.gray,interpolation=None)
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    
    
def acorr(x):
    #Compute the autocorrelation of a sequence x.
    #Xmust be a 1D array
    m = np.mean(x)
    xc = np.correlate(x-m, x-m, mode='full')
    xc /= xc[xc.argmax()]
    xchalf = xc[xc.size / 2:]
    return(xchalf)