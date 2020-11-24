def gauss(x, mu, sigma):
    return np.exp(-( (x-mu)**2 / ( 2.0 * sigma**2 ) ) ) 

def baseline_als(y, lam=4e7, p=0.01, niter=10):
    """Asymmetric Least Squares Baseline correction.
    There are two parameters: 
    p for asymmetry and λ for smoothness. 
    Both have to be tuned to the data at hand. 
    We found that generally 
    0.001 ≤ p ≤ 0.1 is a good choice 
    (for a signal with positive peaks) 
    and 10^2 ≤ λ ≤ 10^9 , 
    but exceptions may occur. 
    In any case one should vary λ on a grid 
    that is approximately linear for log λ
    
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) 
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


x=np.linspace(1, 100,500)
y0=2*gauss(x,50,30)
y1=20*gauss(x,33,4)
y2=14*gauss(x,42,6)

plt.plot(x,y1)
plt.plot(x,y2)

y3=y0+y1+y2 + np.random.rand(len(x))

f=open('task_8.3.csv', 'w')
f.write('x, y\n')
for i in range(len(x)):
    f.write('{:.3f}, {:.3f}\n'.format(x[i],y3[i]))
f.close()

plt.plot(x,y3-baseline_als(y3)) 
