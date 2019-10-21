from pylab import *
from numpy import *
from scipy import linalg

def compute_fundamental(x1,x2):    
    n = x1.shape[1]
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))
    return F/F[2,2]    

class RansacModel(object):
    
    def __init__(self,debug=False):
        self.debug = debug
    
    def fit(self,data):
        
        data = data.T
        x1 = data[:3,:8]
        x2 = data[3:,:8]
        
        F = compute_fundamental(x1,x2)
        return F
    
    def get_error(self,data,F):
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        Fx1 = dot(F,x1)
        Fx2 = dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = ( diag(dot(x1.T,dot(F,x2))) )**2 / denom 
        return err

def F_from_ransac(x1,x2,model,maxiter=200,match_theshold=1e-6):
    import ransac
    data = vstack((x1,x2))
    F,ransac_data = ransac.ransac(data.T,model,8,maxiter,match_theshold,20,return_all=True)
    return F, ransac_data['inliers']