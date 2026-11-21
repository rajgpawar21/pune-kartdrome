import cv2, numpy as np, json
from scipy import ndimage as ndi
from scipy.optimize import lsq_linear
import scipy.sparse as sp
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
C0=np.load('center.npy'); wl=np.load('wl.npy'); wr=np.load('wr.npy'); N0=np.load('N.npy')
n=len(C0)
# recentre + regularise width
off=ndi.gaussian_filter1d((wl-wr)/2,10,mode='wrap')
h=np.clip(ndi.gaussian_filter1d((wl+wr)/2,10,mode='wrap'),26,34)
C=C0+N0*off[:,None]
t=np.roll(C,-1,0)-np.roll(C,1,0); t/=np.linalg.norm(t,axis=1)[:,None]
N=np.c_[-t[:,1],t[:,0]]
STEP=np.mean(np.linalg.norm(np.roll(C,-1,0)-C,axis=1)); print('step',STEP,'n',n)
MARGIN=8.0   # px kart-centre inset from edge
lim=h-MARGIN
# periodic second difference
I=sp.identity(n,format='csr')
P1=sp.csr_matrix(np.roll(np.eye(n),-1,0)); M1=sp.csr_matrix(np.roll(np.eye(n),1,0))
D2=(P1-2*I+M1); D1=(P1-I)
def solve(lam):
    # p = C + a*N  ; minimise |D2 p|^2 + lam|D1 p|^2
    out=[]
    for d,w in ((0,1.0),(1,1.0)):
        pass
    A2=sp.hstack([D2.multiply(N[:,0][None,:]), D2.multiply(N[:,1][None,:])]) # not valid; do directly below
    return None
Nx=sp.diags(N[:,0]); Ny=sp.diags(N[:,1])
def build(lam):
    Ax=D2@Nx; Ay=D2@Ny
    bx=-(D2@C[:,0]); by=-(D2@C[:,1])
    A=sp.vstack([Ax,Ay]); b=np.r_[bx,by]
    if lam>0:
        Bx=np.sqrt(lam)*(D1@Nx); By=np.sqrt(lam)*(D1@Ny)
        A=sp.vstack([A,Bx,By]); b=np.r_[b,-np.sqrt(lam)*(D1@C[:,0]),-np.sqrt(lam)*(D1@C[:,1])]
    return A,b
res={}
for lam in (0.0,0.02,0.1):
    A,b=build(lam)
    r=lsq_linear(A,b,bounds=(-lim,lim),method='trf',lsmr_tol='auto',max_iter=200)
    a=r.x; Pt=C+N*a[:,None]
    d=np.linalg.norm(np.roll(Pt,-1,0)-Pt,axis=1).sum()
    # curvature
    d1=(np.roll(Pt,-1,0)-np.roll(Pt,1,0))/2; d2=np.roll(Pt,-1,0)-2*Pt+np.roll(Pt,1,0)
    k=(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum((d1**2).sum(1),1e-9)**1.5
    res[lam]=(a,Pt,d,np.abs(k).max())
    print(lam,'len',round(d,1),'centre len',round(np.linalg.norm(np.roll(C,-1,0)-C,axis=1).sum(),1),'maxk',np.abs(k).max(),'cost',r.cost)
np.savez('geom.npz',C=C,N=N,h=h,lim=lim,a0=res[0.0][0],a1=res[0.02][0],a2=res[0.1][0])
ov=s.copy()
for lam,col in ((0.0,(0,0,255)),(0.02,(0,255,0)),(0.1,(255,0,255))):
    cv2.polylines(ov,[res[lam][1].astype(np.int32).reshape(-1,1,2)],True,col,2)
cv2.imwrite('line_ov.png',ov)
