import cv2, numpy as np
from scipy import ndimage as ndi
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
C=np.load('center.npy')
L=cv2.cvtColor(s,cv2.COLOR_BGR2LAB)[...,0].astype(np.float32)
Ls=cv2.GaussianBlur(L,(0,0),2.0)
t=np.roll(C,-1,0)-np.roll(C,1,0); t/=np.linalg.norm(t,axis=1)[:,None]
N=np.c_[-t[:,1],t[:,0]]
offs=np.arange(0,60,1.0)
def edge(sign):
    X=C[:,0,None]+sign*N[:,0,None]*offs[None,:]; Y=C[:,1,None]+sign*N[:,1,None]*offs[None,:]
    V=ndi.map_coordinates(Ls,[Y,X],order=1,mode='nearest')
    G=np.gradient(V,axis=1)
    G=ndi.gaussian_filter1d(G,1.5,axis=1)
    lo,hi=22,48
    idx=np.argmax(G[:,lo:hi],1)+lo
    return idx.astype(float)
wl=edge(+1); wr=edge(-1)
def clean(w):
    w=ndi.median_filter(w,size=15,mode='wrap')
    w=ndi.gaussian_filter1d(w,6,mode='wrap')
    return w
wl,wr=clean(wl),clean(wr)
print('left',np.percentile(wl,[5,50,95]),'right',np.percentile(wr,[5,50,95]))
np.save('wl.npy',wl); np.save('wr.npy',wr); np.save('N.npy',N)
ov=s.copy()
Lp=C+N*wl[:,None]; Rp=C-N*wr[:,None]
cv2.polylines(ov,[Lp.astype(np.int32).reshape(-1,1,2)],True,(0,255,0),2)
cv2.polylines(ov,[Rp.astype(np.int32).reshape(-1,1,2)],True,(255,160,0),2)
cv2.imwrite('edges_ov.png',ov)
