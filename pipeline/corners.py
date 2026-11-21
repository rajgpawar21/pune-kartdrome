import cv2, numpy as np, json
from scipy import ndimage as ndi
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
g=np.load('geom.npz'); C=g['C']; N=g['N']; a=g['a0']; h=g['h']
P=C+N*a[:,None]
n=len(P)
SC=7.0/(2*np.median(h))   # m per px, assuming ~7 m wide track (ASSUMPTION)
print('m/px',SC,'track width px',2*np.median(h))
Ps=np.c_[ndi.gaussian_filter1d(P[:,0],3,mode='wrap'),ndi.gaussian_filter1d(P[:,1],3,mode='wrap')]
d1=(np.roll(Ps,-1,0)-np.roll(Ps,1,0))/2; d2=np.roll(Ps,-1,0)-2*Ps+np.roll(Ps,1,0)
ds=np.linalg.norm(np.roll(Ps,-1,0)-Ps,axis=1)
k=(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum((d1**2).sum(1),1e-9)**1.5   # px^-1 ; +ve = right (clockwise on screen)
k=ndi.gaussian_filter1d(k,2,mode='wrap')
s_px=np.r_[0,np.cumsum(ds)[:-1]]; L=ds.sum()
print('lap length px',L,'m',L*SC)
# speed profile
alat=11.0; vmax=17.0; aacc=4.0; abr=9.0
kk=np.abs(k)/SC  # 1/m
vcap=np.minimum(vmax,np.sqrt(alat/np.maximum(kk,1e-6)))
dsm=ds*SC
v=vcap.copy()
for _ in range(3):
    for i in range(2*n):   # forward accel limit (wrap twice)
        j=i%n; jn=(j+1)%n
        v[jn]=min(v[jn],np.sqrt(v[j]**2+2*aacc*dsm[j]),vcap[jn])
    for i in range(2*n):   # backward brake limit
        j=(-i)%n; jp=(j-1)%n
        v[jp]=min(v[jp],np.sqrt(v[j]**2+2*abr*dsm[jp]),vcap[jp])
print('v range m/s',v.min(),v.max())
lap=np.sum(dsm/v); print('est lap time s',lap)
# corner detection
hi,lo=1/16.0,1/45.0   # 1/m thresholds
sign=np.sign(k)
mask_hi=kk>hi; mask_lo=kk>lo
# grow hysteresis, same sign
seg=[]
lab,nl=ndi.label(np.r_[mask_lo,mask_lo,mask_lo])
lab=lab[n:2*n]
cand=[]
for l in np.unique(lab[lab>0]):
    idx=np.where(lab==l)[0]
    if mask_hi[idx].any(): cand.append(idx)
# rotate so we start at index 0; handle wrap segments by unwrapping
segs=[]
for idx in cand:
    if idx[0]==0 and idx[-1]==n-1:
        # wraps: reorder
        brk=np.where(np.diff(idx)>1)[0]
        if len(brk): idx=np.r_[idx[brk[0]+1:],idx[:brk[0]+1]]
    segs.append(idx)
# split segments where sign flips
out=[]
for idx in segs:
    sg=sign[idx]; cut=np.where(np.diff(sg)!=0)[0]
    parts=np.split(idx,cut+1)
    for p in parts:
        if len(p)>=4 and (kk[p]>hi).any(): out.append(p)
out.sort(key=lambda p:p[0])
# merge same-sign neighbours separated by short gap
merged=[]
for p in out:
    if merged:
        q=merged[-1]; gap=(p[0]-q[-1])%n
        if sign[p[0]]==sign[q[0]] and gap*np.mean(ds)*SC<8.0:
            fill=np.arange(q[-1]+1,q[-1]+gap+1)%n
            merged[-1]=np.r_[q,fill,p]; continue
    merged.append(p)
corners=[]
for i,p in enumerate(merged):
    ap=p[np.argmin(v[p])]  # apex = min speed
    kmax=kk[p].max(); R=1/kmax
    corners.append(dict(id=i+1,start=int(p[0]),end=int(p[-1]),apex=int(ap),dir='R' if sign[ap]>0 else 'L',
        rmin_m=round(float(R),1),vmin_kmh=round(float(v[ap]*3.6),0),
        vin_kmh=round(float(v[(p[0]-int(10/ (np.mean(dsm))))%n]*3.6),0),
        vout_kmh=round(float(v[(p[-1]+int(10/(np.mean(dsm))))%n]*3.6),0),
        len_m=round(float(dsm[p].sum()),0),
        pos=[float(P[ap,0]),float(P[ap,1])], side=round(float(a[ap]/ (h[ap]) ),2)))
for c in corners: print(c)
np.savez('line.npz',P=P,k=k,v=v,s_px=s_px,dsm=dsm,Ps=Ps)
json.dump(dict(corners=corners,SC=SC,Lm=float(L*SC),lap=float(lap)),open('corners.json','w'))
ov=s.copy()
cv2.polylines(ov,[P.astype(np.int32).reshape(-1,1,2)],True,(0,255,255),2)
for c in corners:
    x,y=int(c['pos'][0]),int(c['pos'][1])
    cv2.circle(ov,(x,y),6,(0,0,255),-1)
    cv2.putText(ov,f"{c['id']}{c['dir']}",(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
cv2.imwrite('corners_ov.png',ov)
