import numpy as np, json, cv2
from scipy import ndimage as ndi
g=np.load('geom.npz'); C=g['C']; N=g['N']; a=g['a0']; h=g['h']
d=np.load('line.npz'); P=d['P']; k=d['k']; v=d['v']; dsm=d['dsm']
meta=json.load(open('corners.json')); SC=meta['SC']
n=len(P)
kk=ndi.gaussian_filter1d(np.abs(k)/SC,5,mode='wrap')
sg=np.sign(ndi.gaussian_filter1d(k,5,mode='wrap'))
apex_idx=[213,260,315,472,523,646,730,842,930,1016,1191,1243]
apex_idx.append(1399)
cum=np.r_[0,np.cumsum(dsm)[:-1]]   # metres
def circ(i): return i%n
corners=[]
for j,ap in enumerate(apex_idx):
    # if double peak 1243/1265 pick lower speed apex
    if ap==1243:
        ap=1243 if kk[1243]>=kk[1265] else 1265
    prev=apex_idx[j-1]; nxt=apex_idx[(j+1)%len(apex_idx)]
    kap=kk[ap]; s0=sg[ap]
    # extent backward
    i=ap
    while True:
        i2=circ(i-1)
        if (kk[i2]<max(0.4*kap,1/70.) ) or sg[i2]!=s0 or ((ap-i2)%n)>110 or i2==prev: break
        i=i2
    tin=i
    i=ap
    while True:
        i2=circ(i+1)
        if (kk[i2]<max(0.4*kap,1/70.)) or sg[i2]!=s0 or ((i2-ap)%n)>110 or i2==nxt: break
        i=i2
    tout=i
    dirn=1 if s0>0 else -1
    # lateral fractions: +ve = towards inside of corner
    def side(ix): return float(a[ix]/h[ix]*dirn)
    # braking: decel mask
    acc=(np.roll(v,-1)**2-v**2)/(2*np.roll(dsm,0))
    br=None
    i=tin; lim_back=0
    # search backward from tin for contiguous decel < -2.5
    j0=circ(tin); 
    idx=[]
    ii=ap
    while ((ap-ii)%n)<260:
        if acc[ii]<-2.5: idx.append(ii)
        ii=circ(ii-1)
    brk=None
    if idx:
        idx=sorted(idx,key=lambda q:(ap-q)%n)   # nearest first
        # contiguous chain from apex backwards
        chain=[idx[0]]
        for q in idx[1:]:
            if (chain[-1]-q)%n<=4: chain.append(q)
            else: break
        if (ap-chain[0])%n<40: brk=chain[-1]
    corners.append(dict(id=j+1,apex=int(ap),turn_in=int(tin),exit=int(tout),dir='R' if dirn>0 else 'L',
      r_m=round(float(1/kap),1),v_apex=round(float(v[ap]*3.6)),
      v_entry=round(float(v[tin]*3.6)),v_exit=round(float(v[tout]*3.6)),
      brake=None if brk is None else int(brk),
      brake_m=None if brk is None else round(float((cum[ap]-cum[brk])%meta['Lm'])),
      side_in=round(side(tin),2),side_apex=round(side(ap),2),side_exit=round(side(tout),2),
      pos=[float(P[ap,0]),float(P[ap,1])], arc_m=round(float((cum[tout]-cum[tin])%meta['Lm']))))
for c in corners: print(c)
# turn angle
for c in corners:
    ii=np.arange(c['turn_in'],c['turn_in']+((c['exit']-c['turn_in'])%n)+1)%n
    dpx=np.linalg.norm(np.roll(P,-1,0)-P,axis=1)
    c['turn_deg']=round(float(abs(np.degrees((k[ii]*dpx[ii]).sum()))))
    print(c['id'],c['turn_deg'])
json.dump(dict(corners=corners,SC=SC,Lm=meta['Lm'],lap=meta['lap']),open('final.json','w'))
