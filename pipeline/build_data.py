import numpy as np, json, cv2, base64
g=np.load('geom.npz'); C=g['C']; N=g['N']; a=g['a0']; h=g['h']
d=np.load('line.npz'); P=d['P']; v=d['v']
F=json.load(open('final.json')); SC=F['SC']; n=len(P)
cor=F['corners']
# bounds & crop
Lp=C+N*h[:,None]; Rp=C-N*h[:,None]
allp=np.vstack([Lp,Rp]); pad=60
x0,y0=np.floor(allp.min(0)-pad).astype(int); x1,y1=np.ceil(allp.max(0)+pad).astype(int)
x0=max(x0,0);y0=max(y0,0);x1=min(x1,1919);y1=min(y1,1079)
W,H=x1-x0,y1-y0; print('crop',x0,y0,W,H)
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
crop=s[y0:y1,x0:x1]
ok,buf=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,72]); img='data:image/jpeg;base64,'+base64.b64encode(buf).decode()
print('img kb',len(img)//1024)
def r1(x): return [round(float(t),1) for t in x]
st3=np.arange(0,n,3)
cx=r1(C[st3,0]-x0); cy=r1(C[st3,1]-y0); nx=[round(float(t),3) for t in N[st3,0]]; ny=[round(float(t),3) for t in N[st3,1]]; hh=r1(h[st3])
st2=np.arange(0,n,2)
rx=r1(P[st2,0]-x0); ry=r1(P[st2,1]-y0); rv=[int(round(float(t*3.6))) for t in v[st2]]
tang=lambda i:(lambda t:t/np.linalg.norm(t))(P[(i+1)%n]-P[(i-1)%n])
def tick(i,ext=None):
    t=C[(i+1)%n]-C[(i-1)%n]; t/=np.linalg.norm(t); nn=np.array([-t[1],t[0]]); hw=h[i]
    p1=C[i]+nn*hw; p2=C[i]-nn*hw
    return r1([p1[0]-x0,p1[1]-y0,p2[0]-x0,p2[1]-y0])
cs=[]
allc=C
for c in cor:
    ap=c['apex']; dirn=1 if c['dir']=='R' else -1
    ti,ex=c['turn_in'],c['exit']; arc=(ex-ti)%n or 1
    pre=(ti-15)%n; post=(ex+10)%n
    def lab(sv):
        return 'outside' if sv<-0.25 else ('inside' if sv>0.25 else 'mid-track')
    sd=lambda i:float(a[i]/h[i]*dirn)
    frac=((ap-ti)%n)/arc
    # badge position: pick side with more clearance from other track
    best=None
    for sgn in (1,-1):
        t=C[(ap+1)%n]-C[(ap-1)%n]; t/=np.linalg.norm(t); nn=np.array([-t[1],t[0]])
        p=C[ap]+sgn*nn*(h[ap]+22)
        idx=np.arange(n); far=np.minimum((idx-ap)%n,(ap-idx)%n)>70
        dmin=np.linalg.norm(C[far]-p,axis=1).min()
        if best is None or dmin>best[0]: best=(dmin,p)
    bp=best[1]
    vb=None; drop=0
    if c['brake'] is not None:
        vb=float(v[c['brake']]*3.6); drop=vb-c['v_apex']
    typ='Hairpin' if c['r_m']<6 else ('Tight' if c['r_m']<10 else ('Medium' if c['r_m']<20 else 'Fast'))
    cs.append(dict(id=c['id'],dir=c['dir'],type=typ,r=c['r_m'],va=c['v_apex'],
      drop=round(drop),brk=('Hard brake' if drop>=20 else ('Brake' if drop>=8 else 'Lift')),
      deg=c['turn_deg'],arc=c['arc_m'],
      sIn=round(sd(ti),2),sAp=round(sd(ap),2),sOut=round(sd(ex),2),
      apexAt='early' if frac<0.4 else ('late' if frac>0.6 else 'mid'),
      i0=ti//2,ia=ap//2,i1=(ex//2) if ex>=ti else (ex//2)+len(st2),
      badge=r1([bp[0]-x0,bp[1]-y0]), apexPos=r1([P[ap,0]-x0,P[ap,1]-y0]),
      tTurn=tick(ti),tApex=tick(ap),tExit=tick(ex),tBrake=tick(c['brake']) if c['brake'] is not None else None))
data=dict(W=int(W),H=int(H),cx=cx,cy=cy,nx=nx,ny=ny,hh=hh,rx=rx,ry=ry,rv=rv,corners=cs,
          lenM=round(F['Lm']),widthM=7,sf=r1([C[0,0]-x0,C[0,1]-y0]),sfTick=tick(0))
json.dump(data,open('data.json','w'))
json.dump(img,open('img.json','w'))
print(len(json.dumps(data))//1024,'kb data')
for c in cs: print(c['id'],c['dir'],c['type'],c['r'],c['va'],c['brk'],c['drop'],c['entry'],c['apex'],c['exit'],c['apexAt'],c['i0'],c['i1'],len(st2))
