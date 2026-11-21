"""Recompute the racing line and corner analysis for both directions of travel.

For each direction: start from the smoothest (minimum-curvature) path, then refine it
to minimise a quasi-steady-state lap time (grip-limited corner speeds, finite braking
and acceleration), so braking and acceleration asymmetry changes the line.
Inputs: geom.npz (centerline C, normals N, half-width h, limits lim, min-curvature offsets a0).
Output: data.json used by the page.
"""
import numpy as np, json, math, time, sys
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import find_peaks

g=np.load('geom.npz'); C0=g['C']; N0=g['N']; h0=g['h']; lim0=g['lim']; a00=g['a0']
n=len(C0)
SC=7.0/(2*np.median(h0))          # m per px (assumes 7 m wide track)
ALAT,VMAX,AACC,ABR=11.0,17.0,4.0,9.0   # m/s^2, m/s, m/s^2, m/s^2

def dirdata(rev):
    idx=np.arange(n) if not rev else (-np.arange(n))%n
    s=-1.0 if rev else 1.0
    return dict(C=C0[idx],N=s*N0[idx],h=h0[idx],lim=lim0[idx],a=s*a00[idx])

def curv(P):
    Ps=np.c_[ndi.gaussian_filter1d(P[:,0],3,mode='wrap'),ndi.gaussian_filter1d(P[:,1],3,mode='wrap')]
    d1=(np.roll(Ps,-1,0)-np.roll(Ps,1,0))/2; d2=np.roll(Ps,-1,0)-2*Ps+np.roll(Ps,1,0)
    k=(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/np.maximum((d1**2).sum(1),1e-9)**1.5
    ds=np.linalg.norm(np.roll(Ps,-1,0)-Ps,axis=1)
    return ndi.gaussian_filter1d(k,2,mode='wrap'),ds

def speed(P):
    k,ds=curv(P); kk=np.abs(k)/SC; dsm=ds*SC
    vcap=np.minimum(VMAX,np.sqrt(ALAT/np.maximum(kk,1e-6)))
    vc=vcap.tolist(); dl=dsm.tolist(); v=vc[:]
    sq=math.sqrt
    for _ in range(2):
        for i in range(2*n):
            j=i%n; jn=(j+1)%n
            t=sq(v[j]*v[j]+2*AACC*dl[j])
            if t<v[jn]: v[jn]=t
        for i in range(2*n):
            j=(-i)%n; jp=(j-1)%n
            t=sq(v[j]*v[j]+2*ABR*dl[jp])
            if t<v[jp]: v[jp]=t
    v=np.array(v)
    return k,v,dsm,float(np.sum(dsm/v))

def optimise(D,M=150,iters=50):
    """Refine the minimum-curvature path: optimise a smooth lateral correction c (M control points)."""
    C,N,lim,a0=D['C'],D['N'],D['lim'],D['a']
    ctrl=np.linspace(0,n,M,endpoint=False)
    cs_x=np.r_[ctrl,n]
    B=np.zeros((n,M))
    for m in range(M):
        y=np.zeros(M+1); y[m]=1
        if m==0: y[M]=1
        B[:,m]=CubicSpline(cs_x,y,bc_type='periodic')(np.arange(n))
    def a_of(c): return np.clip(a0+B@c,-lim,lim)
    def P_of(c): return C+N*a_of(c)[:,None]
    def f(c):
        k,v,dsm,T=speed(P_of(c))
        return T+2e-5*np.sum(np.diff(c,2)**2)
    T0=speed(P_of(np.zeros(M)))[3]
    t0=time.time()
    r=minimize(f,np.zeros(M),method='L-BFGS-B',bounds=[(-14,14)]*M,options=dict(maxiter=iters,eps=0.4,maxfun=iters*(M+2)))
    T1=speed(P_of(r.x))[3]
    if T1>T0: r.x=np.zeros(M); T1=T0
    print('  smooth-line lap %.3fs -> refined %.3fs  (%.1fs, %d it)'%(T0,T1,time.time()-t0,r.nit),flush=True)
    return a_of(r.x),T0,T1

def tick(C,h,i):
    t=C[(i+1)%n]-C[(i-1)%n]; t/=np.linalg.norm(t); nn=np.array([-t[1],t[0]])
    p1=C[i]+nn*h[i]; p2=C[i]-nn*h[i]
    return [round(float(q),1) for q in (p1[0],p1[1],p2[0],p2[1])]

def analyse(D,a,x0,y0):
    C,N,h=D['C'],D['N'],D['h']
    P=C+N*a[:,None]
    k,v,dsm,T=speed(P)
    kk=ndi.gaussian_filter1d(np.abs(k)/SC,5,mode='wrap')
    sg=np.sign(ndi.gaussian_filter1d(k,5,mode='wrap'))
    cum=np.r_[0,np.cumsum(dsm)[:-1]]; Lm=float(dsm.sum())
    kt=np.r_[kk,kk,kk]
    pk,pr=find_peaks(kt,height=1/25.,prominence=0.017)
    cand=sorted([(int(i-n),float(kt[i])) for i in pk if n<=i<2*n])
    # merge peaks closer than 25 samples (same corner), keep tighter apex
    apx=[]
    for i,kv in cand:
        if apx and i-apx[-1][0]<25:
            if kv>apx[-1][1]: apx[-1]=(i,kv)
        else: apx.append((i,kv))
    if len(apx)>1 and (apx[0][0]+n-apx[-1][0])<25:
        if apx[0][1]>=apx[-1][1]: apx.pop()
        else: apx.pop(0)
    apex=[i for i,_ in apx]
    acc=(np.roll(v,-1)**2-v**2)/(2*dsm)
    dpx=np.linalg.norm(np.roll(P,-1,0)-P,axis=1)
    cors=[]
    for j,ap in enumerate(apex):
        prev=apex[j-1]; nxt=apex[(j+1)%len(apex)]
        kap=kk[ap]; s0=sg[ap]
        def walk(step,stop):
            i=ap
            while True:
                i2=(i+step)%n
                if kk[i2]<max(0.4*kap,1/70.) or sg[i2]!=s0 or ((i2-ap)*step)%n>110 or i2==stop: break
                i=i2
            return i
        ti=walk(-1,prev); ex=walk(1,nxt)
        dirn=1 if s0>0 else -1
        sd=lambda i:float(a[i]/h[i]*dirn)
        arc=(ex-ti)%n or 1
        frac=((ap-ti)%n)/arc
        # braking zone: contiguous decel before apex
        brk=None; chain=[]
        for q in range(ap,ap-260,-1):
            q%=n
            if acc[q]<-2.5:
                if not chain or (chain[-1]-q)%n<=4: chain.append(q)
                else: break
        if chain and (ap-chain[0])%n<40: brk=chain[-1]
        vb=float(v[brk]*3.6) if brk is not None else None
        drop=(vb-float(v[ap]*3.6)) if vb is not None else 0.0
        # badge on the side with more clearance from other parts of the track
        best=None
        t=C[(ap+1)%n]-C[(ap-1)%n]; t/=np.linalg.norm(t); nn=np.array([-t[1],t[0]])
        far=np.minimum((np.arange(n)-ap)%n,(ap-np.arange(n))%n)>70
        for sgn in (1,-1):
            p=C[ap]+sgn*nn*(h[ap]+22); dmin=np.linalg.norm(C[far]-p,axis=1).min()
            if best is None or dmin>best[0]: best=(dmin,p)
        R=1/kap; typ='Hairpin' if R<6 else ('Tight' if R<10 else ('Medium' if R<20 else 'Fast'))
        ii=np.arange(ti,ti+arc+1)%n
        deg=abs(math.degrees(float((k[ii]*dpx[ii]).sum())))
        r1=lambda *xs:[round(float(q),1) for q in xs]
        cors.append(dict(dir='R' if dirn>0 else 'L',type=typ,r=round(float(R),1),va=int(round(float(v[ap]*3.6))),
            drop=int(round(drop)),brk='Hard brake' if drop>=20 else ('Brake' if drop>=8 else 'Lift'),
            deg=int(round(deg)),arc=int(round(float((cum[ex]-cum[ti])%Lm))),
            sIn=round(sd(ti),2),sAp=round(sd(ap),2),sOut=round(sd(ex),2),
            apexAt='early' if frac<0.4 else ('late' if frac>0.6 else 'mid'),
            ia=ap//2,i0=ti//2,i1=(ex//2) if ex>=ti else (ex//2)+len(range(0,n,2)),
            badge=r1(best[1][0]-x0,best[1][1]-y0),apexPos=r1(P[ap,0]-x0,P[ap,1]-y0),
            tTurn=[round(q,1) for q in np.array(tick(C,h,ti))-[x0,y0,x0,y0]],
            tApex=[round(q,1) for q in np.array(tick(C,h,ap))-[x0,y0,x0,y0]],
            tExit=[round(q,1) for q in np.array(tick(C,h,ex))-[x0,y0,x0,y0]],
            tBrake=None if brk is None else [round(q,1) for q in np.array(tick(C,h,brk))-[x0,y0,x0,y0]]))
    for i,c in enumerate(cors): c['id']=i+1
    st2=np.arange(0,n,2)
    netturn=float((k*dpx).sum())
    return dict(rx=[round(float(q),1) for q in P[st2,0]-x0],ry=[round(float(q),1) for q in P[st2,1]-y0],
        rv=[int(round(float(q*3.6))) for q in v[st2]],corners=cors,lapS=round(T,1),lenM=round(Lm),
        nR=sum(c['dir']=='R' for c in cors),nL=sum(c['dir']=='L' for c in cors),net=netturn,P=P,v=v)

if __name__=='__main__':
    # crop offsets used by the page (same as before)
    Lp=C0+N0*h0[:,None]; Rp=C0-N0*h0[:,None]; allp=np.vstack([Lp,Rp]); pad=60
    x0,y0=np.floor(allp.min(0)-pad).astype(int); x1,y1=np.ceil(allp.max(0)+pad).astype(int)
    x0=max(x0,0);y0=max(y0,0);x1=min(x1,1919);y1=min(y1,1079); W,H=int(x1-x0),int(y1-y0)
    out={}
    for rev in (False,True):
        D=dirdata(rev); print('reverse' if rev else 'forward',flush=True)
        a,T0,T1=optimise(D)
        R=analyse(D,a,x0,y0); R['T0']=round(T0,1)
        key='cw' if R['net']>0 else 'ccw'
        print('  net turning %.0f deg -> %s ; %d corners (%dR %dL) lap %.1fs'%(math.degrees(R['net']),key,len(R['corners']),R['nR'],R['nL'],R['lapS']),flush=True)
        np.savez('dir_%s.npz'%key,P=R.pop('P'),v=R.pop('v'),a=a)
        out[key]=R
    # shared geometry (direction independent)
    st3=np.arange(0,n,3); r1=lambda x:[round(float(t),1) for t in x]
    Dcw=dirdata(False)
    geo=dict(W=W,H=H,cx=r1(Dcw['C'][st3,0]-x0),cy=r1(Dcw['C'][st3,1]-y0),nx=[round(float(t),3) for t in Dcw['N'][st3,0]],
        ny=[round(float(t),3) for t in Dcw['N'][st3,1]],hh=r1(Dcw['h'][st3]),widthM=7)
    # start/finish: same physical point; heading differs per direction, tick is direction independent
    t=tick(Dcw['C'],Dcw['h'],0); geo['sfTick']=[round(float(q),1) for q in np.array(t)-[x0,y0,x0,y0]]
    geo['sf']=[round(float(Dcw['C'][0,0]-x0),1),round(float(Dcw['C'][0,1]-y0),1)]
    for k in out: out[k]['lenM']=int(out[k]['lenM'])
    json.dump(dict(geo=geo,**out),open('analysis.json','w'))
    print({k:(len(v['corners']),v['lapS'],v['T0']) for k,v in out.items()})
