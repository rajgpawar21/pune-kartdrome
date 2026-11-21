import cv2, numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
keep=cv2.imread('keep.png',0)>0
dt=ndi.distance_transform_edt(keep)
dts=cv2.GaussianBlur(dt.astype(np.float32),(0,0),3)
wp=[(1015,450),(1005,560),(985,660),(950,740),(910,810),(850,870),(780,915),(700,935),(630,925),(578,893),
(548,850),(535,800),(545,740),(555,680),(530,645),(500,640),(468,670),(440,730),(390,770),(320,760),(285,700),(285,620),(310,555),
(370,510),(440,495),(540,500),(600,530),(625,590),(625,700),(660,770),(715,778),(770,740),(775,700),(775,600),
(750,520),(710,450),(650,400),(580,385),(500,395),(440,410),(395,375),(380,320),(410,270),(470,230),(560,205),
(650,190),(730,205),(790,260),(805,330),(810,420),(830,490),(870,525),(920,510),(940,470),(925,390),(905,300),
(905,240),(930,190),(990,150),(1050,135),(1130,120),(1200,115),(1300,100),(1400,70),(1470,62),(1540,75),(1590,120),
(1620,170),(1630,240),(1590,290),(1540,305),(1500,275),(1440,240),(1350,230),(1250,235),(1150,225),(1070,225),
(1030,260),(1020,330)]
P=np.array(wp,float)
tck,u=splprep([P[:,0],P[:,1]],s=0,per=True)
def resample(tck,step=4.0):
    uu=np.linspace(0,1,20000); x,y=splev(uu,tck)
    d=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
    n=int(d[-1]/step); t=np.linspace(0,d[-1],n,endpoint=False)
    return np.c_[np.interp(t,d,x),np.interp(t,d,y)],d[-1]
C,total=resample(tck)
print('length px',total,len(C))
def normals(C):
    t=np.roll(C,-1,0)-np.roll(C,1,0); t/=np.linalg.norm(t,axis=1)[:,None]
    return np.c_[-t[:,1],t[:,0]]
offs=np.arange(-18,19,1.0)
for it in range(10):
    N=normals(C)
    X=C[:,0,None]+N[:,0,None]*offs[None,:]; Y=C[:,1,None]+N[:,1,None]*offs[None,:]
    V=ndi.map_coordinates(dts,[Y,X],order=1,mode='nearest')-0.02*np.abs(offs)[None,:]
    best=offs[np.argmax(V,1)]
    best=ndi.gaussian_filter1d(best,5,mode='wrap')
    C=C+N*best[:,None]
    C[:,0]=ndi.gaussian_filter1d(C[:,0],3,mode='wrap'); C[:,1]=ndi.gaussian_filter1d(C[:,1],3,mode='wrap')
np.save('center.npy',C)
w=2*ndi.map_coordinates(dt,[C[:,1],C[:,0]],order=1)
print('width px pctiles',np.percentile(w,[5,25,50,75,95]))
ov=s.copy()
pts=C.astype(np.int32).reshape(-1,1,2)
cv2.polylines(ov,[pts],True,(0,255,255),2)
cv2.circle(ov,tuple(C[0].astype(int)),8,(0,0,255),-1)
cv2.imwrite('center2_ov.png',ov)
