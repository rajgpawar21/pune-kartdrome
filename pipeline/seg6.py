import cv2, numpy as np
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
valid=~((s>245).all(2)); valid=cv2.erode(valid.astype(np.uint8),np.ones((25,25)))>0
L=cv2.cvtColor(s,cv2.COLOR_BGR2LAB)[...,0].astype(np.float32)
Ls=cv2.GaussianBlur(L,(0,0),3)
bg=cv2.GaussianBlur(np.where(valid,L,np.median(L[valid])),(0,0),45)
m=((Ls-bg<-7)&valid).astype(np.uint8)
# exclusion: buildings/green pitch, outside track area
excl=np.zeros_like(m)
for poly in [
 [(1180,440),(1520,440),(1520,760),(1180,760)],   # pitch+building
 [(1000,880),(1300,880),(1300,1080),(1000,1080)], # bottom right trees
 [(0,0),(560,0),(300,200),(0,420)],               # top-left grass
 [(300,950),(800,1080),(300,1080)],               # bottom-left bushes
 [(1650,0),(1920,0),(1920,300),(1700,300)],       # far right
]:
    cv2.fillPoly(excl,[np.array(poly,np.int32)],1)
m[excl>0]=0
m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
cv2.imwrite('mask6.png',m*255)
ov=s.copy(); ov[m>0]=(0.45*ov[m>0]+0.55*np.array([0,0,255])).astype(np.uint8)
cv2.imwrite('ov6.png',ov)
