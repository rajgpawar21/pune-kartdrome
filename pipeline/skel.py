import cv2, numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
s=cv2.imread('/mnt/user-data/uploads/Satellite_Image.png')
m=cv2.imread('mask6.png',0)>0
# remove tiny comps, keep big
lab=label(m)
sizes=np.bincount(lab.ravel()); sizes[0]=0
keep=np.isin(lab,np.where(sizes>8000)[0])
print('components kept',[ (i,sizes[i]) for i in np.where(sizes>8000)[0]])
sk=skeletonize(keep)
cv2.imwrite('keep.png',keep.astype(np.uint8)*255)
ov=s.copy()
ov[cv2.dilate(sk.astype(np.uint8),np.ones((3,3)))>0]=(0,255,255)
cv2.imwrite('skel_ov.png',ov)
np.save('sk.npy',sk)
