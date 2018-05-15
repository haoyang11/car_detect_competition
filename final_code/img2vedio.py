import glob as gb
import cv2
import shutil
import os
# path="newjpg"
img_path = gb.glob("img\\*.jpg") 
videoWriter = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (640,480))
# for i in  range(len(img_path)):
# 	# print len(img_path[i])
# 	# print img_path[i][0:6]
# 	if len(img_path[i])==12:
# 		newname=img_path[i][0:6]+'0'+img_path[i][6:]
# 	if len(img_path[i])==11:
# 		newname=img_path[i][0:6]+'00'+img_path[i][6:]
# 	shutil.move(img_path[i], os.path.join(newname))
# 	# if i>20:
# 	# 	break
# print(img_path)

# img_path.sort()
# print(img_path)

for path in img_path:
    # print(len(path))
    img  = cv2.imread(path) 
    img = cv2.resize(img,(640,480))
    videoWriter.write(img)
print "finish"