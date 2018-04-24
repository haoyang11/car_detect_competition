# -*- coding: UTF-8 -*-
import json
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import os

# test_path 是处理后结果
gt_path="valid" # 标准结果
testpath="bbvote_pre" #处理结果
imgpath= "edistance"  #图片存储路径

# gt_path="valid" # 标准结果 shuru
# testpath="tmp_1" #处理结果
# imgpath="tmp1_img"  #图片存储路径


def data_reader(gtjson,prejson):
	imgfilename=os.path.join(imgpath,gtjson.split(".")[0]+".jpg") 
	gtjson=os.path.join(gt_path,gtjson)
	prejson=os.path.join(testpath,prejson)
	with open(gtjson,"r+") as f:
		gtjson = json.load(f)
		gtframe = gtjson["frame_data"]
		gtnum = len(gtframe)

	with open(prejson,"r+") as f:
		testjson = json.load(f)
		testframe = testjson["frame_data"]
		# areaframe = testjson["arealist"]
		testnum = len(testframe)
		# areatnum = len(areaframe)
	# print(testnum)
	# print(areatnum)

	if testnum != gtnum:
		return 
	data = np.zeros((4,testnum),dtype=float)
	for i in range(testnum):
		data[0][i]=gtframe[i]['vx']
		data[1][i]=gtframe[i]['x']
		data[2][i]=testframe[i]['vx']
		data[3][i]=testframe[i]['x']
	process_data=np.zeros((2,testnum),dtype=float)
	process_data[0]=data[0]-data[2] # diff v
	process_data[1]=(data[1]-data[3])/data[1] # diff x


	print("mean_err[v,x]:", np.mean(np.abs(process_data), axis=1))
	plt.figure()
	plt.subplot(211) 
	plt.title('pos(gt_pre)')
	plt.legend(process_data[1])
	plt.plot(process_data[1],'g-',label='diffx',markersize=20)
	plt.subplot(212) 
	plt.title('speed_pre')
	plt.legend(process_data[0])
	plt.plot(process_data[0],'y-',label='diffv',markersize=20)
	plt.savefig(imgfilename)

	# print("diffx:")
	# print(np.argmax((process_data[1])))
	# print(process_data[1])

	# print("diffv:")
	# print(np.argmax((process_data[0])))
	# print(process_data[0])






if __name__ == '__main__':
	files_gt= os.listdir(gt_path)
	# print(files_gt)

	isExists=os.path.exists(imgpath)
	if not isExists:
		os.makedirs(imgpath)

	for i in files_gt:
		if i.find(".avi")<0:
			continue
		if i.find("效果")>-1:
			continue

		prejson=i.split(".")[0]+"_pre.json"
		print(prejson)
		injson=i.split(".")[0]+"_gt.json"
		data_reader(injson,prejson)
