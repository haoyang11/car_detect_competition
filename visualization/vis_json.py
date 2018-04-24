# -*- coding: UTF-8 -*-
import json
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import os

# test_path 是处理后结果
gt_path="test_pre" # 标准结果
testpath="24valid" #处理结果
imgpath="24vjson-1"  #图片存储路径
# gt_path="test_pr`:e" # 标准结果
# testpath="bbvote_pre" #处理结果
# imgpath="testfinal"  #图片存储路径

def data_reader(gtjson,prejson):
	imgfilename=os.path.join(imgpath,gtjson.split(".")[0]+".jpg") 
	gtjson=os.path.join(gt_path,gtjson)
	prejson=os.path.join(testpath,prejson)
	with open(gtjson,"r+") as f:
		gtjson = json.load(f)
		gtframe = gtjson["frame_data"]
		gtnum = len(gtframe)


	data = np.zeros((2,gtnum),dtype=float)
	for i in range(gtnum):
		data[0][i]=gtframe[i]['vx']
		data[1][i]=gtframe[i]['x']



	# print("mean_err[v,x]:", np.mean(np.abs(process_data), axis=1))
	plt.figure()
	plt.subplot(211) 
	plt.title('pos(pre)')
	plt.plot(data[0],'g-',label='vx',markersize=20)
	plt.subplot(212) 
	plt.title('speed(pre)')
	plt.plot(data[1],'y-',label='x',markersize=20)
	# plt.show()
	plt.savefig(imgfilename)





if __name__ == '__main__':
	files_gt= os.listdir(gt_path)
	print(files_gt)

	isExists=os.path.exists(imgpath)
	if not isExists:
		os.makedirs(imgpath)

	for i in files_gt:
		if i.find("json")<0:
			continue
		prejson=i.split(".")[0]+"_pre.json"
		data_reader(i,prejson)
