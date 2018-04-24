# -*- coding: UTF-8 -*-
import json
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import os
from scipy import optimize

# test_path 是处理后结果
gt_path="24gt" # 标准结果
testpath="24area" #处理结果
imgpath= "edistance"  #图片存储路径

# gt_path="valid" # 标准结果 shuru
# testpath="tmp_1" #处理结果
# imgpath="tmp1_img"  #图片存储路径


def data_reader(gtjson,prejson,x_data,r_data):
	imgfilename=os.path.join(imgpath,gtjson.split(".")[0]+"_c.jpg") 
	gtjson=os.path.join(gt_path,gtjson)
	prejson=os.path.join(testpath,prejson)
	with open(gtjson,"r+") as f:
		gtjson = json.load(f)
		gtframe = gtjson["frame_data"]
		gtnum = len(gtframe)

	with open(prejson,"r+") as f:
		testjson = json.load(f)
		testframe = testjson["frame_data"]
		testnum = len(testframe)
		

	if testnum != gtnum:
		return 
	data = np.zeros((4,testnum),dtype=float)
	for i in range(testnum):
		data[0][i]=gtframe[i]['vx']
		data[1][i]=gtframe[i]['x']
		data[2][i]=testframe[i]['vx']
		data[3][i]=testframe[i]['x']
	process_data=np.zeros((2,testnum),dtype=float)
	process_data[0]=data[1] # diff v
	process_data[1]=(data[1]-data[3]) # diff x
	

	print("mean_err[v,x]:", np.mean(np.abs(process_data), axis=1))
	plt.figure()
	plt.subplot(211) 
	plt.title('pos_err(abs)')
	# plt.legend(process_data[1])
	plt.plot(process_data[1],'g-',label='diffx',markersize=20)
	plt.subplot(212) 
	plt.title('pos_value')
	# plt.legend(process_data[0])
	plt.plot(data[3],'y-',label='pos_value',markersize=20)
	plt.savefig(imgfilename)
	# np.append(rdata,process_data[0])
	x_data+=list(process_data[0])
	r_data+=list(process_data[1])
	# print("diffx:")
	# print(np.argmax((process_data[1])))
	# print(process_data[1])

	# print("diffv:")
	# print(np.argmax((process_data[0])))
	# print(process_data[0])






if __name__ == '__main__':
	files_gt= os.listdir(gt_path)
	#print(files_gt)

	isExists=os.path.exists(imgpath)
	if not isExists:
		os.makedirs(imgpath)
	# all_data=np.zeros((1,0),dtype=float)
	x_data=[]

	err_data=[]
	for i in files_gt:
		if i.find(".avi")<0:
			continue
		if i.find("效果")>-1:
			continue
		prejson=i.split(".")[0]+"_pre.json"
		injson=i.split(".")[0]+"_gt.json"
		print(prejson)
		# np.append(all_data,data_reader(i,prejson))
		data_reader(injson,prejson,x_data,err_data)
	print(len(x_data))
	#fit curve
	# k,b=optimize.curve.fit()
	all_flag=True
	if all_flag:
		alldata=[err_data,x_data]
		alldata=np.array(alldata)
		print("shape:",alldata.shape)
		alldata=alldata.T[alldata.T[:,1].argsort()].T
		# print(alldata[1][:10])
		pre=5000
		tail=len(alldata[0])
		xdata=alldata[1][pre:tail]
		edata=alldata[0][pre:tail]
		for j in range(len(xdata)):
			if xdata[j]>10:
				print("pos10:",j)
				break
		for j in range(len(xdata)):
			if xdata[j]>18:
				print("pos18:",j)
				break


		print("min:",min(xdata))
		fitline=np.polyfit(xdata,edata,1)
		print("fitline_all:",fitline)
		x=np.arange(min(xdata),max(xdata)+20,0.01)
		y=np.polyval(fitline,x)

		# other 
		lineparam=[0.14728058, -2.30329372]
		y1=np.polyval(lineparam,x)




		plt.figure()
		plt.title('distance_error')
		plt.plot(xdata,edata,'go',label='diffx',markersize=5)
		plt.plot(x,y,'r-',label='fitline')
		plt.plot(x,y1,'y-',label='fitline1')
		imgfilename=os.path.join(imgpath,"final.jpg") 
		plt.savefig(imgfilename)
		plt.show()
	else:
		N=1500
		alldata=[x_data,err_data]
		# alldata.sort(key=lamda x:x[1])
		zipdata=zip(x_data,err_data)
		sorted(zipdata)
		[xdata,edata]=zip(*zipdata)
		print(xdata[:10])
		fitline=np.polyfit(xdata,edata,1)
		print("fitline_short:",fitline)
		x1=np.arange(min(xdata),max(xdata),0.01)
		y1=np.polyval(fitline,x1)
		fitline=np.polyfit(xdata,edata,1)
		print("fitline_long:",fitline)
		x2=np.arange(min(alldata[1][N:]),max(alldata[1][N:]),0.01)
		y2=np.polyval(fitline,x2)
		plt.figure()
		plt.title('distance_error')
		plt.plot(x_data,err_data,'go',label='diffx',markersize=5)
		plt.plot(x1,y1,'r-',label='fitline')
		plt.plot(x2,y2,'b-',label='fitline')
		imgfilename=os.path.join(imgpath,"final.jpg") 
		plt.savefig(imgfilename)
		# plt.show()