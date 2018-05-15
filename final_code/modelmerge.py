#from __future__ import print_function
import cv2
import numpy as np
import sys
import os
from projection import ground_point_to_bird_view_proj
import json
from pre import Filter
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from array import array
from scipy import interpolate  

FRAME_TIME = 0.05023106289
Y_THRESHOLD = 0.3 #0.2

arealist_1= "new\\qwe56-valid" # bbox
# arealist_2= "new\\s9" # bbox
arealist_2="0509final\\six-valid-900"
outpath= arealist_2+"_json"
imgpath= arealist_1+"_img"


def smooth_curve(x_result,N):
  plen=N
  head = [x_result[0] for z in range(plen)] 
  tail = [x_result[-1] for z in range(plen)] 
  pad_result= head+x_result+tail
  # pad_result= np.concatenate(head,x_result)
  xm_result=[]
  for i in range(len(x_result)):
    xm_result.append(np.mean(pad_result[i:i+plen+1]))
  xm_result.reverse()
  head = [xm_result[0] for z in range(plen)] 
  tail = [xm_result[-1] for z in range(plen)] 
  pad_result= head+xm_result+tail
  # pad_result= np.concatenate(head,x_result,tail)
  hx_result=[]
  for i in range(len(x_result)):
    hx_result.append(np.mean(pad_result[i:i+plen+1]))
  hx_result.reverse()

  return  hx_result

def process_v(v_result):
  N=50
  v_process=[]
  x=range(N)
  for i in range(0,len(v_result),N):
    # print(len(x))
    y=v_result[i:i+N]
    # print(len(y))
    line_p= np.polyfit(x, y, 1)
    pre=np.polyval(line_p,x)
    v_process+=list(pre)
  # print(len(v_process))
  return v_process

def select_v(v_result):
  N=10
  v_process=v_result[:]
  for i in range(len(v_result)-N):
    x=np.array(v_result[i:i+N])
    M=np.mean(x)
    # print(np.abs(x-M))
    sigma=np.mean(np.abs(x-M))
    ratio=1
    idx=[]
    remain=[]
    for j in range(i,i+N):
      if np.abs(v_process[j]-M)>sigma*ratio:
        idx.append(j)
        v_process[j]=0
      else:
        remain.append(v_process[j])
    if len(remain)>1:
      m_c=np.mean(remain)
    else:
      m_c=M

    for j in idx:
      v_process[j]=m_c
      
  # v_process=smooth_curve(list(v_process),10)
  return v_process 


def  fitfun(x,a,b):
  return a/x+b

def cal_sigma(x_result,N):
  plen=N
  head = [x_result[0] for z in range(plen)] 
  tail = [x_result[-1] for z in range(plen)] 
  pad_result= head+x_result+tail
  # pad_result= np.concatenate(head,x_result)
  xm_result=[]
  for i in range(len(x_result)):
    xm_result.append(np.abs((x_result[i]-np.mean(pad_result[i:i+plen+1]))))
  xm_result[0]=1
  xm_result[-1]=1
  return xm_result
def cal_sigma(x_result,N):
  plen=N
  head = [x_result[0] for z in range(plen)] 
  tail = [x_result[-1] for z in range(plen)] 
  pad_result= head+x_result+tail
  # pad_result= np.concatenate(head,x_result)
  xm_result=[]
  for i in range(len(x_result)):
    xm_result.append(np.abs((x_result[i]-np.mean(pad_result[i:i+plen+1]))))
  xm_result[0]=1
  xm_result[-1]=1
  return xm_result

def run(k1, k2,areapath, N,imgfilename,areapath_sum,outfile):
  global tot
  global myxmin
  global myxmax
  global myymax
  global myymin
  with open("camera_parameter.json") as f:
    cam_param = json.load(f)
  cam_param["camera_height"] *=1 
  with open(areapath) as f:
    arealist = json.load(f)
  x_result = []
  vx_result = []
  h_param=[]
  w_param=[]
  y_result=[]
  for bbox in arealist['arealist']:
    x, y = ground_point_to_bird_view_proj((bbox['xmax'] + bbox['xmin'])/2, bbox['ymax'], cam_param)
    h_param.append(bbox['xmax']-bbox['xmin'])
    w_param.append(bbox['ymax']-bbox['ymin'])
    x_result.append(x)
    y_result.append(y)
  finaldata=x_result

  for i in range(1,len(h_param)-1,2):
    h_param[i]=(float(h_param[i+1])+float(h_param[i+1]))/2
    w_param[i]=(float(w_param[i+1])+float(w_param[i+1]))/2

  with open(areapath_sum) as f:
    arealist = json.load(f)
  sx_result = []
  svx_result = []
  sh_param=[]
  sw_param=[]
  sy_result=[]
  for bbox in arealist['arealist']:
    x, y = ground_point_to_bird_view_proj((bbox['xmax'] + bbox['xmin'])/2, bbox['ymax'], cam_param)
    sh_param.append(bbox['xmax']-bbox['xmin'])
    sw_param.append(bbox['ymax']-bbox['ymin'])

    sx_result.append(x)
    sy_result.append(y)
  for i in range(1,len(sh_param)-1,2):
    sh_param[i]=(float(sh_param[i+1])+float(sh_param[i+1]))/2
    sw_param[i]=(float(sw_param[i+1])+float(sw_param[i+1]))/2




  lineparam=[ 4.29128904e+03, -1.99320880e+00] # new 05
  lineparam=[ 4.29169018e+03, -2.00891006e+00] # 24 valid

  fitd_w=[fitfun(i,lineparam[0],lineparam[1]) for i in h_param]
  lineparam=[ 3.33988934e+03, -2.55092719e+00] # new 05
  ineparam=[ 3.33810684e+03, -2.61628836e+00] # 24 valid

  fitd_h=[fitfun(i,lineparam[0],lineparam[1]) for i in w_param]
  #weight
  fitd_o=[]
  thre=0.08
  count=0
  for i in range(len(fitd_h)):
    ratio=0.8
    if np.abs(fitd_w[i]-fitd_h[i])>fitd_w[i]*thre:
      ratio=1
      count=count+1
    fitd_o.append((1-ratio)*fitd_h[i]+ratio*fitd_w[i])
  print(count)

  # fitd_o=[0.2*fitd_h[i]+0.8*fitd_w[i] for i in range(len(fitd_w))]


  lineparam=[ 4.29459193e+03, -2.02106812e+00]
  lineparam=[ 4.30394610e+03, -2.08353087e+00] #6-9
  fitd_sw=[fitfun(i,lineparam[0],lineparam[1]) for i in sh_param]
  lineparam=[ 3.34840932e+03, -2.42931520e+00] # new 05
  lineparam=[ 3.33043890e+03, -2.44178226e+00] # 6-9
  fitd_sh=[fitfun(i,lineparam[0],lineparam[1]) for i in sw_param]
  # fitd_s=[0.2*fitd_sh[i]+0.8*fitd_sw[i] for i in range(len(fitd_sw))]

  fitd_s=[]
  thre=0.08
  count=0
  for i in range(len(fitd_sh)):
    ratio=0.8
    if np.abs(fitd_sw[i]-fitd_sh[i])>fitd_sw[i]*thre:
      ratio=1
      count=count+1
    fitd_s.append((1-ratio)*fitd_sh[i]+ratio*fitd_sw[i])
  print(count)

  # fitd_s=[0.2*fitd_h[i]+0.8*fitd_w[i] for i in range(len(fitd_w))]



  ratio=0.6  # 0.65
  # fitd=fitd_o[:]
  fitd=[ratio*fitd_o[i]+(1-ratio)*fitd_s[i] for i in range(len(fitd_sw))]




  fitd_m=smooth_curve(list(fitd),10)
  if np.mean(fitd)>20:
    # print(30)
    fitd_m=smooth_curve(list(fitd),30)
  else:
    # print(10)
    fitd_m=smooth_curve(list(fitd),10)
  if np.mean(fitd)>20:
    # print(40)
    fitds=smooth_curve(list(fitd),40)
  else:
    # print(15)
    fitds=smooth_curve(list(fitd),15)
  # fitds=fitd_m[:]
  # print(np.mean(y_result))

  # x_result=fitds
  x_result=fitds[:]
  fix_frame_time = True
  time_stamp=[]
  if fix_frame_time:
    for fid, (x1, x2) in enumerate(zip(x_result[2:], x_result[:-2]), 1):
      vx_result += [(x1 - x2) / (2 * FRAME_TIME)]
  else:
    for fid, (x1, x2) in enumerate(zip(x_result[2:], x_result[:-2]), 1):
      vx_result += ([(x1 - x2) / (time_stamp[fid] - time_stamp[fid - 1])])

  # vx_result = vx_result + [vx_result[-1]]
  vx_result = [vx_result[0]]+vx_result + [vx_result[-1]]


  v_result=smooth_curve(list(vx_result),10)
  final_v=process_v(vx_result)
  sv=select_v(vx_result)
  # vx_result=vx_result.append(vx_result[-1])



  frame_data = []
  xm_result=fitd_m
  for fid, (vx, x) in enumerate(zip(v_result, xm_result)):
    frame_data += [{"vx": vx, "x": x, "fid": fid}]
  frame_id=len(v_result)-1
  result = {
    "frame_data": frame_data,
    "end_frame": frame_id,
    "start_frame": 0,
    "track_id": None,
    "arealist": arealist['arealist'],
  }
  
  outjson=os.path.join(outfile)
  with open(outjson, "w") as output_file:
    json.dump(result, output_file) 

  return False

if __name__ == '__main__':
  # gtvalid = sys.argv[1] # gt
  # arealist = sys.argv[2] # bbox
  # time_stamp = sys.argv[3] # time
  
  # time_stamp = sys.argv[3] # time

  gtfile = os.listdir(arealist_1)
  # areafile = os.listdir(arealist)
  isExists=os.path.exists(imgpath)
  if not isExists:
    os.makedirs(imgpath)
  isExists=os.path.exists(outpath)
  if not isExists:
    os.makedirs(outpath)
  # timefile = os.listdir(time_stamp)
  k1=1e-5
  k2=1
  N=28
  print(gtfile)
  myxmin=1000
  myxmax=0
  myymax=0
  myymin=1000
  result=[]
  for i in range(len(gtfile)):
    areapath = os.path.join(arealist_1, gtfile[i])
    areapath_sum = os.path.join(arealist_2, gtfile[i])
    output=os.path.join(outpath, gtfile[i])
    imgfilename=os.path.join(imgpath,gtfile[i].split(".")[0]+".jpg")

    print(gtfile[i])

    tmp = run(pow(0.1,k1), pow(0.1,k2),areapath,N,imgfilename,areapath_sum,output)
    result.append(tmp)

print(np.mean(result,axis=0))
# print(myxmin,myxmax,myymin,myymax)