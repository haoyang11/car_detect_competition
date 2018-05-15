#from __future__ import print_function
import cv2
import numpy as np
import sys
import os
from projection import ground_point_to_bird_view_proj
import json
from pre import Filter

FRAME_TIME = 0.05023106289
Y_THRESHOLD = 0.3 #0.2

def run(k1, k2, gtjson, areapath, timefile):
  global tot
  with open("/data/mcdc_data/valid/camera_parameter.json") as f:
    cam_param = json.load(f)
  cam_param["camera_height"] *=1 
  #with open('/data/mcdc_data/valid/valid_video_00_time.txt') as f:
  with open(timefile) as f: # TODO: modify it: video_00
    time_stamp = [line[:-1] for line in f]
  time_stamp = map(eval, time_stamp)
	
  with open(areapath) as f:
    arealist = json.load(f)
  x_result = []
  vx_result = []
  for bbox in arealist['arealist']:
    x, y = ground_point_to_bird_view_proj((bbox['xmax'] + bbox['xmin'])/2, bbox['ymax'], cam_param)
    x_result.append(x)
  
  xm_result = [x_result[0]] + [(0.25* x0 + 0.5 * x1 + 0.25 * x2) for x0, x1, x2 in zip(x_result[2:], x_result[1:-1], x_result[:-1])] + [x_result[-1]]
  fitparam = [ 0.11502782, -2.15750625]
  bias = np.polyval(fitparam,xm_result)
  for i in range(len(bias)):
    xm_result[i] = xm_result[i]+bias[i]
  
  x_result = Filter(x_result, time_stamp, k1, k2)
  
  
  for fid, (x1, x2) in enumerate(zip(x_result[2:], x_result[:-2]), 1):
    vx_result += [(x1 - x2) / (time_stamp[fid + 1] - time_stamp[fid - 1])]
  vx_result = [vx_result[0]] + vx_result + [vx_result[-1]]
  
  num = len(x_result)
  with open(gtjson,"r+") as f:
    gtjson = json.load(f)
    gtframe = gtjson["frame_data"]
  data = np.zeros((4, num),dtype=float)
  
  #print len(gtframe), len(x_result)
  for i in range(len(gtframe)):
    data[0][i] = gtframe[i]['vx']
    data[1][i] = gtframe[i]['x']
    data[2][i] = vx_result[i]
    data[3][i] = xm_result[i] # if you use xm_result please change
  process_data = np.zeros((2, num),dtype=float)
  process_data[0] = data[0] - data[2] # diff v
  process_data[1] = (data[1] - data[3]) / data[1] # diff x

  #print("mean_err[v,x]:", np.mean(np.abs(process_data), axis=1))
  return np.mean(np.abs(process_data), axis=1) 
if __name__ == '__main__':
  gtvalid = sys.argv[1] # gt
  arealist = sys.argv[2] # bbox
  time_stamp = sys.argv[3] # time

  gtfile = os.listdir(gtvalid)
  areafile = os.listdir(arealist)
  timefile = os.listdir(time_stamp)
  # k2 > k1
  min0 = np.array([[100,100]]);
  for k1 in range(10):
    for k2 in range(10):
      sum0 = np.zeros((1,2),dtype=float)
      List = []
      for i in range(len(gtfile)):
        gtjson = os.path.join(gtvalid, gtfile[i])
        areapath = os.path.join(arealist, areafile[i])
        timepath = os.path.join(time_stamp, timefile[i])
        #print gtjson, areapath, timepath
        tmp = run(pow(0.1,k1), pow(0.1,k2), gtjson, areapath, timepath)
        List.append(tmp)
        #print '-',tmp
        sum0 += tmp
      #print k2
      if (sum0[0][0] < min0[0][0]):
        min0 = sum0
        ans = List
        param = [[pow(0.1, k1)],[pow(0.1, k2)]]
  for i in ans:
    print i
  print min0/(len(gtfile))
  print (param) 
