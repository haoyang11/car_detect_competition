#from __future__ import print_function
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import sys
from cython_bbox import bbox_overlaps
from projection import ground_point_to_bird_view_proj
from projection import bird_view_proj_to_ground_point as bv2gp
#import projection.brid_view_proj_to_ground_point as bv2gp
import os
from scipy.fftpack import fft,ifft
import threading
from scipy.optimize import curve_fit

#sys.path.append('/home/m13/MCDC_FHS/py-R-FCN/caffe/python')
#sys.path.append('/home/m13/MCDC_FHS/py-R-FCN/lib')
sys.path.append('/home/m13/MCDC_FHS/py-R-FCN2/hy-frcnn/R-FCN-PSROIAlign/caffe/python')
sys.path.append('/home/m13/MCDC_FHS/py-R-FCN2/hy-frcnn/R-FCN-PSROIAlign/lib')
import caffe
from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
# hah 
import json
from pre import Filter
CLASSES = ('__background__','vehicle'
)

FRAME_TIME = 0.05023106289
Y_THRESHOLD = 0.35 #0.2
def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #print im_size_min, im_size_max
    processed_ims = []
    im_scale_factors = []

    for target_size in (900,):   #scale
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        #print '1',im_scale
        if np.round(im_scale * im_size_max) > 1300:
            im_scale = float(1300) / float(im_size_max)
            #print '2',im_scale
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        #print '#',im.shape, im_scale
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def _get_blobs(im, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors
def smooth_curve(x_result,N):
  plen=N
  head = [x_result[0] for z in range(plen)] 
  tail = [x_result[-1] for z in range(plen)] 
  pad_result= head+x_result+tail
  xm_result=[]
  for i in range(len(x_result)):
    xm_result.append(np.mean(pad_result[i:i+plen+1]))
  xm_result.reverse()
  head = [xm_result[0] for z in range(plen)] 
  tail = [xm_result[-1] for z in range(plen)] 
  pad_result= head+xm_result+tail
  hx_result=[]
  for i in range(len(x_result)):
    hx_result.append(np.mean(pad_result[i:i+plen+1]))
  hx_result.reverse()

  return  hx_result

def  fitfun(x,a,b):
  return a/x+b


def im_detect(net, im, boxes=None):
    blobs, im_scales = _get_blobs(im, boxes)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    #print blobs['data'].shape
    
    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]

    scores = blobs_out['cls_prob']

    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    #print 'debug--- boxes.size()', pred_boxes.shape
    return scores, pred_boxes

num=0
def vis_detections(im, class_name, dets, thresh=0.8):
    """Draw detected bounding boxes."""
    global num
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    frame = im
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
    print(num)
    cv2.imwrite('./'+str(num)+".jpg", frame)

def run_video(model, deploy, vp,outfile,timefile,nb):
  tot=0
  with open("/data/mcdc_data/valid/camera_parameter.json") as f:
    cam_param = json.load(f)
  cam_param["camera_height"] *=1 
  #with open('/data/mcdc_data/valid/valid_video_00_time.txt') as f:
  with open(timefile) as f: # TODO: modify it: video_00
    time_stamp = [line[:-1] for line in f]
  time_stamp = map(eval, time_stamp)
  caffe.set_mode_gpu()
  net = caffe.Net(deploy, model, caffe.TEST)
  cap = cv2.VideoCapture(vp)
  #fps = round(cap.get(cv2.CAP_PROP_FPS))
  success, image = cap.read()
  #video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  #video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	
  frame_id = 0
  stamp_id = 0
  x_result = []
  arealist=[]
  h_param=[]
  w_param=[]

  while (success):
    tot += 1
    timer = Timer()
    timer.tic()
    #if frame_id > 20: break
    #xmin_crop, xmax_crop, ymin_crop, ymax_crop = 300, 1300, 400, 1200
    #croped_image=image[400:1200,300:1300,:]
    scores, boxes = im_detect(net, image)
    #boxes[: , 4] += xmin_crop
    #boxes[: , 5] += ymin_crop
    #print '----debug-----box', boxes[:4, :]
    #exit(0)

    timer.toc()
    print (str(nb)+'Detection took {:.3f}s for ''{:d} object proposals').format(timer.total_time, boxes.shape[0])
    CONF_THRESH = 0.80
    NMS_THRESH = 0.3
    min_y = 0
    min_x = 51.0
    #Max = 0
    #arealist=[]
    maxb = [0,0,10,10]

    for cls_ind, cls in enumerate(CLASSES[1:]):
      cls_ind += 1
      cls_boxes = boxes[:, 4:8] #boxes[:, 4*cls_ind:4*(cls_ind + 1)]
      cls_scores = scores[:, cls_ind]
      cls_dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
      keep = nms(cls_dets, NMS_THRESH)
      #  add vote
      dets_NMSed = cls_dets[keep, :]
      BBOX_VOTE_FLAG=True
      if BBOX_VOTE_FLAG:
          dets = bbox_vote(dets_NMSed, cls_dets)
      else:
          dets = dets_NMSed

      #vis_detections_video(im, cls, dets, thresh=CONF_THRESH)
      
      ################################
      
      inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

      if len(inds) == 0:
        continue
      #print(inds)
      for i in inds:
        #area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox = dets[i, :4]
        #bbox[0] += 300
        #bbox[1] += 400
        #bbox[2] += 300
        #bbox[3] += 400
        x, y = ground_point_to_bird_view_proj((bbox[2] + bbox[0])/2, bbox[3], cam_param)
        #print(bbox)
        if -cam_param['cam_to_left'] - Y_THRESHOLD < y and y < cam_param['cam_to_right'] + Y_THRESHOLD and x < min_x:
          #Max = area
          min_x, min_y = x, y
          maxb = dets[i, :4] # max area bbox
        #score = dets[i, :4]
        #arealist+=[{"xmin":float(bbox[0]), "ymin":float(bbox[1]), "xmax": float(bbox[2]),"ymax":float(bbox[3])}]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
       #h_param.append(maxb[2]-maxb[0])
        #cv2.rectangle(image, (bbox))
    arealist+=[{"xmin":float(maxb[0]), "ymin":float(maxb[1]), "xmax": float(maxb[2]),"ymax":float(maxb[3])}]
    w_param.append(maxb[2]-maxb[0])
    h_param.append(maxb[3]-maxb[1])
    cv2.rectangle(image, (maxb[0], maxb[1]), (maxb[2], maxb[3]), (0, 255, 0), 2)


    #cv2.imwrite('output/'+str(tot)+".jpg", image)
    #for x in range(5, 50, 3):
    #	for y in range(-3, 4):
    #		u, v = bv2gp(x, y, cam_param)
    #		cv2.putText(image, str(x) + "," + str(y), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
    #		cv2.circle(image, (u, v), 3, (0, (x + y + 103) % 2 * 255, (x + y + 102) % 2 * 255), -1) 
    #cv2.imwrite('output/'+ 'qwe_' +str(nb-1) + '_' + str(tot)+".jpg", image)
    x_result.append(min_x)
        #videoWriter.write(image)
      ##############################
    frame_id += 1
    stamp_id += 1
    success, image = cap.read()
  vx_result = []



  for i in range(1,len(h_param)-1,2):
    h_param[i]=(float(h_param[i+1])+float(h_param[i+1]))/2
    w_param[i]=(float(w_param[i+1])+float(w_param[i+1]))/2


  lineparam=[ 4.29128904e+03, -1.99320880e+00] # new 05
  lineparam= [ 4.29169018e+03, -2.00891006e+00]
  fitd_w=[fitfun(i,lineparam[0],lineparam[1]) for i in w_param]
  lineparam=[ 3.33988934e+03, -2.55092719e+00] # new 05
  lineparam=[ 3.33810684e+03, -2.61628836e+00]
  fitd_h=[fitfun(i,lineparam[0],lineparam[1]) for i in h_param]
  fitd=[0.2*fitd_h[i]+0.8*fitd_w[i] for i in range(len(fitd_w))]

  fitd_m=smooth_curve(list(fitd),10)
  xm_result=fitd_m[:]
  if np.mean(fitd)>20:
    fitd_m=smooth_curve(list(fitd),30)
  else:
    fitd_m=smooth_curve(list(fitd),10)
  if np.mean(fitd)>20:
    fitds=smooth_curve(list(fitd),40)
  else:
    fitds=smooth_curve(list(fitd),15)

  x_result=fitds[:]

  fix_frame_time = False
  if fix_frame_time:
    FRAME_TIME = (time_stamp[-1] - time_stamp[0]) / (len(time_stamp) - 1) 
    for fid, (x1, x2) in enumerate(zip(x_result[2:], x_result[:-2]), 1):
      vx_result += [(x1 - x2) / (2 * FRAME_TIME)]
  else:
    for fid, (x1, x2) in enumerate(zip(x_result[2:], x_result[:-2]), 1):
      vx_result += ([(x1 - x2) / (time_stamp[fid+1] - time_stamp[fid - 1])])
  # vx_result = vx_result + [vx_result[-1]]
  vx_result = [vx_result[0]]+vx_result + [vx_result[-1]]
  vx_result=smooth_curve(list(vx_result),10)


  frame_data = []
  for fid, (vx, x) in enumerate(zip(vx_result, xm_result)):
    #frame_data += [{"vx": 0, "x": x, "fid": fid}]
    frame_data += [{"vx": vx, "x": x, "fid": fid}]
  result = {
    "frame_data": frame_data,
    "end_frame": frame_id,
    "start_frame": 0,
    "track_id": None,
    "arealist": arealist,
  }
  
  outjson=os.path.join("/home/m13/test_pre/",outfile)
  with open(outjson, "w") as output_file:
    json.dump(result, output_file) 
  #videoWriter.release()

def bbox_vote(dets_NMS, dets_all, thresh=0.8):
    dets_voted = np.zeros_like(dets_NMS)   # Empty matrix with the same shape and type

    _overlaps = bbox_overlaps(
      np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
      np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))

    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert(len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]
        out_box = np.dot(scores, boxes)
        dets_voted[i][0:4] = out_box / sum(scores)        # Weighted bounding boxes
        dets_voted[i][4] = det[4]                         # Keep the original score
        # Weighted scores (if enabled)
        BBOX_VOTE_N_WEIGHTED_SCORE=1
        BBOX_VOTE_WEIGHT_EMPTY=0.5
        if BBOX_VOTE_N_WEIGHTED_SCORE > 1:
            n_agreement = BBOX_VOTE_N_WEIGHTED_SCORE
            w_empty = BBOX_VOTE_WEIGHT_EMPTY

            n_detected = len(scores)

            if n_detected >= n_agreement:
                top_scores = -np.sort(-scores)[:n_agreement]
                new_score = np.average(top_scores)
            else:
                new_score = np.average(scores) * (n_detected * 1.0 + (n_agreement - n_detected) * w_empty) / n_agreement

            dets_voted[i][4] = min(new_score, dets_voted[i][4])
    return dets_voted




def action(model, deploy,infile,outfile,timefile, nnn):
  run_video(model, deploy,infile,outfile,timefile, nnn)

if __name__ == '__main__':
  model = sys.argv[2] #model
  deploy = sys.argv[1] #deploy
  imp = sys.argv[3]    # video path
  output = sys.argv[4] # output json
  timer_tot = Timer()
  timer_tot.tic()
  filelist=os.listdir(imp)

  threads = []
  nnn = 0
  for i in filelist:
    if i.find(".avi")>-1:
      if i.find("效果")>-1:
        continue
      outfile=i.split(".")[0]+"_pre.json"
      infile=os.path.join(imp,i)
      timefile=i.split(".")[0]+"_time.txt"
      timefile=os.path.join(imp,timefile)
      print(infile)
      print(outfile)
      print(timefile)
      nnn += 1
      t = threading.Thread(target = action, args = (model, deploy,infile,outfile,timefile, nnn))
      threads.append(t)

      #run_video(model, deploy,infile,outfile,timefile)
  for t in threads:
    #t.setDaemon(True)
    t.start()
    while True:
      if (len(threading.enumerate()) < 3):
        break
  for t in threads:
    t.join()
  timer_tot.toc()
  print(timer_tot.total_time)







'''
tot = 0
if __name__ == '__main__':
  model = sys.argv[2] #model
  deploy = sys.argv[1] #deploy
  imp = sys.argv[3]    # video path
  output = sys.argv[4] # output json

  timer_tot = Timer()
  timer_tot.tic()
  
  filelist=os.listdir(imp)
  for i in filelist:
    if i.find(".avi")>-1:
      if i.find("效果")>-1:
        continue
      outfile=i.split(".")[0]+"_pre.json"
      infile=os.path.join(imp,i)
      timefile=i.split(".")[0]+"_time.txt"
      timefile=os.path.join(imp,timefile)
      print(infile)
      print(outfile)
      print(timefile)
      run_video(model, deploy,infile,outfile,timefile,11)

  timer_tot.toc()
  print(timer_tot.total_time)


#vidcap = cv2.VideoCapture('/data/mcdc_data/valid/valid_video_00.avi')
#success,image = vidcap.read()
#count = 0

#while success:
#  success,image = vidcap.read()

'''
