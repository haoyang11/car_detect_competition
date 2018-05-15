import numpy as np
#import matplotlib.pyplot as plt
from kalman import KalmanFilterLinear
#import json
#f = open("C:\\Users\\conna\\Documents\\MCDC\\test_vedeo_00_pre.json", encoding='utf-8')
#arr = json.load(f)
#dic = arr['frame_data']
#Z = []
#V = []
#for x in dic:
#    Z.append(x['x'])  # guanche
#    V.append(x['vx']) 
#print (Z)

# canshu Z:x t:time change



def Filter(Z, T, k1, k2):
  t =T[1] - T[0];
  A = np.array([[1,0,t,0],[0,1,0,t],[0,0,1,0],[0,0,0,1]], np.float32)
  H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
  X = np.array([[np.float32(Z[0])],[np.float32(0)],[np.float32(0)],[np.float32(0)]])
  Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)*k1
  P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
  R = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)*k2

  new_z = []
  new_z.append(Z[0])
  kf = KalmanFilterLinear(A,H,X,P,Q,R)
  size = len(Z)
  for i in range(1,size):
      current_pre = kf.GetCurrentState()
      if i + 1 < size:
          A[0][2] = A[1][3] = T[i+1] - T[i]
      #print A[0][2]
      kf.Step(np.array([[np.float32(Z[i])], [np.float32(0)],[np.float32(0)],[np.float32(0)]]))
      new_z.append(1.0*current_pre[0][0] + Z[i-1]*0.0) 
      #print 'kalman'
  return new_z
  #print Z

'''
kf = KalmanFilter(Z[0])
for i in range(1,10):
    current_pre = kf.Estimate(0.1,Z[i]);    
    plt.plot(i, current_pre[0], '.')
                      
plt.plot(range(len(Z)), Z, '+')
'''
#Filter([1,2,3],1)
