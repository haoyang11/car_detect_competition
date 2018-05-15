
import numpy as np
'''
class KalmanFilter:
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    def __init__(self, pos):
        self.pos = pos
        self.kf.statePost =  np.array([[np.float32(self.pos)],[np.float32(0)],[np.float32(0)],[np.float32(0)]])
    
    def Estimate(self, t, coordX, coordY = 0):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.transitionMatrix = np.array([[1,0,t,0],[0,1,0,1],[0,0,t,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
        predicted = self.kf.predict()
        self.kf.correct(measured)
        return predicted

KalmanFilter(10);
'''

class KalmanFilterLinear:

    def __init__(self,_A, _H, _x, _P, _Q, _R):

        self.A = _A                      # State transition matrix.

        #self.B = _B                     # Control-input matrix.

        self.H = _H                      # Observation model matrix.

        self.current_state_estimate = _x # Initial state estimate.

        self.current_prob_estimate = _P  # Initial covariance estimate.

        self.Q = _Q                      # Estimated error in process.

        self.R = _R                      # Estimated error in measurements.

        self.KG = None

    def GetCurrentState(self):

        return self.current_state_estimate

    def Step(self,measurement_vector):

        #---------------------------Prediction step-----------------------------

        predicted_state_estimate = self.A * self.current_state_estimate 

        predicted_prob_estimate = (self.A * self.current_prob_estimate) * np.transpose(self.A) + self.Q

        #--------------------------Observation step-----------------------------

        innovation = measurement_vector - self.H*predicted_state_estimate

        self.Innovation = innovation[0, 0]

        innovation_covariance = self.H*predicted_prob_estimate*np.transpose(self.H) + self.R

        #-----------------------------Update step-------------------------------

        kalman_gain = predicted_prob_estimate * np.transpose(self.H) * np.linalg.inv(innovation_covariance)

        self.KG = kalman_gain

        self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation

        # We need the size of the matrix so we can make an identity matrix.

        size = self.current_prob_estimate.shape[0]

        self.current_prob_estimate = (np.eye(size)-kalman_gain*self.H)*predicted_prob_estimate
