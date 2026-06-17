from PyQt5.QtWidgets import *
from PyQt5.Qt import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import random
import sys
import seaborn as sns
import numpy as np
class Rule: 
    def __init__(self, a, b, c: int = 0):
        self._a = a
        self._b = b
        self._c = c

    def quadratic(self, t):
        a = self._a
        b = self._b
        c = self._c
        return a*t**2 + b*t + c
    
    def linear(self, t):
        a = self._a
        b = self._b
        return a*t + b
    
    def derivative(self, dt, deg = 1):
        if deg == 1: 
            return self._a * dt 
        if deg == 2: 
            return 2 * self._a * dt + self._b * dt


class DataGenerator: 
    def __init__(self, w, rule: Rule):
        """w: process noise"""
        self._rule = rule
        self._w = w

    def generate(self, t1, ticks):
        """Generate data with process noise, Real."""
        time = np.linspace(0, t1, ticks)
        self._time = time
        noise_w = self._w
        genrule = self._rule
        return [genrule(t) + random.normalvariate(0, noise_w) for t in time]
    
    def get_rule(self):
        return self._rule
    def get_timeline(self):
        return self._time

class ModelComplex: 
    def __init__(self, GeneratorInstance, R, Q, P0, E0): 
        """
        R: Sensor noise,
        Q: Innate process noise, 
        P0: Initial estimate of Estimation Variance,
        E0: Initial estimate, 
        """
        self._R = R
        self._Q = Q
        
        self._measurements = None
        self._generator = GeneratorInstance
        self._generation_rule = self._generator.get_rule()
        self._timeline = self._generator.get_timeline()

        self._estimation = E0
        self._P = P0
        self._KalmanG = P0/(P0+R)
        self._state_id = 0

        self._estimations_updated = []
        self._Plist = []
        self._KGlist = []

    def ensure_generators(self):
        pass

    def measure(self, real_data: list = None):
        if real_data is None:
            raise ValueError("Invalid, None entered") 
        if not isinstance(real_data, list):
            raise ValueError("Enter a list")
        if len(real_data) <= 1: 
            raise ValueError("The length of the real data should m be bigger than 1.")
      
        try:
            self._real_data = real_data            
            self._measurements = [x + random.normalvariate(0, self._R) for x in real_data]
        except ValueError:
            print ("Value Error occured.")   
    
    def estimate_update(self): 

        zt = self._measurements[self._state_id]

        #model estimation
        if self._state_id > 0: 
            dt = self._timeline[self._state_id] - self._timeline[self._state_id - 1]
            self._estimation += self._generation_rule.derivative(dt, deg = 1)
        else: 
            dt = 0

        #update estimation
        self._estimation = self._estimation + self._KalmanG * (zt - self._estimation)
        #set estimation variance
        self._P = self._P + self._Q
        #update kalman
        self._KalmanG = self._P/(self._P + self._R)
        #update P, estimation variance
        self._P = (1 - self._KalmanG) * self._P

        self._estimations_updated.append(self._estimation)
        self._Plist.append(self._P)
        self._KGlist.append(self._KalmanG)
        self._state_id += 1        
    def get_data(self):
        return self._estimations_updated, self._measurements

class SimulationApp(QWidget): 
    def __init__(self):
        super().__init__()

        self.in_Q_label = QLabel("Set process noise 'Q'")
        self.in_Q = QSlider(Qt.Horizontal)
        self.in_Q.setMinimum(1)
        self.in_Q.setMaximum(5)

        self.in_R_label = QLabel("Set measurement noise 'R'")
        self.in_R = QSlider(Qt.Horizontal)
        self.in_R.setMinimum(1)
        self.in_R.setMaximum(5)
        
        self.in_ledit_estimate0 = QLineEdit()
        self.in_ledit_estimate0.setPlaceholderText("Enter initial estimate 'X0'")
        
        self.in_ledit_p0 = QLineEdit()
        self.in_ledit_p0.setPlaceholderText("Enter initial estimate variance 'P0'")
        
        self.in_ledit_t1 = QLineEdit()
        self.in_ledit_t1.setPlaceholderText("Enter time end")
        
        self.button = QPushButton("Generate and Draw")
        self.button.clicked.connect(self.run)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.in_Q_label)
        layout.addWidget(self.in_Q)
        layout.addWidget(self.in_R_label)
        layout.addWidget(self.in_R)
        layout.addWidget(self.in_ledit_estimate0)
        layout.addWidget(self.in_ledit_p0)
        layout.addWidget(self.in_ledit_t1)
        layout.addWidget(self.button)
        layout.addWidget(self.canvas,1)
        self.setLayout(layout)
    
    def run(self): 
        process_noise = self.in_Q.value()
        measurement_noise = self.in_R.value()
        time = int(self.in_ledit_t1.text())
        est0 = int(self.in_ledit_estimate0.text())
        p0 = int(self.in_ledit_p0.text())

        rule = Rule(1.5, 1)
        gen = DataGenerator(process_noise, rule.linear)
        real_pos = gen.generate(time, time*10)
        time = gen.get_timeline()
        mc = ModelComplex(gen, measurement_noise, process_noise, p0, est0)
        n = len(time)

        mc.measure(real_pos)

        for _ in range(n):
            mc.estimate_update()

        filtered_pos, measured_pos = mc.get_data()
        self.figure.clear()

        gs = self.figure.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_main = self.figure.add_subplot(gs[0, 0]) # Left column
        ax_res = self.figure.add_subplot(gs[0, 1])  # Right column

        # Main Plot
        ax_main.plot(time, real_pos, label="Real Positions", linestyle = "--", color = "darkblue")
        ax_main.scatter(time, filtered_pos, label="Filtered Positions", s=20, color="limegreen")
        ax_main.scatter(time, measured_pos, label="Measured", s=20, color="orangered")
        ax_main.legend()

        # Calculate raw residuals
        residuals_filter = np.array(real_pos) - np.array(filtered_pos)
        residuals_measurements = np.array(real_pos) - np.array(measured_pos)

        # Calculate absolute errors for comparison and plotting
        abs_res_filter = np.abs(residuals_filter)
        abs_res_meas = np.abs(residuals_measurements)

# Calculate raw absolute errors
        abs_res_filter = np.abs(np.array(real_pos) - np.array(filtered_pos))
        abs_res_meas = np.abs(np.array(real_pos) - np.array(measured_pos))

        # Quick convolution to get a moving average (e.g., window size of 5)
        window = 5
        weights = np.ones(window) / window
        smooth_filter = np.convolve(abs_res_filter, weights, mode='valid')
        smooth_meas = np.convolve(abs_res_meas, weights, mode='valid')
        smooth_time = time[window-1:] # Adjust time array to match

        ax_res.plot(smooth_time, smooth_meas, color='orange', label="Smoothed Measurement Error")
        ax_res.plot(smooth_time, smooth_filter, color='darkgreen', label="Smoothed Filter Error")
        
        ax_res.set_title(f"Moving Average Absolute Residuals (Window={window})")
        ax_res.legend()

        self.canvas.draw()

app = QApplication(sys.argv)
w = SimulationApp()
w.show()
sys.exit(app.exec_())
    



