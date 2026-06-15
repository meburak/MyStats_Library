from PyQt5.QtWidgets import *
import random
from matplotlib.pyplot import plot
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import sys
from PyQt5.QtCore import Qt

class Sensor:
    def __init__(self, noise):
        self._noise = noise
    def observe(self, state):
        noise = random.normalvariate(0, self._noise)
        return state + noise
    def get_noise(self):
        return self._noise
    
class Agent: 
    def __init__(self, speed, acc=0, pr_noise = 1):
        """acc: default 0, pr_noise: process noise default 1"""
        self._speed = speed
        self._acc = acc
        self._state_idx = 0
        self._pr_noise = pr_noise
        self._history_real = []
        self._history_observation = []
        self._history_model = []

    def get_noise(self):
        return self._pr_noise
    
    def set_time(self, tfinal, ticks = 10):
        self._timeline = list(np.linspace(0, tfinal, ticks))
        return self._timeline
    
    def place_agent(self, posx):
        self._posx = posx
        self._posx_model = posx
        self._history_real.append(posx)
        self._history_observation.append(posx)
        self._history_model.append(posx)

        return self._posx

    def move(self):
        """next state = state_idx + 1"""
        """move the agent to next state, with innate noise"""
        state = self._state_idx
        time = self._timeline

        self._posx_model = self._posx_model + (time[state+1]-time[state]) * self._speed
        noise = random.normalvariate(0, self._pr_noise)

        self._posx =  self._posx_model + noise

        self._state_idx += 1

        self._history_real.append(self._posx)
        self._history_model.append(self._posx_model)
        return self._posx, self._posx_model
    
    def observe(self, sensor: Sensor):
        observer = sensor
        observation = observer.observe(self._posx)
        self._history_observation.append(observation)
        return observation

class KalmanFilter: 
    def __init__(self, agent: Agent,  sensor: Sensor): 
        """Actual position | Models prediction of position | Observed position"""
        self.agent = agent
        self.sensor = sensor
        self.sensor_noise = sensor.get_noise()
        self.model_noise = agent.get_noise()
        self._gain = self.model_noise / (self.sensor_noise + self.model_noise)

    def filter(self): 
        self.history_filtered = [self.agent._history_real[0]]
        history_model = self.agent._history_model
        history_real = self.agent._history_real
        history_obs = self.agent._history_observation
        for i in range(len(history_model)-1):
            prior = history_model[i]
            projection = history_model[i]
            observation = history_obs[i]
            self.history_filtered.append(prior + self._gain * (observation - projection))
        return self.history_filtered

class KalmanGUI(QWidget): 
    def __init__(self):
        super().__init__()

        self.label_mn = QLabel()
        self.label_mn.setText("Model noise | Process noise")
        self.input_mn = QSlider(Qt.Horizontal)
        self.input_mn.setMinimum(1)
        self.input_mn.setMaximum(10)
        
        self.label_wn = QLabel()
        self.label_wn.setText("White noise | Observation noise")
        self.input_wn = QSlider(Qt.Horizontal)
        self.input_wn.setMinimum(1)
        self.input_wn.setMaximum(10)
        
        self.button = QPushButton("Observe")
        self.button.clicked.connect(self.run)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label_mn)
        layout.addWidget(self.input_mn)
        layout.addWidget(self.label_wn)
        layout.addWidget(self.input_wn)
        layout.addWidget(self.button)
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

    def run(self):
        agent = Agent(5, 0, self.input_mn.value())
        sensor = Sensor(self.input_wn.value())

        tg = agent.set_time(5, 100)

        agent.place_agent(0)
        for i in range(1, len(tg)):
            agent.move()
            agent.observe(sensor)
            

        kf = KalmanFilter(agent, sensor)
        kf.filter()
        filtered = kf.history_filtered
        model = agent._history_model
        real = agent._history_real
        observations = agent._history_observation
        time = tg

        
        self.figure.clear()
        
        # 1. Create a GridSpec layout: 2 rows, 2 columns. 
        # width_ratios=[2, 1] makes the left column twice as wide as the right column.
        gs = self.figure.add_gridspec(2, 2, width_ratios=[2, 1])
        ax_main = self.figure.add_subplot(gs[:, 0]) # Spans both rows in col 0
        ax_obs = self.figure.add_subplot(gs[0, 1])  # Top row, col 1
        ax_mod = self.figure.add_subplot(gs[1, 1])  # Bottom row, col 1

        # 2. Convert to numpy arrays for element-wise math
        time_arr = np.array(time)
        f_arr = np.array(filtered)
        m_arr = np.array(model)
        r_arr = np.array(real)
        o_arr = np.array(observations) # Grab observations directly from agent

        # --- MAIN PLOT ---
        ax_main.scatter(time_arr, f_arr, label="Filtered", color="mediumslateblue", s=15)
        ax_main.plot(time_arr, m_arr, label="Model", color="blue", linestyle="--")
        ax_main.plot(time_arr, r_arr, label="Real", color="black", linewidth=2)
        ax_main.scatter(time_arr, o_arr, label="Observed", color="sandybrown", s=15)
        ax_main.set_title(f"Q = {self.input_mn.value()}, R = {self.input_wn.value()}")
        ax_main.legend()

        # --- METRICS CALCULATIONS ---
        # Error = absolute distance from the real ground-truth position
        err_filter = np.abs(f_arr - r_arr)
        err_obs = np.abs(o_arr - r_arr)
        err_model = np.abs(m_arr - r_arr)

        # Metric: > 0 means competitor error is larger (Filter Wins -> Green)
        # Metric: < 0 means filter error is larger (Filter Loses -> Red)
        metric_obs = err_obs - err_filter
        metric_mod = err_model - err_filter

        # --- SIDE PLOT 1: Filter vs Observation ---
        ax_obs.plot(time_arr, metric_obs, color="black", linewidth=1)
        ax_obs.fill_between(time_arr, metric_obs, 0, where=(metric_obs > 0), color='green', alpha=0.5, interpolate=True)
        ax_obs.fill_between(time_arr, metric_obs, 0, where=(metric_obs <= 0), color='red', alpha=0.5, interpolate=True)
        ax_obs.axhline(0, color='black', linewidth=0.5, linestyle='--') # Add a zero line
        ax_obs.set_title("Filter vs. Sensor", fontsize=10)

        # --- SIDE PLOT 2: Filter vs Model ---
        ax_mod.plot(time_arr, metric_mod, color="black", linewidth=1)
        ax_mod.fill_between(time_arr, metric_mod, 0, where=(metric_mod > 0), color='green', alpha=0.5, interpolate=True)
        ax_mod.fill_between(time_arr, metric_mod, 0, where=(metric_mod <= 0), color='red', alpha=0.5, interpolate=True)
        ax_mod.axhline(0, color='black', linewidth=0.5, linestyle='--') # Add a zero line
        ax_mod.set_title("Filter vs. Model", fontsize=10)

        # tight_layout prevents the subplot titles and labels from overlapping
        self.figure.tight_layout() 
        self.canvas.draw()

app = QApplication(sys.argv)
my_app = KalmanGUI()

my_app.resize(1200, 1200)
my_app.show()

# Run the event loop
sys.exit(app.exec_())