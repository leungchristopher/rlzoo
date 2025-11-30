import os
import jax
import matplotlib
try:
    matplotlib.use('MacOSX')
except ImportError:
    print("MacOSX backend not found, using default.")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flax.training import checkpoints

class CheckpointManager:
    def __init__(self, directory: str, max_to_keep: int = 5):
        self.directory = directory
        self.max_to_keep = max_to_keep
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save(self, step: int, state, **kwargs):
        checkpoints.save_checkpoint(
            ckpt_dir=self.directory,
            target=state,
            step=step,
            keep=self.max_to_keep,
            overwrite=True,
            **kwargs
        )
        print(f"Saved checkpoint at step {step}")

    def restore(self, state):
        return checkpoints.restore_checkpoint(ckpt_dir=self.directory, target=state)

class MetricLogger:
    def __init__(self, window_size: int = 100, csv_file: str = "training_metrics.csv"):
        self.window_size = window_size
        self.csv_file = csv_file
        self.losses = []
        self.steps = []
        
        # Setup plot
        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss')

    def log(self, step: int, loss: float):
        self.steps.append(step)
        self.losses.append(loss)
        
        # Update plot every N steps to avoid slowing down training too much
        if len(self.steps) % 1000 == 0:
            self.line.set_data(self.steps, self.losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
            # Save to CSV
            df = pd.DataFrame({'step': self.steps, 'loss': self.losses})
            df.to_csv(self.csv_file, index=False)
