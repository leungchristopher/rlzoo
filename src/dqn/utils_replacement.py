import os
import jax
import numpy as np
import pandas as pd
from flax.training import checkpoints
import matplotlib

# Force MacOSX backend for interactive plotting on Mac
try:
    matplotlib.use('MacOSX')
except ImportError:
    print("Warning: MacOSX backend not found. Plotting might not work correctly.")

import matplotlib.pyplot as plt

class CheckpointManager:
    def __init__(self, directory: str, max_to_keep: int = 5):
        self.directory = os.path.abspath(directory)
        self.max_to_keep = max_to_keep
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save(self, step: int, state, **kwargs):
        """Saves the train state to a checkpoint."""
        checkpoints.save_checkpoint(
            ckpt_dir=self.directory,
            target=state,
            step=step,
            keep=self.max_to_keep,
            overwrite=True,
            **kwargs
        )
        print(f"Checkpoint saved at step {step}")

    def restore(self, state):
        """Restores the latest checkpoint."""
        return checkpoints.restore_checkpoint(ckpt_dir=self.directory, target=state)
    
    def latest_step(self):
        """Returns the step of the latest checkpoint, or None if no checkpoint exists."""
        return checkpoints.latest_checkpoint(self.directory)

class MetricLogger:
    def __init__(self, window_size: int = 100, csv_file: str = "training_metrics.csv", plot_file: str = "training_loss.png"):
        self.window_size = window_size
        self.csv_file = csv_file
        self.plot_file = plot_file
        self.losses = []
        self.steps = []
        
        # Setup plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-', label='Loss')
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss')
        self.ax.legend()
        self.ax.grid(True)
        
        # Show window initially
        plt.show(block=False)
        plt.pause(0.1)

    def log(self, step: int, loss: float):
        self.steps.append(step)
        self.losses.append(loss)
        
        # Update plot every 1000 steps
        if len(self.steps) % 1000 == 0:
            self._update_plot()
            self._save_csv()

    def _update_plot(self):
        try:
            self.line.set_data(self.steps, self.losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
            # Save plot to file as backup
            self.fig.savefig(self.plot_file)
        except Exception as e:
            print(f"Warning: Plot update failed: {e}")

    def _save_csv(self):
        try:
            df = pd.DataFrame({'step': self.steps, 'loss': self.losses})
            df.to_csv(self.csv_file, index=False)
        except Exception as e:
            print(f"Warning: CSV save failed: {e}")
