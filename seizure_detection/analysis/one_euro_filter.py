import math

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff  # Minimum cutoff frequency
        self.beta = beta              # Speed-based smoothing factor
        self.d_cutoff = d_cutoff      # Cutoff frequency for derivative
        self.x_prev = None            # Previous value
        self.dx_prev = None           # Previous derivative
        self.t_prev = None            # Previous time

    def alpha(self, cutoff, dt):
        """Compute smoothing factor alpha based on cutoff frequency and delta time."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        """Apply the One Euro Filter to input x at time t."""
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        # Time delta
        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev

        # Compute derivative
        dx = (x - self.x_prev) / dt
        dx_alpha = self.alpha(self.d_cutoff, dt)
        dx = dx_alpha * dx + (1.0 - dx_alpha) * self.dx_prev

        # Update cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx)

        # Apply low-pass filter
        alpha = self.alpha(cutoff, dt)
        x_filtered = alpha * x + (1.0 - alpha) * self.x_prev

        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx
        self.t_prev = t

        return x_filtered