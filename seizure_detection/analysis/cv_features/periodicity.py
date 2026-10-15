"""Periodicity analysis for rhythmic motion detection."""
import numpy as np

class Periodicity:
    def __init__(self, fs=20.0, win_sec=10.0, band=(2.0,6.0)):
        self.fs=fs
        self.N=int(fs*win_sec)
        self.band=band
        self.buf=np.zeros(self.N)
        self.i=0
        self.filled=False

    def update(self, t_ms, motion_scalar):
        self.buf[self.i] = motion_scalar
        self.i = (self.i+1)%self.N
        if self.i==0:
            self.filled=True
        length = self.N if self.filled else self.i
        if length < int(self.fs*2):
            return dict(peakHz=None, peakPower=0.0, bandPower=0.0)

        x = np.roll(self.buf, -self.i)[:length]
        x = x - x.mean()
        w = np.hanning(len(x))
        X = np.fft.rfft(x*w)
        P = (np.abs(X)**2)
        freqs = np.fft.rfftfreq(len(x), d=1.0/self.fs)

        lo, hi = self.band
        band_mask = (freqs>=lo)&(freqs<=hi)
        total = P.sum()+1e-9
        band_power = P[band_mask].sum()/total if band_mask.any() else 0.0
        if band_mask.any():
            k = np.argmax(P[band_mask])
            peak_hz = float(freqs[band_mask][k])
            peak_power = float(P[band_mask][k]/(P.max()+1e-9))
        else:
            peak_hz, peak_power = None, 0.0
        return dict(peakHz=peak_hz, peakPower=peak_power, bandPower=float(band_power))
