"""Breathing-rate estimator from chest micromotion."""
import numpy as np
import cv2

class Breathing:
    def __init__(self, fs=20.0, win_sec=30.0, band_bpm=(8,40)):
        self.fs=fs
        self.N=int(fs*win_sec)
        self.sig=np.zeros(self.N)
        self.i=0
        self.filled=False
        self.lo, self.hi = [x/60.0 for x in band_bpm]

    def update(self, ts_ms, gray, torso_box=None):
        H,W = gray.shape[:2]
        if torso_box:
            x,y,w,h = torso_box
            ry = y + int(h*0.5)
            rh = max(8, int(h*0.45))
            rx = x
            rw = w
        else:
            rx, rw = W//4, W//2
            ry, rh = H//2, H//4

        roi = gray[max(0,ry):min(H,ry+rh), max(0,rx):min(W,rx+rw)]
        if roi.size==0:
            return dict(rrBpm=None, microMotion=0.0)

        val = float(cv2.absdiff(roi, cv2.GaussianBlur(roi, (0,0), 1.0)).mean()/255.0)
        self.sig[self.i]=val
        self.i=(self.i+1)%self.N
        if self.i==0:
            self.filled=True
        L = self.N if self.filled else self.i
        if L < int(self.fs*10):
            return dict(rrBpm=None, microMotion=val)

        x = np.roll(self.sig, -self.i)[:L]
        x = x - x.mean()
        w = np.hanning(L)
        X = np.fft.rfft(x*w)
        P = (np.abs(X)**2)
        f = np.fft.rfftfreq(L, d=1.0/self.fs)
        mask = (f>=self.lo)&(f<=self.hi)
        if not mask.any():
            return dict(rrBpm=None, microMotion=val)
        k = np.argmax(P[mask])
        rr = float(f[mask][k]*60.0)
        return dict(rrBpm=rr, microMotion=val)
