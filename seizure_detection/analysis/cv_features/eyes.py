"""Eye activity estimator for flutter and saccade spikes."""
import numpy as np
import cv2

class Eyes:
    def __init__(self, fs=20.0, win_sec=3.0, band=(10,25)):
        self.fs=fs
        self.N=int(fs*win_sec)
        self.lbuf=np.zeros(self.N)
        self.rbuf=np.zeros(self.N)
        self.i=0
        self.lo, self.hi = band

    def update(self, ts_ms, gray, head_box=None):
        if not head_box:
            return dict(flutterHz=None, saccadeSpike=False)
        x,y,w,h = head_box
        H,W = gray.shape[:2]
        strip = gray[y:y+int(h*0.4), x:x+w]
        if strip.size==0:
            return dict(flutterHz=None, saccadeSpike=False)
        mid = w//2
        L = strip[:, :mid]
        R = strip[:, mid:]

        lval = float(np.abs(L.astype(np.float32)-cv2.blur(L,(3,3))).mean()/255.0)
        rval = float(np.abs(R.astype(np.float32)-cv2.blur(R,(3,3))).mean()/255.0)

        self.lbuf[self.i]=lval
        self.rbuf[self.i]=rval
        self.i=(self.i+1)%self.N
        sacc = (lval>0.35 or rval>0.35)

        if self.i!=0:
            return dict(flutterHz=None, saccadeSpike=sacc)
        
        def peak(buf):
            x = buf - buf.mean()
            X = np.fft.rfft(x*np.hanning(len(x)))
            P = (np.abs(X)**2)
            f=np.fft.rfftfreq(len(x), d=1.0/self.fs)
            mask=(f>=self.lo)&(f<=self.hi)
            if not mask.any():
                return None,0.0
            k=np.argmax(P[mask])
            return float(f[mask][k]), float(P[mask][k]/(P.max()+1e-9))
        
        fl,pl = peak(self.lbuf)
        fr,pr = peak(self.rbuf)
        f = None
        p=0.0
        for (ff,pp) in [(fl,pl),(fr,pr)]:
            if ff is not None and pp>p:
                f, p = ff, pp
        return dict(flutterHz=f, saccadeSpike=sacc)
