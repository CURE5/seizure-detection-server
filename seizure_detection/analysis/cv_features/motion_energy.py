"""Motion-energy feature extractor for frame differencing and flow proxies."""
import numpy as np
import cv2

class MotionEnergy:
    def __init__(self, alpha=0.9, grid=(32,18)):
        self.bg = None
        self.alpha = alpha
        self.grid_w, self.grid_h = grid
        self.prev = None

    def update(self, gray):
        if self.bg is None:
            self.bg = gray.astype(np.float32)
            self.prev = gray
            return dict(absMean=0.0, flowRMS=0.0, mapDownsample=[0.0]*(self.grid_w*self.grid_h))

        # EMA background & abs diff
        self.bg = self.alpha*self.bg + (1-self.alpha)*gray
        diff = cv2.absdiff(gray, self.bg.astype(np.uint8))
        abs_mean = float(diff.mean()/255.0)

        # cheap "flow": gradient magnitude * frame diff
        sobx = cv2.Sobel(gray, cv2.CV_32F, 1,0,ksize=3)
        soby = cv2.Sobel(gray, cv2.CV_32F, 0,1,ksize=3)
        mag  = cv2.magnitude(sobx, soby)
        fdiff = cv2.absdiff(gray, self.prev)
        self.prev = gray.copy()
        flow = (mag*(fdiff.astype(np.float32)+1)).mean()
        flow_rms = float(np.sqrt(flow)/255.0)

        # downsample grid
        ds = cv2.resize(diff, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)
        ds = (ds.astype(np.float32)/255.0).clip(0,1)
        return dict(absMean=abs_mean, flowRMS=flow_rms, mapDownsample=ds.flatten().tolist())
