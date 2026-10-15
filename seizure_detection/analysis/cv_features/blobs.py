"""Foreground blob tracker for torso/head estimation."""
import numpy as np
import cv2

class Blobs:
    def __init__(self, diff_thresh=18):
        self.diff_thresh = diff_thresh
        self.bg = None

    def _fg(self, gray):
        if self.bg is None:
            self.bg = gray.astype(np.float32)
            return np.zeros_like(gray)
        self.bg = 0.98*self.bg + 0.02*gray
        diff = cv2.absdiff(gray, self.bg.astype(np.uint8))
        fg = (diff > self.diff_thresh).astype(np.uint8)*255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        return fg

    def update(self, gray, person_box=None, head_hint=None):
        H,W = gray.shape[:2]
        fg = self._fg(gray)
        coverage = float(fg.mean()/255.0)

        torso=head=left=None
        right=None
        if person_box is not None:
            x,y,w,h = person_box
            roi = fg[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
            if roi.size>0:
                cnts,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    tx,ty,tw,th = cv2.boundingRect(c)
                    torso = (x+tx, y+ty, tw, th)
                    if head_hint:
                        hx,hy,hw,hh = head_hint
                        head = (hx,hy,hw,hh)
                    else:
                        head = (x+tx, y+ty, tw, max(8, int(th*0.35)))

        return dict(torso=torso, head=head, leftArm=left, rightArm=right, coverage=coverage)
