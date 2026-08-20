import numpy as np
import random as pyrandom

class Vec2:
    __slots__ = ('x', 'y')
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x); self.y = float(y)
    def __add__(self, other): return Vec2(self.x+other.x, self.y+other.y)
    def __sub__(self, other): return Vec2(self.x-other.x, self.y-other.y)
    def __mul__(self, s): return Vec2(self.x*s, self.y*s)
    def dot(self, other): return self.x*other.x + self.y*other.y
    def norm2(self): return self.x**2 + self.y**2
    def norm(self): return np.sqrt(self.norm2())

def perp(v): return Vec2(-v.y, v.x)

class RNG:
    def __init__(self):
        self.uni = pyrandom.uniform
        self._gauss_cache = None

    def uniform_in_box(self, xmin, xmax, ymin, ymax):
        return Vec2(self.uni(xmin,xmax), self.uni(ymin,ymax))

    def gaussian(self, mean=0.0, std=1.0):
        # Box-Muller，带缓存
        if self._gauss_cache is not None:
            val = self._gauss_cache
            self._gauss_cache = None
            return mean + std * val
        u1 = self.uni(0,1); u2 = self.uni(0,1)
        r = np.sqrt(-2*np.log(u1))
        self._gauss_cache = r*np.sin(2*np.pi*u2)
        return mean + std*(r*np.cos(2*np.pi*u2))