import os
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
from pyrender import OffscreenRenderer

r = OffscreenRenderer(512, 512)