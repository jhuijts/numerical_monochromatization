import numpy as np

def pix2q(pixelnumbers, Dpixel, z, lam, centered_in_pixel_zero=True):
    # input in SI units
    # q in half-cycles per m (so the inverse - resolution is equal to the Abbe criterion)
    if not centered_in_pixel_zero:
        # centered on the border between pixels -1 and 0
        pixelnumbers = pixelnumbers + 1/2
        
    theta = np.arctan(pixelnumbers*Dpixel/z)
    q = 2*np.sin(theta)/lam
    return q