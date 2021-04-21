import numpy as np
import cv2

C_W0 = 1530
C_H0 = 890
C_W = 640
C_H = 480
C_GAP = 40
C_GAP2 = 80

x_cut = (C_W0 - C_W)>>1
y_cut = (C_H0 - C_H)>>1

diffrows = np.zeros([C_H, C_GAP2, C_GAP2], float)
volt = np.zeros([C_H, C_GAP2, C_GAP2], float)

def get(npic):
	name0 = "/tftpboot/cv/game/ss%04i.png"%npic
	name1 = "/tftpboot/cv/game/ss%04i.png"%(npic+1)
	image0 = cv2.imread(name0)/255.
	image1 = cv2.imread(name1)/255.
	image = image0[y_cut:y_cut+C_H, x_cut:x_cut+C_W, :]

	for i in range(C_H):
		row = image[i, :, :]
		for iy in range(C_GAP2):
			y_pos = y_cut - C_GAP + iy
			for it in range(C_GAP2):
				x_pos = x_cut - C_GAP + it
				flux = image1[y_pos, x_pos:x_pos+C_W, :]
				diff = np.abs(row - flux)
				diffrows[i, iy, it] = np.sum(diff)
		vmax = np.max(diffrows[i,])
		vmin = np.min(diffrows[i,])
		volt[i,:,:] = (diffrows[i,:,:]-vmin)/(vmax - vmin)

	gb_diff = np.sum(volt, axis=0)
	vmax = np.max(gb_diff)
	vmin = np.min(gb_diff)
	diffimg = (gb_diff-vmin)/(vmax - vmin)*255.
	return diffimg
	




def main():
	img = get(22)
	cv2.imshow("test", img)
	cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()


