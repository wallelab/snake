import numpy as np
import matplotlib.pyplot as plt
import cv2

#cutx = 1230
#cuty = 390
cutx = 240
cuty = 260


work_dir = "/tftpboot/cv/game/"
work_f1 = work_dir + "ss0004.png"
work_f2 = work_dir + "ss0089.png"
work_f3 = work_dir + "ss0169.png"
workfile = [work_f1, work_f2, work_f3]

def adjust():
	global cutx, cuty
	image = cv2.imread(work_f3)
	cv2.cvtColor(image, image, cv2.COLOR_BRG2GRAY)
	while (True):
		cutimg = image[0:cuty, 0:cutx]
		cv2.imshow("image", cutimg)
		c = cv2.waitKey()
		if (c == 27):
			break;
		elif (c == 81):
			cutx = cutx - 1
		elif (c == 83):
			cutx = cutx + 1
		if (c == 82):
			cuty = cuty - 1
		elif (c == 84):
			cuty = cuty + 1

		print(cutx, cuty)
	cv2.destroyAllWindows()


def histogram(file):
	image = cv2.imread(file)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cutimg = image[0:cuty, cutx:-1]
	histo = np.zeros(cuty)
	for i in range(cuty):
		histo[i] = np.sum(cutimg[i, :])
	return histo

histos = []
def stdhisto():
	for file in workfile:
		histo = histogram(file)
		histos.append(histo)

	for d in histos:
		plt.plot(d)
		plt.show()

curve1 = []
curve2 = []
curve3 = []
def distribute():
	stdhisto()
	histo = np.zeros(cuty)
	for i in range(1, 1800):
		testfile = work_dir + "ss%04i.png"%(i)
		image = cv2.imread(testfile)
		cutimg = image[0:cuty, cutx:-1, 0]
		diff1 = 0
		diff2 = 0
		diff3 = 0
		for i in range(cuty):
			histo[i] = np.sum(cutimg[i, :])
			delta = np.abs(histos[0][i] - histo[i])
			if (delta < 3000):
				diff1 = diff1 + 1
			delta = np.abs(histos[1][i] - histo[i])
			if (delta < 3000):
				diff2 = diff2 + 1
			delta = np.abs(histos[2][i] - histo[i])
			if (delta < 3000):
				diff3 = diff3 + 1
		#diff1 = np.sum(np.abs(histos[0] - histo))
		#diff2 = np.sum(np.abs(histos[1] - histo))
		#diff3 = np.sum(np.abs(histos[2] - histo))
		curve1.append(diff1)
		curve2.append(diff2)
		curve3.append(diff3)
		print(testfile)
	plt.plot(curve1)
	plt.plot(curve2)
	plt.plot(curve3)
	plt.show()


if __name__ == '__main__':
	#adjust()
	stdhisto()
	#distribute()




