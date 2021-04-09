import numpy as np
import os
import cv2
import random
C_WIDTH = 153
C_HEIGHT = 89
C_ANGLE = 72
C_LAYER = 5



status = []
labels = []

class Dataset(object):
	def __init__(self):
		self.length = 0

	def prepare(self, work_dir = "/tftpboot/cv/data/"):
		with open(work_dir + "mouse.dat","rb") as file:
			szdat = file.read()
			mskey = []
			mslen = int(len(szdat)/8)
			for i in range(mslen):
				ip = i*8
				light = szdat[ip] + szdat[ip+1]*256
				seq = szdat[ip+2] + szdat[ip+3]*256
				key = szdat[ip+4] + szdat[ip+5]*256
				angle = szdat[ip+6] + szdat[ip+7]*256
				mskey.append([light, seq, key, angle])
			file.close()

		en = np.zeros(5, np.uint8)
		lay0 = np.zeros([C_HEIGHT, C_WIDTH], dtype = np.uint8)
		lay1 = lay0.copy()
		lay2 = lay0.copy()
		lay3 = lay0.copy()
		lay4 = lay0.copy()
		for i in range(mslen):
			picname = work_dir + "ss%04i.png"%mskey[i][1]
			picimg = cv2.imread(picname)
			lay4 = lay3.copy()
			lay3 = lay2.copy()
			lay2 = lay1.copy()
			lay1 = lay0.copy()
			lay0 = picimg[:,:,0]/255.
			onehot = np.zeros(C_ANGLE)
		
			en[1:5] = en[0:4]
			if (mskey[i][0]):
				en[0] = 1
			else:
				en = [0,0,0,0,0]

			if (np.sum(en) == 5):
				status.append(np.stack([lay0,lay1,lay2,lay3,lay4], axis = 2))
				onehot[mskey[i][3]] = 1
				labels.append(onehot.copy())
				self.length = self.length + 1

	def getdata(self, nbatch = 32):
		tblx = []
		tbly = []
		for i in range(nbatch):
			id = random.randrange(self.length)
			tblx.append(status[id])
			tbly.append(labels[id])
		return (tblx,tbly)


