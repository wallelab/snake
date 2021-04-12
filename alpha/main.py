from load_data import Dataset
from policy import Network

d = Dataset()
d.prepare("/tftpboot/cv/data2/")

n = Network()
n.initialize_variables("./saved_network/")

print("dataset length = ", d.length)

for i in range(100000):
	traindata = d.getdata(32)
	n.train(traindata, i)

n.save_variables("./saved_network/test", 1)


