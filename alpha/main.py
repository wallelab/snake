from load_data import Dataset
from policy import Network

d = Dataset()
n = Network()

n.initialize_variables()


for i in range(100000):
	traindata = d.getdata(32)
	n.train(traindata, i)

n.save_variables("./saved_network/test")


