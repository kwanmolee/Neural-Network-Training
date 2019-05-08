import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import subplots, cm
import matplotlib.image as mpimg

"""""""""""""""""""""""""""""""""""""""""""""""
--------------- plot 1 -----------------------
Full batch learning 
- T / Epoch = 1024
- validation : 10% of training set = 6000 examples
- risk: normalized over training examples and classes

- plots: accuracy + risk over training/validation 
"""""""""""""""""""""""""""""""""""""""""""""""

path = "/nndata"

with open(path,'rb') as f:
	data = pickle.load(f)

[train_ACC,validation_ACC,train_loss,validation_loss] = data

c = ["r","g","blue","orange"]
title = ["Accuracy per epoch over training set", 
			"Accuracy per epoch over validation set", 
			"Risk per epoch over training set", 
			"Risk per epoch over validation set"]
yaxis = ["Accuracy","Accuracy","Risk","Risk"]
xaxis = ["Epoch" for i in range(len(data))]


annotate = [["acc_max = "+str(max(train_ACC))]]
for i in range(len(data)):
	plt.figure(i)
	d = np.array(data[i])
	x = range(len(d))
	plt.xlabel(xaxis[i],weight = "bold")
	plt.ylabel(yaxis[i],weight = "bold")
	plt.title(title[i],weight = "bold", fontsize = 18)
	# start
	plt.text(0, d[0], "start value = " + "{0:.4f}".format(d[0]), fontsize = 12)
	# max
	if i<=1:
		loc = 700
	else:
		loc = np.argwhere(d == np.max(d))
	plt.text(loc, np.max(d), "max value = " + "{0:.4f}".format(np.max(d)), fontsize = 12)
	if i == 2 or i == 3:
		# min
		plt.text(800, np.min(d), "min value = " + "{0:.4f}".format(np.min(d)), fontsize = 12)
	plt.plot(x,d,c[i])

"""""""""""""""""""""""""""""""""""""""""""""""
--------------- plot 2 -----------------------
Full batch learning 

- weight matrix reshaped to 10 images
- 10 average images from 10 classes
"""""""""""""""""""""""""""""""""""""""""""""""
path = "/avgimg"

with open(path,'rb') as f:
	data = pickle.load(f)

[c_avg, w_avg] = data

for i in range(2):
	fig,axs=subplots(2,5,sharex=True,sharey=True)
	n=0
	img = data[i]
	for ax in axs.flatten():
		im = img[n]
		ax.imshow(im,cmap='binary')
		ax.set_title("Digit "+str(n),weight='bold')
		n+=1



"""""""""""""""""""""""""""""""""""""""""""""""
--------------- plot 3 -----------------------
Full batch learning: hyperparameter tuning 

- function of learning rate against accuracy over epochs
"""""""""""""""""""""""""""""""""""""""""""""""

g = np.array([-1,-2,-3,-4,-5,-6,-7]) # gamma = 1e-n -> log(gamma) -> g
path = "/eta"
c = ["lightskyblue", "tomato"]
title = ["Function of $\eta_{0}$: training accruacy", "Function of $\eta_{0}$: validation accruacy"]
yaxis = ["Training Accuracy","Validation Accuracy"]
xaxis = ["$log_{10}(\eta_{0})$" for i in range(2)]
with open(path,'rb') as f:
	data = pickle.load(f)

for i in range(2):
	plt.figure(i+7)
	plt.title(title[i],weight = "bold")
	plt.xlabel(xaxis[i],weight = "bold")
	plt.ylabel(yaxis[i],weight = "bold")
	d = np.array(data[i])
	plt.text(g[np.argwhere(d == np.max(d))], np.max(d), "max acc = " + "{0:.4f}".format(np.max(d)), fontsize = 12)
	plt.text(g[np.argwhere(d == np.min(d))], np.min(d), "min acc = " + "{0:.4f}".format(np.min(d)), fontsize = 12)
	plt.plot(g,d,c[i])


"""""""""""""""""""""""""""""""""""""""""""""""
--------------- plot 4 -----------------------
Batch learning - batch size = 64 

- epoch = 8 
- accuracy + risk over epochs training / validation set
"""""""""""""""""""""""""""""""""""""""""""""""

x = np.arange(8)
path = "/batchdata"
with open(path,'rb') as f:
	data = pickle.load(f)
[t_acc,v_acc,t_risk,v_risk] = data
c = ["#ff9999", "#7777aa", "yellowgreen", "purple"]
title = ["Batch learning: trianing accuracy", "Batch learning: validation accuracy", "Batch learning: traning risk", "batch learning: validation risk"]
yaxis = ["Training Accuracy","Validation Accuracy", "Training Risk", "Validation Risk"]
xaxis = ["Epoch" for i in range(4)]
with open(path,'rb') as f:
	data = pickle.load(f)

for i in range(len(data)):
	plt.figure(i+10)
	plt.title(title[i],weight = "bold")
	plt.xlabel(xaxis[i],weight = "bold")
	plt.ylabel(yaxis[i],weight = "bold")
	d = np.array(data[i])
	plt.text(x[np.argwhere(d == np.max(d))], np.max(d), "max value = " + "{0:.4f}".format(np.max(d)), fontsize = 12)
	plt.text(x[np.argwhere(d == np.min(d))], np.min(d), "min value = " + "{0:.4f}".format(np.min(d)), fontsize = 12)
	plt.plot(x,d,c[i])
plt.show()







