### Deep Learning Assignment 1

### Name: Max Emanuel Feucht
### Student Number: 2742061

#%%

import math


#%%
######################
#### Forward Pass ####
######################

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
def argmax(array):
    array = list( array )
    return array.index(max(array))


def forward_pass(input_val, target_val, no_hidden, no_output, w, v, bw, bv, b):
    """
    Function that computes the forward pass through the network to give a loss and activations in the single layers

    Returns:
        loss: scalar indicating the loss
        y: softmax layer
        h: sigmoid activation layer
    """
    
    # Pass through first layer
    a = [0.]*no_hidden # Instantiate activation layer
    for i in range(len(input_val)):
        for j in range(no_hidden):
            a[j] += input_val[i] * w[i][j]
            
    # Adding Bias
    for j in range(no_hidden):
        a[j] += b * bw[j]
    
    # Obtaining final hidden layer through passing activation through sigmoid function
    h = [sigmoid(x) for x in a]
    
    # Pass through second layer
    o = [0.]*no_output
    for i in range(no_hidden):
        for j in range(no_output):
            o[j] += h[i] * v[i][j]
            
    # Adding Bias
    for j in range(no_output):
        o[j] += b * bv[j]

    # Calculate softmax
    y = []
    for i in range(no_output):
        y.append(math.exp(o[i]) / sum([math.exp(val) for val in o]))

    # Calculate loss for target class
    loss = 0
    loss = -math.log(y[target_val])
    
    pred = argmax(y)
        
    return loss, y, h, pred


#########################
##### Backward Pass #####
#########################



def backward_pass(input_val, target_val, y, h, v, bias, no_output, no_hidden):
    """
    Function that computes the derivatives for all weights by passing back through the network

    Returns:
        dL_dw: Derivative of the Loss wr to the weights in layer 1
        dL_dv: Derivative of the Loss wr to the weights in layer 2
        dL_dbw: Derivative of the Loss wr to the bias weights in layer 1
        dL_dbv: Derivative of the Loss wr to the bias weights in layer 2
    """

    t = [0,0]
    t[target_val] = 1
    
    dL_dw = [[0.,0.,0.],[0.,0.,0.]]
    dL_dv = [[0.,0.],[0.,0.],[0.,0.]]
    dL_dbw = [0.,0.,0.]
    dL_dbv = [0.,0.]
        
    # Backward pass through the network
    
    # Derivatives of Loss wr to output layer (before softmax)
    for i in range(no_output):
        
        dL_do = (y[i] - t[i]) # Derivative of Loss wrt the output neurons (before softmax)
        
        # Derivatives of Loss wr to weights in second layer (v), sigmoid activation layer (dL_dh), and activation before sigmoid (dL_da)
        for j in range(no_hidden):
            
            dL_dv[j][i] = round(dL_do * h[j],7) #dL_dv * hj
            dL_dh = dL_do * v[j][i] # dL_do * (do / dh) 
            dL_da = dL_dh * (h[j]*(1-h[j])) #dL_dh * (dh / da)
                        
            for k in range(len(input_val)):
                
                dL_dw[k][j] += round(dL_da * input_val[k],7) # += bc of multivariate chain rule: Summing over all output neurons
            dL_dbw[j] += round(dL_da * bias,7) # += bc of multivariate chain rule: Summing over all output neurons
                        
        dL_dbv[i] = round(bias * (y[i] - t[i]),7)
    
    return  dL_dw, dL_dv, dL_dbw, dL_dbv 




def update_weights(w, v, bw, bv, dL_dw, dL_dv, dL_dbw, dL_dbv, learning_rate):
    """
    Function to update the weights based on the derivatives computed with the backward function

    Returns:
        w: Updated weights for layer 1
        v: Updated weights for layer 2
        ubw: Updated bias weights for layer 1
        ubv: Updated bias weights for layer 2
    """
    
    for subset_w in range(len(w)):
        for i in range(len(w[subset_w])):
            w[subset_w][i] -= learning_rate * dL_dw[subset_w][i]

    for subset_v in range(len(v)):
        for i in range(len(v[subset_v])):
            v[subset_v][i] -= learning_rate * dL_dv[subset_v][i]
            
    ubw = [bw[i] - learning_rate*dL_dbw[i] for i in range(len(bw))]
    ubv = [bv[i] - learning_rate*dL_dbv[i] for i in range(len(bv))]

    return w, v, ubw, ubv


#%%

##########################
###### Example Pass ######
##########################

test_w = [[1.,1.,1.],[-1.,-1.,-1.]]
test_v = [1.,1.],[-1.,-1.],[-1.,-1.]
test_bw = [0.,0.,0.]
test_bv = [0.,0.]
bias = 1

no_hidden = 3
no_output = 2
test_target = 0
test_input = [1.,-1.]

test_loss, test_y, test_h, test_pred = forward_pass(input_val = test_input, target_val = test_target, no_hidden = no_hidden, no_output = no_output, w = test_w, v = test_v, bw = test_bw, bv = test_bv, b = bias)
test_dL_dw, test_dL_dv, test_dL_dbw, test_dL_dbv = backward_pass(input_val = test_input, target_val = test_target, y = test_y, h = test_h, v = test_v, bias = bias, no_output=no_output, no_hidden=no_hidden) 
test_w, test_v, test_bw, test_bv = update_weights(test_w, test_v, test_bw, test_bv, test_dL_dw, test_dL_dv, test_dL_dbw, test_dL_dbv, learning_rate = 1)

print("derivative w: {}".format(test_dL_dw))
print("derivative v: {}".format(test_dL_dv))
print("derivative bw: {}".format(test_dL_dbw))
print("derivative bv: {}".format(test_dL_dbv))


print("\n\n")

#%%

####################
#### LOAD DATA #####
####################

import data
import random 
from tqdm import tqdm
import matplotlib.pyplot as plt

(xtrain, ytrain), (xval, yval), num_cls = data.load_synth()

#%%
##########################
##### Training Loop ######
##########################

epochs = 5
learning_rate = 0.05

# Random initialization #
w = [[random.random() for i in range(3)],[random.random() for i in range(3)]]
v = [[random.random() for i in range(2)],[random.random() for i in range(2)],[random.random() for i in range(2)]]

bw = [0 for i in range(3)]
bv = [0 for i in range(2)]

no_hidden = 3
no_output = 2
bias = 1

loss_history = []
loss_history_avg = []

for e in range(epochs): 
    loss_history_epoch = []
    train_hits, train_misses = 0,0

    for x_,y_ in zip(xtrain, ytrain):
        loss, y, h, pred = forward_pass(input_val = x_, target_val = y_, no_hidden = no_hidden, no_output = no_output, w = w, v = v, bw = bw, bv = bv, b = bias)
        dL_dw, dL_dv, dL_dbw, dL_dbv = backward_pass(input_val = x_, target_val = y_, y = y, h = h, v = v, bias = bias, no_output=no_output, no_hidden=no_hidden) 
        w, v, bw, bv = update_weights(w, v, bw, bv, dL_dw, dL_dv, dL_dbw, dL_dbv, learning_rate = learning_rate)

        if pred == y_:
            train_hits += 1
        else:
            train_misses += 1
        
        loss_history_epoch.append(loss)
        loss_history.append(loss)
        
    loss_history_avg.append(sum(loss_history_epoch) / len(loss_history_epoch))
    print("Training Accuracy in epoch {}: {}".format(e, train_hits / (train_hits+train_misses)))
    print('Loss in epoch {}: {}'.format(e, sum(loss_history_epoch) / len(loss_history_epoch)))

window_size = 200

windows = [loss_history[i:i+window_size] for i in range(0,len(loss_history)-window_size,window_size)]
loss_avg = [sum(window)/window_size for window in windows]

plt.figure(figsize=(20,10))
plt.plot(loss_avg)
ticks = [round(i/window_size) for i in range(0, len(xtrain)*epochs + 1, window_size*10*epochs)]
plt.xticks(ticks = ticks, labels = [tick * window_size for tick in ticks], size = 18, rotation = 45)
plt.yticks(size = 18)
plt.title("\n Cross-Entropy Loss over training steps, \naveraged in intervals of {} iterations \n".format(window_size), size = 22)
plt.ylabel("\nCross-Entropy Loss\n", size = 18)
plt.xlabel("\nTraining steps\n", size = 18)
plt.show


# %%
#######################
##### Evaluation ######
#######################

train_hits, train_misses = 0,0
for x_, y_ in zip(xtrain, ytrain):
    loss, y, h, pred = forward_pass(input_val = x_, target_val = y_, no_hidden = no_hidden, no_output = no_output, w = w, v = v, bw = bw, bv = bv, b = bias)
    
    if pred == y_:
        train_hits += 1
    else:
        train_misses += 1
        
val_hits, val_misses = 0,0
for x_, y_ in zip(xval, yval):
    loss, y, h, pred = forward_pass(input_val = x_, target_val = y_, no_hidden = no_hidden, no_output = no_output, w = w, v = v, bw = bw, bv = bv, b = bias)
    
    if pred == y_:
        val_hits += 1
    else:
        val_misses += 1
        
print("Training Accuracy: {}".format(train_hits / (train_hits+train_misses)))
print("Validation Accuracy: {}".format(val_hits / (val_hits+val_misses)))
        

#%%

#######################################
############ Vectorization ############
#######################################

#####################
#### Load MNIST #####
#####################

import numpy as np
from mnist import MNIST

mndata = MNIST('samples')
train_imgs, train_labels = mndata.load_training()
test_imgs, test_labels = mndata.load_testing()

train_imgs, train_labels = np.asarray(train_imgs), np.asarray(train_labels)
test_imgs, test_labels = np.asarray(test_imgs), np.asarray(test_labels)

train_imgs = train_imgs / 255
test_imgs = test_imgs / 255


#%%

### Vectorized sigmoid and softmax function ###

class VectorizedMLP:
    
    def __init__(self, input_data, target_data, no_hidden, no_output, bias, lr, epochs, plot = False, window_size = 200, validation_input = None, validation_target = None, initialization = "normal"):
        self.input = input_data
        self.target = target_data
        self.no_hidden = no_hidden
        self.no_output = no_output
        self.lr = lr
        self.bias = bias
        self.epochs = epochs
        self.plot = plot
        self.window_size = window_size
        self.validation_input = validation_input
        self.validation_target = validation_target
        self.initialization = initialization
    
    def instantiate_weights(self):
        
        if self.initialization == "normal":
            self.w = np.random.normal(size = (self.input.shape[1], self.no_hidden))
            self.v = np.random.normal(size = (self.no_hidden, self.no_output))
            self.bw = np.random.normal(size = (self.no_hidden))
            self.bv = np.random.normal(size = (self.no_output))
        elif self.initialization == "he":
            self.w = np.random.normal(size = (self.input.shape[1], self.no_hidden)) * np.sqrt(2 / self.no_hidden)
            self.v = np.random.normal(size = (self.no_hidden, self.no_output))  * np.sqrt(2 / self.no_output)
            self.bw = np.random.normal(size = (self.no_hidden)) * np.sqrt(2 / self.no_hidden)
            self.bv = np.random.normal(size = (self.no_output)) * np.sqrt(2 / self.no_output)
        elif self.initialization == "xavier":
            self.w = np.random.uniform(low = -np.sqrt(6) / np.sqrt(self.input.shape[1] + self.no_hidden), high = -np.sqrt(6) / np.sqrt(self.input.shape[1] + self.no_hidden), size = (self.input.shape[1], self.no_hidden))
            self.v = np.random.uniform(low = -np.sqrt(6) / np.sqrt(self.no_hidden + self.no_output), high = np.sqrt(6) / np.sqrt(self.no_hidden + self.no_output), size = (self.no_hidden, self.no_output))
            self.bw = np.random.uniform(low = -np.sqrt(6) / np.sqrt(1 + self.no_hidden), high = np.sqrt(6) / np.sqrt(1 + self.no_hidden), size = (self.no_hidden))
            self.bv = np.random.uniform(low = -np.sqrt(6) / np.sqrt(1 + self.no_output), high = np.sqrt(6) / np.sqrt(1 + self.no_output), size = (self.no_output))
            
    def sigmoid_vec(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax_vec(self, x):
        exps = np.exp(x)
        return exps/sum(exps)

    def forward(self):
        self.a = np.matmul(self.single_input, self.w) + self.bias * self.bw
        self.h = self.sigmoid_vec(self.a)
        self.o = np.matmul(self.h, self.v) + self.bias * self.bv
        self.y = self.softmax_vec(self.o)
        self.loss = -np.log(self.y[self.single_target])
        self.pred = np.argmax(self.y)    
    
    def val_forward(self):
        self.val_a = np.matmul(self.val_img, self.w) + self.bias * self.bw
        self.val_h = self.sigmoid_vec(self.val_a)
        self.val_o = np.matmul(self.val_h, self.v) + self.bias * self.bv
        self.val_y = self.softmax_vec(self.val_o)
        self.val_loss = -np.log(self.val_y[self.val_target])
        self.val_pred = np.argmax(self.val_y)   
    
    def backward(self):
        self.t = np.zeros(self.no_output)
        self.t[self.single_target] = 1

        self.dL_do = self.y - self.t
        self.dL_dv = np.outer(self.h, self.dL_do)
        self.dL_dbv = np.squeeze(np.outer(self.bias, self.dL_do))
        self.dL_dh = np.matmul(self.v, self.dL_do)
        self.dL_da = self.dL_dh * (self.h*(1-self.h))
        self.dL_dw = np.outer(self.single_input, self.dL_da)
        self.dL_dbw = np.squeeze(np.outer(self.bias, self.dL_da))
    
    def update(self):
        self.w -= self.lr * self.dL_dw
        self.v -= self.lr * self.dL_dv
        self.bw -= self.lr * self.dL_dbw
        self.bv -= self.lr * self.dL_dbv
    
    def train(self):
        
        self.instantiate_weights()
        self.loss_history = []
        self.loss_history_avg = []
        self.loss_history_avg_val = []

        for e in range(self.epochs): 
            self.loss_history_epoch = []
            self.train_hits, self.train_misses = 0,0
            
            # Shuffle Data randomly every epoch
            p = np.random.permutation(self.input.shape[0])
            self.input, self.target = self.input[p], self.target[p]
            
            for input_img, target in tqdm(zip(self.input, self.target)):
                self.single_input, self.single_target = input_img, target
                self.forward()
                self.backward()
                self.update()

                if self.pred == self.single_target:
                    self.train_hits += 1
                else:
                    self.train_misses += 1
                
                self.loss_history_epoch.append(self.loss)
                self.loss_history.append(self.loss)
                
            self.train_accuracy = self.train_hits / (self.train_hits+self.train_misses)
            self.loss_history_avg.append(sum(self.loss_history_epoch) / len(self.loss_history_epoch))
            print("Training Accuracy during epoch {}: {}".format(e, self.train_accuracy))
            print('Training Loss during epoch {}: {}\n'.format(e, sum(self.loss_history_epoch) / len(self.loss_history_epoch)))
            
            # If validation data is given, evaluate the performance after training for both the training and the validation set
            if self.validation_input is not None:
                self.train_hits, self.train_misses, self.val_hits, self.val_misses = 0,0,0,0
                
                self.loss_history_eval = []
                for input_img, target in zip(self.input, self.target):
                    self.single_input, self.single_target = input_img, target
                    self.forward()
                    if self.pred == self.single_target:
                        self.train_hits += 1
                    else:
                        self.train_misses += 1
                    self.loss_history_eval.append(self.loss)
                    
                self.loss_history_eval_val = []
                for val_img, val_target in zip(self.validation_input, self.validation_target):
                    self.val_img, self.val_target = val_img, val_target
                    self.val_forward()
                    if self.val_pred == self.val_target:
                        self.val_hits += 1
                    else:
                        self.val_misses += 1
                    self.loss_history_eval_val.append(self.val_loss)

                self.train_accuracy = self.train_hits / (self.train_hits+self.train_misses)
                self.val_accuracy = self.val_hits / (self.val_hits+self.val_misses)

                print("Training Accuracy after epoch {}: {}".format(e, self.train_hits / (self.train_hits+self.train_misses)))
                print("Validation Accuracy after epoch {}: {}".format(e, self.val_hits / (self.val_hits+self.val_misses)))
                print('Training Loss after epoch {}: {}'.format(e, sum(self.loss_history_eval) / len(self.loss_history_eval)))
                print('Validaiton Loss after epoch {}: {}'.format(e, sum(self.loss_history_eval_val) / len(self.loss_history_eval_val)))
        
        if self.plot:
                            
            window_size = 200

            windows = [self.loss_history[i:i+self.window_size] for i in range(0,len(self.loss_history)-self.window_size,self.window_size)]
            loss_avg = [sum(window)/window_size for window in windows]

            plt.figure(figsize=(20,10))
            plt.plot(loss_avg)
            ticks = [round(i/self.window_size) for i in range(0, self.input.shape[0]*self.epochs + 1, self.window_size*10*epochs)]
            plt.xticks(ticks = ticks, labels = [tick * self.window_size for tick in ticks], size = 18, rotation = 45)
            plt.yticks(size = 18)
            plt.title("\n Cross-Entropy Loss over training steps, \nin vectorized MLP\n", size = 22)
            plt.ylabel("\nCross-Entropy Loss\n", size = 18)
            plt.xlabel("\nTraining steps\n", size = 18)
            plt.show
            
            return self.loss_history, self.train_accuracy, self.loss, plt
        
        else:
            return self.loss_history, self.train_accuracy, self.loss



vmlp = VectorizedMLP(train_imgs, train_labels, no_hidden = 300, no_output = 10, bias = 1, lr = 0.05, epochs = 1, plot = True, validation_input=test_imgs, validation_target=test_labels)
loss_history, accuracy, loss, plot = vmlp.train()





#%%

### Vectorized sigmoid and softmax function ###

class BatchedVectorizedMLP:
    
    def __init__(self, input_data, target_data, batch_size, no_hidden, no_output, bias, lr, epochs, plot = False, window_size = 200, normalize = True):
        self.input = input_data
        self.target = target_data
        self.batch_size = batch_size
        self.no_hidden = no_hidden
        self.no_output = no_output
        self.lr = lr
        self.bias = bias
        self.epochs = epochs
        self.plot = plot
        self.window_size = window_size
        self.normalize = normalize
    
    def instantiate_weights(self):
        self.w = np.array(np.random.normal(size = (self.input.shape[1], self.no_hidden)))
        self.v = np.array(np.random.normal(size = (self.no_hidden, self.no_output)))
        self.bw = np.array(np.random.normal(size = (self.no_hidden)))
        self.bv = np.array(np.random.normal(size = (self.no_output)))
        
    def sigmoid_vec(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax_vec(self, x):
        exps = np.exp(x)
        return exps/sum(exps)

    def forward(self):
        
        self.a = np.matmul(self.single_input, self.w) + self.bias[0] * self.bw
        self.h = self.sigmoid_vec(self.a)
        self.o = np.matmul(self.h, self.v) + self.bias[0] * self.bv
        self.y = self.softmax_vec(self.o)
        self.loss = -np.log([self.y[i][t] for i,t in zip(range(self.y.shape[0]), self.single_target)])
        self.pred = np.argmax(self.y, axis = 1)


    def backward(self):

        # Batched one-hot encoding
        self.t = np.zeros((self.batch_size, self.no_output))
        self.t[np.arange(self.single_target.size), self.single_target] = 1

        self.dL_do = self.y - self.t
        self.dL_dv = self.dL_do[:,np.newaxis,:] * self.h[:,:,np.newaxis]
        self.dL_dbv = np.squeeze(self.dL_do[:,np.newaxis,:] * np.repeat(self.bias[np.newaxis,:, np.newaxis], self.batch_size, axis = 0))
        self.dL_dh = np.matmul(np.repeat(self.v[np.newaxis,:,:], self.batch_size, axis = 0),self.dL_do[:,:,None]).squeeze(-1)
        self.dL_da = self.dL_dh * (self.h*(1-self.h))
        self.dL_dw = self.dL_da[:,np.newaxis,:] * self.single_input[:,:,np.newaxis]
        self.dL_dbw = np.squeeze(self.dL_da[:,np.newaxis,:] * np.repeat(self.bias[np.newaxis,:,np.newaxis], self.batch_size, axis = 0))


    def update(self):
        
        # Sum over batch axis:
        self.dL_dw, self.dL_dv, self.dL_dbw, self.dL_dbv = np.sum(self.dL_dw, axis = 0), np.sum(self.dL_dv, axis = 0), np.sum(self.dL_dbw, axis = 0), np.sum(self.dL_dbv, axis = 0)
        
        self.w = self.w - self.lr * self.dL_dw
        self.v = self.v - self.lr * self.dL_dv
        self.bw = self.bw - self.lr * self.dL_dbw
        self.bv = self.bv - self.lr * self.dL_dbv
        
        # Normalize Weights
        if self.normalize: 
            self.w = ( self.w - np.mean( self.w)) / np.std( self.w)
            self.v = (self.v - np.mean(self.v)) / np.std(self.v)
            self.bw = (self.bw - np.mean(self.bw)) / np.std(self.bw)
            self.bv = (self.bv - np.mean(self.bv)) / np.std(self.bv)

    
    def train(self):
        
        self.instantiate_weights()
        self.loss_history = []
        self.loss_history_avg = []

        for e in range(self.epochs): 
            self.loss_history_epoch = []
            self.train_hits, self.train_misses = 0,0
            
            # Shuffle Data every epoch
            p = np.random.permutation(self.input.shape[0])
            self.input, self.target = self.input[p], self.target[p]   
            
            for i in tqdm(range(0, self.input.shape[0], self.batch_size)):
                self.single_input, self.single_target = self.input[i:i+self.batch_size], self.target[i:i+self.batch_size]
                self.forward()
                self.backward()
                self.update()

                for p,t in zip(self.pred, self.single_target): 
                    if p == t:
                        self.train_hits += 1
                    else:
                        self.train_misses += 1
                
                self.loss_history_epoch.append(np.average(self.loss))
                self.loss_history.append(np.average(self.loss))
                
            self.loss_history_avg.append(sum(self.loss_history_epoch) / len(self.loss_history_epoch))
            print("Training Accuracy in epoch {}: {}".format(e, self.train_hits / (self.train_hits+self.train_misses)))
            print('Loss in epoch {}: {}'.format(e, sum(self.loss_history_epoch) / len(self.loss_history_epoch)))

        self.accuracy = self.train_hits / (self.train_hits+self.train_misses)
        
    
        if self.plot:
            self.window_size = int(320 / self.batch_size)

            windows = [self.loss_history[i:i+self.window_size] for i in range(0,len(self.loss_history)-self.window_size,self.window_size)]
            loss_avg = [sum(window)/window_size for window in windows]

            plt.figure(figsize=(20,10))
            plt.plot(loss_avg)
            ticks = [round(i/self.window_size) for i in range(0, int((self.input.shape[0]*self.epochs) / self.batch_size) + 10, self.window_size*10*self.epochs)]
            plt.xticks(ticks = ticks, labels = [tick * self.window_size * self.batch_size for tick in ticks], size = 18, rotation = 45)
            plt.yticks(size = 18)
            plt.title("\n Cross-Entropy Loss over training steps, \nin batched vectorized MLP\n", size = 22)
            plt.ylabel("\nCross-Entropy Loss\n", size = 18)
            plt.xlabel("\nTraining steps\n", size = 18)
            plt.show
            
            return self.loss_history, self.accuracy, self.loss, plt
        
        else:
            return self.loss_history, self.accuracy, self.loss
        

bvmlp = BatchedVectorizedMLP(train_imgs, train_labels, batch_size = 16, no_hidden = 300, no_output = 10, bias = np.array([1]), lr = 0.05, epochs = 1, plot = True)
loss_history, accuracy, loss, plot = bvmlp.train()


#%%

##################
#### Analysis ####
##################

# 5 epochs, unbatched NN

# 1. Training vs. Validation Loss: 

vmlp = VectorizedMLP(train_imgs, train_labels, no_hidden = 300, no_output = 10, bias = 1, lr = 0.05, epochs = 5, plot = True, validation_input = test_imgs, validation_target= test_labels)
loss_history, train_accuracy, loss, plot = vmlp.train()

# Training Accuracy during epoch 0: 0.8888833333333334
# Training Loss during epoch 0: 0.629543759074927

# Training Accuracy after epoch 0: 0.9258
# Validation Accuracy after epoch 0: 0.9151
# Training Loss after epoch 0: 0.3460667195609362
# Validaiton Loss after epoch 0: 0.43054193128357277


# Training Accuracy during epoch 1: 0.9439
# Training Loss during epoch 1: 0.24780257281749848

# Training Accuracy after epoch 1: 0.95395
# Validation Accuracy after epoch 1: 0.9429
# Training Loss after epoch 1: 0.18447736967482747
# Validaiton Loss after epoch 1: 0.28008523590355666


# Training Accuracy during epoch 2: 0.96285
# Training Loss during epoch 2: 0.14680742239581288

# Training Accuracy after epoch 2: 0.9680166666666666
# Validation Accuracy after epoch 2: 0.9506
# Training Loss after epoch 2: 0.1173963605630428
# Validaiton Loss after epoch 2: 0.23152203027219725


# Training Accuracy during epoch 3: 0.9739
# Training Loss during epoch 3: 0.09546210435081418

# Training Accuracy after epoch 3: 0.9701333333333333
# Validation Accuracy after epoch 3: 0.9514
# Training Loss after epoch 3: 0.10438129014810815
# Validaiton Loss after epoch 3: 0.2337747130676738


# Training Accuracy during epoch 4: 0.98125
# Training Loss during epoch 4: 0.06575311849236208

# Training Accuracy after epoch 4: 0.9757666666666667
# Validation Accuracy after epoch 4: 0.9532
# Training Loss after epoch 4: 0.07781246248154385
# Validaiton Loss after epoch 4: 0.2281844200135179



#%%

# 2. Initialize 5 times and analyze how loss and / or accuracy depend on random initialization

epochs = 5
window_size = 200
loss_histories = np.zeros(shape = (int((epochs * train_imgs.shape[0])/window_size)-1,0))

for i in range(5):
    vmlp = VectorizedMLP(train_imgs, train_labels, no_hidden = 300, no_output = 10, bias = 1, lr = 0.05, epochs = epochs, plot = False)
    loss_history, accuracy, loss = vmlp.train()
    windows = [loss_history[i:i+window_size] for i in range(0,len(loss_history)-window_size,window_size)]
    loss_avg = [sum(window)/window_size for window in windows]
    loss_histories = np.hstack((loss_histories, np.array(loss_avg)[:,np.newaxis]))


mean_loss = np.mean(loss_histories, axis = 1)
std_loss = np.std(loss_histories, axis = 1)

plt.figure(figsize=(20,10))
plt.plot(mean_loss)
plt.fill_between(range(len(mean_loss)), mean_loss - 3*std_loss, mean_loss + 3*std_loss, alpha = 0.5)
ticks = [round(i/window_size) for i in range(0, int((train_imgs.shape[0]*epochs)) + 1, window_size*10*epochs)]
plt.xticks(ticks = ticks, labels = [tick * window_size for tick in ticks], size = 18, rotation = 45)
plt.yticks(size = 18)
plt.title("\n Average Loss over training steps for 5 epochs, \nrepeated 5 times\n", size = 22)
plt.ylabel("\nCross-Entropy Loss\n", size = 18)
plt.xlabel("\nTraining steps\n", size = 18)
plt.show


#%%
# 3. Analyze different LRs: 0.001, 0.003, 0.01, 0.03, 0.1

epochs = 5
window_size = 200

lr_histories = np.zeros(shape = (int((epochs * train_imgs.shape[0])/window_size)-1,0))
    
for lr in (0.001, 0.003, 0.01, 0.03, 0.05, 0.1):
    vmlp = VectorizedMLP(train_imgs, train_labels, no_hidden = 300, no_output = 10, bias = 1, lr = lr, epochs = epochs, plot = True, validation_input = test_imgs, validation_target = test_labels)
    loss_history, train_accuracy, loss, plot = vmlp.train()
    windows = [loss_history[i:i+window_size] for i in range(0,len(loss_history)-window_size,window_size)]
    loss_avg = [sum(window)/window_size for window in windows]
    lr_histories = np.hstack((lr_histories, np.array(loss_avg)[:,np.newaxis]))
    

plt.figure(figsize=(20,10))
plt.plot(lr_histories[:,0],label = "Learning Rate 0.001")
plt.plot(lr_histories[:,1],label = "Learning Rate 0.003")
plt.plot(lr_histories[:,2],label = "Learning Rate 0.01")
plt.plot(lr_histories[:,3],label = "Learning Rate 0.03")
plt.plot(lr_histories[:,4],label = "Learning Rate 0.05")
plt.plot(lr_histories[:,5],label = "Learning Rate 0.1")
plt.legend()
ticks = [round(i/window_size) for i in range(0, int((train_imgs.shape[0]*epochs)) + 1, window_size*10*epochs)]
plt.xticks(ticks = ticks, labels = [tick * window_size for tick in ticks], size = 18, rotation = 45)
plt.yticks(size = 18)
plt.ylim(0, 10)
plt.title("\n Loss over training steps for 5 epochs with different Learning Rates, \nrepeated 5 times\n", size = 22)
plt.ylabel("\nCross-Entropy Loss\n", size = 18)
plt.xlabel("\nTraining steps\n", size = 18)
plt.show


# 0.001
# Training Accuracy after epoch 4: 0.89225
# Validation Accuracy after epoch 4: 0.8837
# Training Loss after epoch 4: 0.47108905546183455
# Validaiton Loss after epoch 4: 0.5192711550632148

# 0.003
# Training Accuracy after epoch 4: 0.9285
# Validation Accuracy after epoch 4: 0.9067
# Training Loss after epoch 4: 0.27027447544287675
# Validaiton Loss after epoch 4: 0.3826532005916663

# 0.01
# Training Accuracy after epoch 4: 0.9594166666666667
# Validation Accuracy after epoch 4: 0.9292
# Training Loss after epoch 4: 0.13228780501133178
# Validaiton Loss after epoch 4: 0.27272357029303496

# 0.03
# Training Accuracy after epoch 4: 0.9808333333333333
# Validation Accuracy after epoch 4: 0.9551
# Training Loss after epoch 4: 0.059155131021803815
# Validaiton Loss after epoch 4: 0.18040242912540935

#0.05
# Training Accuracy after epoch 4: 0.9757666666666667
# Validation Accuracy after epoch 4: 0.9532
# Training Loss after epoch 4: 0.07781246248154385
# Validaiton Loss after epoch 4: 0.2281844200135179

# 0.1
# Training Accuracy after epoch 4: 0.9674333333333334
# Validation Accuracy after epoch 4: 0.9539
# Training Loss after epoch 4: 0.1439820945597458
# Validaiton Loss after epoch 4: 0.2702988548674491


#%%
# 4. Try other Hyperparameters

## Different initialization method

epochs = 5
window_size = 200

#init_histories = np.zeros(shape = (int((epochs * train_imgs.shape[0])/window_size)-1,0))
for init in ("normal", "xavier", "he"):
    vmlp = VectorizedMLP(train_imgs, train_labels, no_hidden = 300, no_output = 10, bias = 1, lr = 0.05, epochs = epochs, plot = True, validation_input = test_imgs, validation_target= test_labels, initialization = init)
    loss_history, train_accuracy, loss, plot = vmlp.train()
    windows = [loss_history[i:i+window_size] for i in range(0,len(loss_history)-window_size,window_size)]
    loss_avg = [sum(window)/window_size for window in windows]
    init_histories = np.hstack((init_histories, np.array(loss_avg)[:,np.newaxis]))
    
plt.figure(figsize=(20,10))
plt.plot(init_histories[:,0],label = "Normal Initialization")
plt.plot(init_histories[:,1],label = "Xavier Initialization")
plt.plot(init_histories[:,2],label = "He Initialization")
plt.legend()
ticks = [round(i/window_size) for i in range(0, int((train_imgs.shape[0]*epochs)) + 1, window_size*10*epochs)]
plt.xticks(ticks = ticks, labels = [tick * window_size for tick in ticks], size = 18, rotation = 45)
plt.yticks(size = 18)
plt.title("\n Loss over training steps for 5 epochs with different Weight initialization methods, \nrepeated 5 times\n", size = 22)
plt.ylabel("\nCross-Entropy Loss\n", size = 18)
plt.xlabel("\nTraining steps\n", size = 18)
plt.legend()
plt.show


#%%

# Final Model 

vmlp = VectorizedMLP(train_imgs, train_labels, no_hidden = 300, no_output = 10, bias = 1, lr = 0.05, epochs = 5, plot = True, validation_input = test_imgs, validation_target= test_labels, initialization = "he")
vmlp.train()

# Training Accuracy during epoch 4: 0.9886
# Training Loss during epoch 4: 0.038374630222884724

# Training Accuracy after epoch 4: 0.9911333333333333
# Validation Accuracy after epoch 4: 0.9783
# Training Loss after epoch 4: 0.03071671679810386
# Validaiton Loss after epoch 4: 0.07212814515228178

#%%



