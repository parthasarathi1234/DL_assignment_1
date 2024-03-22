from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.metrics import confusion_matrix
# !pip install -U wandb
import wandb
from wandb.keras import WandbCallback
import socket
import argparse
socket.setdefaulttimeout(30)
wandb.login()
wandb.init(project="cs23m035_DL_Assignment1")

def images():
    (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    x_train_images, x_validation_images,y_train_labels, y_validation_labels  = train_test_split(x_train,y_train,test_size = 0.1)
    n=0
    i=0
    while(n!=10):
    if(y_train[i]==n):
        plt.subplot(2,5,n+1)
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train[i])
        plt.xlabel(y_train[i])
        n=n+1
    i=i+1
    wandb.log({'plt':plt})
    plt.show()


def weight(No_of_layers,neurons,weightInit):
  weights=[]
  bias=[]

  if(weightInit=='random'):
  # input layer
    weights.append(0.01*np.random.randn(neurons,784))
    bias.append(0.01*np.random.randn(neurons,1))

    # Hidden layer
    for i in range(1,No_of_layers):
      weights.append(0.01*np.random.randn(neurons,neurons))
      bias.append(0.01*np.random.randn(neurons,1))

    # output layer
    weights.append(0.01*np.random.randn(10,neurons))
    bias.append(0.01*np.random.randn(10,1))

  elif(weightInit=='Xavier'):
    # input layer
    weights.append(0.01*np.random.randn(neurons,784))
    bias.append(np.zeros((neurons,1)))

    # Hidden layer
    for i in range(1,No_of_layers):
      weights.append(0.01*np.random.randn(neurons,neurons))
      bias.append(np.zeros((neurons,1)))

    # output layer
    weights.append(0.01*np.random.randn(10,neurons))
    bias.append(np.zeros((10,1)))

  else:
    weights.append(np.random.randn(neurons,784))
    bias.append(np.random.randn(neurons,1))

    # Hidden layer
    for i in range(1,No_of_layers):
      weights.append(np.random.randn(neurons,neurons))
      bias.append(np.random.randn(neurons,1))

    # output layer
    weights.append(0.01*np.random.randn(10,neurons))
    bias.append(0.01*np.random.randn(10,1))


  return weights,bias



def sigmoid(a):
  z=np.clip(a,-500, 500)
  return 1 / (1 + np.exp(-z))

def tanh(a):
  z=np.clip(a,-50,50)
  return np.tanh(z)

def softmax(a):
  x=1e-6
  return (np.exp(a-max(a))/(sum(np.exp(a-max(a)))+x))

def tanh_derivative(z):
  return 1-np.tanh(z)**2

def relu(Z):
    A = np.maximum(0, Z)
    return A

def relu_derivative(z):
  return np.where(z>0,1,0)

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def identity(a):
  return a

def identity_derivative(a): 
  res = np.ones(a.shape)
  return res



def  feeb_forward(x,layers,act,wei,b):
  pre_activation=[0 for i in range(layers+1)]
  activation=[0 for i in range(layers+1)]
  z=x.reshape(784,1)/255.0
  for i in range(layers+1):
    if(i==0):
      pre_activation[i]=np.matmul(wei[i],z)+b[i]
    elif(i==layers):
        pre_activation[i]=b[i]+np.matmul(wei[i],z)
        activation[i]=np.copy(softmax(pre_activation[i]))
        continue
    else:
      pre_activation[i]=b[i]+np.matmul(wei[i],z)  # pre activation
    if(act=="sigmoid"):
      z=sigmoid(pre_activation[i])
    elif(act=="tanh"):
      z=tanh(pre_activation[i])
    elif(act=="ReLU"):
      z=relu(pre_activation[i])
    else:
      z=identity(pre_activation[i])
    activation[i]=z
  return pre_activation,activation,z

def back_propagation(x_t,y_train,z,H,A,layers,W,activation,loss_Fun):
  update_x=x_t.reshape(784,1)/255.0
  dw=[0 for i in range(layers+1)]
  db=[0 for i in range(layers+1)]
  y=np.zeros((10,1))
  y[y_train]=1
  if(loss_Fun=='cross_entropy'):
    loss=-(y-H[layers])
  else:
    loss=(H[layers]-y)*H[layers]*(1-H[layers])
  for k in range(len(W)-1,-1,-1):
    if(k==0):
        dw[0]=np.matmul(loss,update_x.T)
        db[0]=np.copy(loss)
        continue
    dw[k]=np.matmul(loss,(H[k-1].T))
    db[k]=np.copy(loss)
    loss_dh=np.matmul((W[k].T),loss)
    if(activation=="tanh"):
      x=tanh_derivative(A[k-1])
    elif(activation=="sigmoid"):
      x=sigmoid_derivative(A[k-1])
    elif(activation=="ReLU"):
      x=relu_derivative(A[k-1])
    else:
      x=identity_derivative(A[k-1])
    loss=np.multiply(loss_dh,x)
  return dw,db

def accuracy(theta_w,theta_b,X,Y,activationfun,layers,loss_Fun):
  count=0
  loss=0
  for train_ima,train_labe in zip(X,Y):
    p_act, act, z = feeb_forward(train_ima, layers, activationfun, theta_w, theta_b)
    if(np.argmax(act[layers])==train_labe):
      count=count+1
    if(loss_Fun=='cross_entropy'):
      loss+=-np.log(act[layers])[train_labe][0]
    else:
      loss+=(np.argmax(act[layers]-Y.shape[0])**2)
    loss=loss/Y.shape[0]
    acc=(count/Y.shape[0])
  return acc*100,loss*100

def accuracy_confusion(theta_w,theta_b,X,Y,activationfun,layers,loss_Fun,predicted,original):
  count=0
  loss=0
  for train_ima,train_labe in zip(X,Y):
    p_act, act, z = feeb_forward(train_ima, layers, activationfun, theta_w, theta_b)
    original.append(train_labe)
    predicted.append(np.argmax(act[layers]))
    if(np.argmax(act[layers])==train_labe):
      count=count+1
    if(loss_Fun=='cross_entropy'):
      loss+=-np.log(act[layers])[train_labe][0]
    else:
      loss+=(np.argmax(act[layers]-Y.shape[0])**2)
    loss=loss/Y.shape[0]
    acc=(count/Y.shape[0])
  return acc*100,loss*100



def stochastic_gradient_descent(epochs,eta,layers,neurons,activation_fun,weightInit,weightDecay,loss_Fun):
  W, B = weight(layers,neurons,weightInit)
  for i in range(epochs):
    dw=[[0 for colu in range(row)] for row in range(len(W))]
    db=[[0 for colu in range(row)] for row in range(len(B))]

    for train_ima,train_labe in zip(x_train_images,y_train_labels):
      A,H,Z=feeb_forward(train_ima,layers,activation_fun,W,B)
      current_dw,current_db=back_propagation(train_ima,train_labe,Z,H,A,layers,W,activation_fun,loss_Fun)

      for k in range(layers+1):
        W[k]=W[k]-eta*current_dw[k]-(weightDecay*W[k])
        B[k]=B[k]-eta*current_db[k]

    acc,loss=accuracy(W,B,x_train_images,y_train_labels,activation_fun,layers,loss_Fun)
    v_acc,v_loss=accuracy(W,B,x_validation_images,y_validation_labels,activation_fun,layers,loss_Fun)
    print("Accuracy")
    print(acc,loss,v_acc,v_loss)
    wandb.log({"Train_Accuracy" : acc,"Train_Loss" : loss,"Validation_acc" : v_acc,"validation_loss" : v_loss,'epoch':i})
    return W,B


def momentum_gradient_descent(epochs,eta,layers,neurons,activation_fun,batchSize,weightInit,weightDecay,loss_Fun,beta):
  W,B=weight(layers,neurons,weightInit)
  pre_v_w=[0 for i in range(layers+1)]
  pre_v_b=[0 for i in range(layers+1)]

  for i in range(epochs):
    batch=1
    dw=[0 for i in range(layers+1)]
    db=[0 for i in range(layers+1)]
    for train_ima,train_labe in zip(x_train_images,y_train_labels):
      A,H,Z=feeb_forward(train_ima,layers,activation_fun,W,B)
      current_dw, current_db = back_propagation(train_ima,train_labe,Z,H,A,layers,W,activation_fun,loss_Fun)

      for k in range(len(W)):
        dw[k]+=current_dw[k]
        db[k]+=current_db[k]

      if(batch%batchSize==0):
        for k in range(len(W)):
          v_w=beta*pre_v_w[k]+dw[k]
          v_b=beta*pre_v_b[k]+db[k]

          W[k]=W[k]-(eta*v_w) - (weightDecay * W[k])
          B[k]=B[k]-eta*v_b

          pre_v_w[k]=v_w
          pre_v_b[k]=v_b

        for xw in dw:
          xw[:]=0
        for xb in db:
          xb[:]=0
      batch+=1

    acc,loss=accuracy(W,B,x_train_images,y_train_labels,activation_fun,layers,loss_Fun)
    v_acc,v_loss=accuracy(W,B,x_validation_images,y_validation_labels,activation_fun,layers,loss_Fun)
    print("Accuracy")
    print(acc,loss,v_acc,v_loss)
    wandb.log({"Train_Accuracy" : acc,"Train_Loss" : loss,"Validation_acc" : v_acc,"validation_loss" : v_loss,'epoch':i})
    return W,B


def nesterov_accelerated_gradient_descent(epochs,layers,neurons,eta,activation_fun,batchSize,weightInit,weightDecay,loss_Fun,beta):
  W,B=weight(layers,neurons,weightInit)
  pre_v_w=[0 for i in range(layers+1)]
  pre_v_b=[0 for i in range(layers+1)]

  v_w=[0 for i in range(layers+1)]
  v_b=[0 for i in range(layers+1)]

  for i in range(epochs):
    batch=1
    dw=[0 for i in range(layers+1)]
    db=[0 for i in range(layers+1)]

    for k in range(len(W)):
      W[k] = W[k] - beta * pre_v_w[k]
      B[k] = B[k] - beta * pre_v_b[k]
    for train_ima,train_labe in zip(x_train_images,y_train_labels):
      A,H,Z=feeb_forward(train_ima,layers,activation_fun,W,B)
      current_dw,current_db=back_propagation(train_ima,train_labe,Z,H,A,layers,W,activation_fun,loss_Fun)

      for k in range(len(W)):
          dw[k]+=current_dw[k]
          db[k]+=current_db[k]

      if(batch%batchSize==0):
        for k in range(len(W)):
          W[k]=W[k]-eta*dw[k]-(weightDecay*W[k])
          B[k]=B[k]-eta*db[k]
          pre_v_w[k]=eta*dw[k]+beta*pre_v_w[k]
          pre_v_b[k]=eta*db[k]+beta*pre_v_b[k]
        for xw in dw:
          xw[:]=0
        for xb in db:
          xb[:]=0
      batch+=1
    acc, loss = accuracy(W,B,x_train_images,y_train_labels,activation_fun,layers,loss_Fun)
    v_acc, v_loss = accuracy(W,B,x_validation_images,y_validation_labels,activation_fun,layers,loss_Fun)
    print("Accuracy")
    print(acc,loss,v_acc,v_loss)
    wandb.log({"Train_Accuracy" : acc,"Train_Loss" : loss,"Validation_acc" : v_acc,"validation_loss" : v_loss,'epoch':i})
    return W,B


def rms_prop(epochs,layers,neurons,eta,activation_fun,batchSize,weightInit,weightDecay,loss_Fun,rms_beta):
  beta=rms_beta
  W,B=weight(layers,neurons,weightInit)
  eps=1e-4
  v_w=[0 for i in range(layers+1)]
  v_b=[0 for i in range(layers+1)]

  for i in range(epochs):
    batch=1
    dw=[0 for i in range(layers+1)]
    db=[0 for i in range(layers+1)]
    for train_ima,train_labe in zip(x_train_images,y_train_labels):
      A, H, Z = feeb_forward(train_ima,layers,activation_fun,W,B)
      current_dw, current_db = back_propagation(train_ima,train_labe,Z,H,A,layers,W,activation_fun,loss_Fun)

      for k in range(len(W)):
        dw[k]=dw[k]+current_dw[k]
        db[k]=db[k]+current_db[k]
      if(batch%batchSize==0):
        for k in range(len(W)):
          v_w[k]=beta*v_w[k]+(1-beta)*pow(dw[k],2)
          v_b[k]=beta*v_b[k]+(1-beta)*pow(db[k],2)
          W[k]=W[k]-eta*dw[k]/(np.sqrt(v_w[k])+eps) -(weightDecay*W[k])
          B[k]=B[k]-eta*db[k]/(np.sqrt(v_b[k])+eps)

        for xw in dw:
          xw[:]=0
        for xb in db:
          xb[:]=0
      batch+=1

    acc,loss=accuracy(W,B,x_train_images,y_train_labels,activation_fun,layers,loss_Fun)
    v_acc,v_loss=accuracy(W,B,x_validation_images,y_validation_labels,activation_fun,layers,loss_Fun)
    print("Accuracy")
    print(acc,loss,v_acc,v_loss)
    wandb.log({"Train_Accuracy" : acc,"Train_Loss" : loss,"Validation_acc" : v_acc,"validation_loss" : v_loss,'epoch':i})
    return W,B

def adam(epochs,eta,activation_fun,layers,neurons,batchSize,weightInit,weightDecay,loss_Fun,beta1,beta2):
  W,B=weight(layers,neurons,weightInit)
  v_w=[np.zeros_like(w) for w in W]
  v_b=[np.zeros_like(w) for w in B]
  m_w=[np.zeros_like(w) for w in W]
  m_b=[np.zeros_like(w) for w in B]
  eps=1e-10
  for i in range(epochs):
    batch=1
    dw=[np.zeros_like(w) for w in W]
    db=[np.zeros_like(w) for w in B]

    for train_ima,train_labe in zip(x_train_images,y_train_labels):
      A,H,Z=feeb_forward(train_ima,layers,activation_fun,W,B)
      current_dw,current_db=back_propagation(train_ima,train_labe,Z,H,A,layers,W,activation_fun,loss_Fun)

      for k in range(len(W)):
        dw[k]=dw[k]+current_dw[k]
        db[k]=db[k]+current_db[k]

      if(batch%batchSize==0):
        for k in range(len(W)):
          m_w[k]=beta1*m_w[k]+(1-beta1)*dw[k]
          m_b[k]=beta1*m_b[k]+(1-beta1)*db[k]
          v_w[k]=beta2*v_w[k]+(1-beta2)*pow(dw[k],2)
          v_b[k]=beta2*v_b[k]+(1-beta2)*pow(db[k],2)

          m_w_hat=m_w[k]/(1-pow(beta1,k+1))
          m_b_hat=m_b[k]/(1-pow(beta1,k+1))
          v_w_hat=v_w[k]/(1-pow(beta2,k+1))
          v_b_hat=v_b[k]/(1-pow(beta2,k+1))

          W[k]=W[k]-eta*m_w_hat/(np.sqrt(v_w_hat)+eps) - (weightDecay*W[k])
          B[k]=B[k]-eta*m_b_hat/(np.sqrt(v_b_hat)+eps)
        for xw in dw:
          xw[:]=0
        for xb in db:
          xb[:]=0
      batch+=1
    acc, loss = accuracy(W,B,x_train_images,y_train_labels,activation_fun,layers,loss_Fun)
    v_acc, v_loss = accuracy(W,B,x_validation_images,y_validation_labels,activation_fun,layers,loss_Fun)
    print("Accuracy")
    print(acc,loss,v_acc,v_loss)

    wandb.log({"Train_Accuracy" : acc,"Train_Loss" : loss,"Validation_acc" : v_acc,"validation_loss" : v_loss,'epoch':i})
  return W,B


def nadam(eta,layers,neurons,epochs,activation_fun,batchSize,weightInit,weightDecay,loss_Fun,beta1,beta2):
  W,B=weight(layers,neurons,weightInit)
  v_w=[np.zeros_like(w) for w in W]
  v_b=[np.zeros_like(w) for w in B]
  m_w=[np.zeros_like(w) for w in W]
  m_b=[np.zeros_like(w) for w in B]
  eps=1e-10
  for i in range(epochs):
    temp=1
    dw=[np.zeros_like(w) for w in W]
    db=[np.zeros_like(w) for w in B]
    for train_ima,train_labe in zip(x_train_images,y_train_labels):
      A,H,Z=feeb_forward(train_ima,layers,activation_fun,W,B)
      current_dw,current_db=back_propagation(train_ima,train_labe,Z,H,A,layers,W,activation_fun,loss_Fun)
      # current_dw.reverse()
      # current_db.reverse()

      for k in range(len(W)):
        dw[k]+=current_dw[k]
        db[k]+=current_db[k]

      if(temp%batchSize==0):
        for k in range(len(W)):
          m_w[k]=beta1*m_w[k]+(1-beta1)*dw[k]
          m_b[k]=beta1*m_b[k]+(1-beta1)*db[k]
          v_w[k]=beta2*v_w[k]+(1-beta2)*pow(dw[k],2)
          v_b[k]=beta2*v_b[k]+(1-beta2)*pow(db[k],2)

          m_w_hat=m_w[k]/(1-pow(beta1,i+1))
          m_b_hat=m_b[k]/(1-pow(beta1,i+1))
          v_w_hat=v_w[k]/(1-pow(beta2,i+1))
          v_b_hat=v_b[k]/(1-pow(beta2,i+1))

          W[k]=W[k]-(eta/(np.sqrt(v_w_hat+eps)))*(beta1*m_w_hat+(1-beta1)*dw[k]/(1-beta1**(k+1))) - (weightDecay*W[k])
          B[k]=B[k]-(eta/(np.sqrt(v_b_hat+eps)))*(beta1*m_b_hat+(1-beta1)*db[k]/(1-beta1**(k+1)))
        for xw in dw:
          xw[:]=0
        for xb in db:
          xb[:]=0
      temp+=1
    acc,loss=accuracy(W,B,x_train_images,y_train_labels,activation_fun,layers,loss_Fun)
    v_acc,v_loss=accuracy(W,B,x_validation_images,y_validation_labels,activation_fun,layers,loss_Fun)
    print("Accuracy")
    print(acc,loss,v_acc,v_loss)
    wandb.log({"Train_Accuracy" : acc,"Train_Loss" : loss,"Validation_acc" : v_acc,"validation_loss" : v_loss,'epoch':i})
  return W,B



def confusion_Matrix(epochs, layers, neurons, learningRate, optimizer, batchSize, activationFun, weightInit,weightDecay,loss_Fun):
  predicted=[]
  original=[]
  W,B=nadam(learningRate, layers, neurons, epochs, activationFun, batchSize, weightInit,weightDecay,loss_Fun)
  acc,loss=accuracy_confusion(W,B,x_test,y_test,activationFun,layers,loss_Fun,predicted,original)
  confusion=confusion_matrix(original,predicted)
  plt.figure(figsize=(10,10))
  sn.heatmap(confusion, annot=True, fmt='d',cmap='Oranges',linewidths=2,cbar=True,linecolor='black',
          xticklabels=['0','1','2','3','4','5','6','7','8','9'], yticklabels=['0','1','2','3','4','5','6','7','8','9'])
  plt.xlabel("PREDICTED")
  plt.ylabel("ORIGINAL")
  plt.title('confusion matrix')
  plt.savefig('confusion_matric_1.png')
  wandb.log({'confusion_matrix':wandb.Image('confusion_matric_1.png')})
  plt.show()



def main_function(epochs, layers, neurons, learningRate, optimizer, batchSize, activationFun, weightInit,weightDecay,loss_Fun,beta,momentum,beta1,beta2,epsilon):
    if optimizer == "sgd":
        w,b=stochastic_gradient_descent(epochs, learningRate, layers, neurons, activationFun, weightInit,weightDecay,loss_Fun)
    elif optimizer == "momentum":
        w,b=momentum_gradient_descent(epochs, learningRate, layers, neurons, activationFun, batchSize, weightInit,weightDecay,loss_Fun,momentum)
    elif optimizer == "nag":
        w,b=nesterov_accelerated_gradient_descent(epochs, layers, neurons, learningRate, activationFun, batchSize, weightInit,weightDecay,loss_Fun,beta)
    elif optimizer == "rmsprop":
        w,b=rms_prop(epochs, layers, neurons, learningRate, activationFun, batchSize, weightInit,weightDecay,loss_Fun,beta)
    elif optimizer == "adam":
        w,b=adam(epochs, learningRate, activationFun, layers, neurons, batchSize, weightInit,weightDecay,loss_Fun,beta1,beta2)
    elif optimizer == "nadam":
        w,b=nadam(learningRate, layers, neurons, epochs, activationFun, batchSize, weightInit,weightDecay,loss_Fun,beta1,beta2)



def parse_arguments():
  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('-wp', '--wandb_project', type=str, default='AssignmentDL_1',
                        help='Project name')
  
  parser.add_argument('-we', '--wandb_entity', type=str, default='Entity_DL',
                        help='Wandb Entity')
  
  parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',choices=["mnist", "fashion_mnist"],
                        help='Dataset choice: fashion_mnist , mnist')
  
  parser.add_argument('-e', '--epochs', type=int, default=10,help='Number of epochs for training network')

  parser.add_argument('-b', '--batch_size', type=int, default=64,help='Batch size for training neural network')

  parser.add_argument('-l', '--loss', type=str, default='cross_entropy',choices=["cross_entropy", "mean_squared_error"],help='Choice of mean_squared_error or cross_entropy')
  
  parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices = ["sgd", "momentum", "ngd", "rmsprop", "adam", "nadam"],help='Choice of optimizer')
   
  parser.add_argument('-lr', '--learning_rate', type=int, default=0.001, help='Learning rate')

  parser.add_argument( '-m', '--momentum', type=int, default=0.65, help='Momentum parameter')

  parser.add_argument('-beta', '--beta', type=int, default=0.58, help='Beta parameter')

  parser.add_argument('-beta1', '--beta1', type=int, default=0.89, help='Beta1 parameter')

  parser.add_argument('-beta2', '--beta2', type=int, default=0.85, help='Beta2 parameter')

  parser.add_argument( '-eps', '--epsilon', type=int, default=0.000001, help='Epsilon used by optimizers')

  parser.add_argument( '-w_i', '--weight_init',type=str, default="Xavier",choices=["random", "Xavier"], help='randomizer for weights')

  parser.add_argument('-w_d','--weight_decay',  type=int, default=0.0005, help='Weight decay parameter')

  parser.add_argument( '-nhl', '--num_layers',type=int, default=3, help='Number of hidden layers')
  
  parser.add_argument( '-sz','--hidden_size', type=int, default=128, help='Number of neurons in each layer')

  parser.add_argument( '-a','--activation', type=str, default="relu",choices=["sigmoid", "tanh", "ReLU"], help='activation functions')

  return parser.parse_args()

args = parse_arguments()

wandb.init(project=args.wandb_project)

if args.dataset == 'mnist':
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train_images, x_validation_images,y_train_labels, y_validation_labels  = train_test_split(x_train,y_train,test_size = 0.1)
else:
    (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    x_train_images, x_validation_images,y_train_labels, y_validation_labels  = train_test_split(x_train,y_train,test_size = 0.1)

wandb.run.name=f'activation {args.activation} weight_init{args.weight_init}opt{args.optimizer}'

main_function(args.epochs, args.num_layers, args.hidden_size, args.learning_rate, args.optimizer, args.batch_size, args.activation, args.weight_init, args.weight_decay, args.loss, args.beta, args.momentum, args.beta1, args.beta2, args.epsilon):
