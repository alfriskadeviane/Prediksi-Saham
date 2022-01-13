import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

def loadData():
    datenow=datetime.date.today()
    name_emiten ="BBCA.JK"
    saham=yf.download(name_emiten, start="2017-1-1", end=datenow)
    #saham= pd.read_csv('E:\SEMANGAT SEMINAR\PYTHON\FlaskApp\static\datasaham.csv')
    data=pd.DataFrame(saham)
    return data
def normData(data):
    datac=pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Volume', 'Close'])
    datac= datac.replace(0, np.nan)
    dataclear=datac.dropna()
    dataNorm=(dataclear-dataclear.min())/(dataclear.max()-dataclear.min())
    return dataclear,dataNorm
def splitDataHarian(dataNorm):        
    x= dataNorm.drop(columns=['Close'])
    xa= np.array(x)
    y_label=dataNorm['Close']
    y=pd.DataFrame(y_label)
    yt=np.array(y)
    Xtrain=xa
    ytrain=yt
    #Xtrain, Xtest, ytrain, ytest = train_test_split(xa, yt, test_size=0.2, random_state=2)
    return Xtrain, ytrain
def splitData(dataNorm):        
    x= dataNorm.drop(columns=['Close'])
    xa= np.array(x)
    y_label=dataNorm['Close']
    y=pd.DataFrame(y_label)
    yt=np.array(y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(xa, yt, test_size=0.2, random_state=2)
    return Xtrain, Xtest, ytrain, ytest

class NeuralNet():
    '''
    A two layer neural network
    '''
        
    def __init__(self, layers, learning_rate, epoch):
        self.params = {}
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1']  =np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
        '''
        print(self.params['W1'])
        print(self.params['b1'])
        print(self.params['W2'])
        print(self.params['b2'])
        '''
        return self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']

    def sigmoid(self,Z,deriv=False):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        if(deriv==True):
            return Z*(1-Z)
        return 1/(1+np.exp(-Z))

    def forward_propagation(self,X):
        '''
        Performs the forward propagation
        '''
        self.z= np.dot(X,self.params["W1"])+self.params['b1']
        self.z2= self.sigmoid(self.z)
        self.z3= np.dot(self.z2,self.params["W2"])+self.params['b2']
        output= self.sigmoid(self.z3)
        #print(output)
        return output
    
    def backward(self,X,y,output):
        self.output_error=y-output
        self.output_delta=self.output_error* self.sigmoid(output, deriv=True)
        
        self.z2_error=self.output_delta.dot(self.params["W2"].T)
        self.z2_delta=self.z2_error* self.sigmoid(self.z2, deriv=True)
        #print(self.z2_delta)
                  
        self.params["W1"] += X.T.dot(self.z2_delta)*self.learning_rate
        self.params["W2"] += self.z2.T.dot(self.output_delta)*self.learning_rate
        self.params["b1"] += self.z2_delta.sum()*self.learning_rate
        self.params["b2"] += self.output_delta.sum()*self.learning_rate
        '''
        print(self.params['W1'])
        print(self.params['b1'])
        print(self.params['W2'])
        print(self.params['b2'])
        '''
        
    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights() #initialize weights and bias
        
        for i in range(self.epoch):
            output=self.forward_propagation(X)
            self.backward(X,y,output)
        return self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        self.z= np.dot(X,self.params["W1"])+self.params['b1']
        self.z2= self.sigmoid(self.z)
        self.z3= np.dot(self.z2,self.params["W2"])+self.params['b2']
        output= self.sigmoid(self.z3)
        return output
    
    def mse(self, y, output):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        mse = sum((y - output)**2 / len(y))
        return mse
    def popawal(self,popsize):
        n=(self.layers[0]*self.layers[1])+(self.layers[1]*self.layers[2])+self.layers[1]+self.layers[2]
        #print(n)
        pop=[]
        population=[]
        for i in range(popsize):
            np.random.seed(i+1)
            pops=np.random.randn(n,)
            pop.append(pops)
            population.append(pops)
        #print("pop")
        #p=pd.DataFrame(pop)
        #print(p)
        #print(pop[1][2])
        return n,pop,population
    def  algen(self,Xtrain,ytrain,popsize,cr,mr,gensize):
        n,pop,population=self.popawal(popsize)
        for g in range(gensize):
            #print(g)
            #crossover
            offspringC=round(popsize*cr)
            #print(offspringC)
            np.random.seed(0)
            alpha=np.random.randn(n,)
            #print(alpha)
            #print('crossover')
            c=[]
            for i in range(offspringC):
                c=pop[i]+alpha*(pop[i+1]-pop[i])
                #print(c)
                population.append(c)

            #mutation
            offspringM=round(popsize*mr)
            #print(offspringM)
            #print('mutation')
            
            for i in range(offspringM):
                ran=np.random.randint(0,n)
                #print("ran", ran)
                mi=pop[-(i+1)][ran]+(np.random.randn()*(pop[-(i+1)].max()-pop[-(i+1)].min()))
                mo=pop[-(i+1)].copy()
                #print(pop[-(i+1)])
                #print(mi)
                mo[ran]=mi
                #print(mo)
                population.append(mo)
            #print(population)

            #selection
            j=self.layers[1]
            k=self.layers[1]*self.layers[0]+self.layers[1]
            l=k+j+1
            ftns=[]
            for i in range(len(population)):
                self.params['b1']=population[i][0:j]
                self.params['W1']=population [i][j:k].reshape(self.layers[0],self.layers[1])
                self.params['b2']=population[i][k].reshape(1)
                self.params['W2']=population[i][k+1:l].reshape(self.layers[1],1)
                predik=self.predict(Xtrain)
                fitness=1/self.mse(ytrain,predik)
                ftns.append(fitness)
            #print(ftns)
            fitn=np.array(ftns)
            #print(fitn)
            ind=np.argsort(fitn, axis=0)[::-1][:popsize]
            p= int(ind[0])
            #print(p)
            #print(population[p])
            if g==(gensize-1):
                self.params['b1']=population[int(ind[0])][0:j]
                self.params['W1']=population [int(ind[0])][j:k].reshape(self.layers[0],self.layers[1])
                self.params['b2']=population[int(ind[0])][k].reshape(1)
                self.params['W2']=population[int(ind[0])][k+1:l].reshape(self.layers[1],1)
                '''
                print(self.params['W1'])
                print(self.params['b1'])
                print(self.params['W2'])
                print(self.params['b2'])
                '''
            else: 
                pop=[]
                #print("pop",pop)
                for i in range(popsize):
                    pops=population[int(ind[i])]
                    #print("pops",pops)
                    pop.append(pops)
                #print(pop)
                population=pop.copy()
                #print("popu", population)
        #return self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']

    def fitalgen(self, X, y,popsize,cr,mr,gensize):
        self.X = X
        self.y = y
        popsize=popsize
        cr=cr
        mr=mr
        gensize=gensize
        self.algen(X,y,popsize,cr,mr,gensize)
        w1=self.params['W1'].copy()
        b1=self.params['b1'].copy()
        w2=self.params['W2'].copy()
        b2=self.params['b2'].copy()
        '''
        print(self.params['W1'])
        print(self.params['b1'])
        print(self.params['W2'])
        print(self.params['b2'])
        '''
        for i in range(self.epoch):
            #print(i)
            output = self.forward_propagation(X)
            self.backward(X,y,output)
        return w1,b1,w2,b2,output,self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2'] 