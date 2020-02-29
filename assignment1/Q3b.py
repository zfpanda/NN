import numpy as np
import matplotlib.pyplot as plt

# inputs initializing
# columns = (bias, x1, x2, y)
AND_set = np.array([[1,0,0,0],
                    [1,0,1,0],
                    [1,1,0,0],
                    [1,1,1,1]])

OR_set = np.array([[1,0,0,0],
                   [1,0,1,1],
                   [1,1,0,1],
                   [1,1,1,1]])

COMPLEMENT_set = np.array([[1,0,1],
                           [1,1,0]])

NAND_set =np.array([[1,0,0,1],
                    [1,0,1,1],
                    [1,1,0,1],
                    [1,1,1,0]]) 

XOR_set =np.array([[1,0,0,0],
                    [1,0,1,1],
                    [1,1,0,1],
                    [1,1,1,0]]) 

# learning rate 
learn_rate = 1

def activation_fn(x):
    return 1 if x >=0 else 0

def train(x):
    dim = x.shape[1]
    col = x.shape[0]
    iterate = True
    iteration = 0

    #initial weights
    np.random.seed(0)
    weights = np.random.randn(1,dim -1)
    W = weights
    print("Initial weights is " + str(weights))
    # for i in range(iteration):
    while iterate:
        iterate = False
        for j in range(col):
            v = weights.dot(x[j,:-1])
            v_out = activation_fn(v)
            error = x[j,-1] - v_out
            if error != 0 :
                # print("wrong weights is " + str(weights))
                weights = weights + learn_rate*x[j,:-1]*error
                # W = np.append(W,weights,axis = 0)
                # print("Updated weights is " + str(weights))
                iterate = True 
        if iterate == True:
            W = np.append(W,weights,axis = 0)
            print("Updated weights is " + str(weights))           
        iteration += 1
    print(str(iteration) + " times of epochs")
    return W
    
def plotweights(W_AND,W_OR,W_NAND,W_COMPLEMENT):
    iter_range = range(W_AND.shape[0])
    iter_OR = range(W_OR.shape[0])
    iter_NAND = range(W_NAND.shape[0])
    iter_COMPLEMENT = range(W_COMPLEMENT.shape[0])

    plt.figure(num=3, figsize=(10, 3),)
    plt.subplots_adjust(wspace =0.2, hspace =0.3)
    plt.subplot(2,2,1)
    plt.plot(iter_range,W_AND[:,0:1],'g*',iter_range,W_AND[:,1:2],'r^',iter_range,W_AND[:,2:3],'bs')
    # plt.plot(iter_range,W[:,1:2],'r^','w1')
    # plt.plot(iter_range,W[:,2:3],'bs')
    plt.title("Learned weights for AND logic")
    plt.xlabel("epochs")
    plt.ylabel("Weight values")
    plt.legend(('b','W1','W2'),loc = 'upper right')

    plt.subplot(2,2,2)
    plt.plot(iter_OR,W_OR[:,0:1],'g*',iter_OR,W_OR[:,1:2],'r^',iter_OR,W_OR[:,2:3],'bs')
    # plt.scatter(iter_OR,W3,c='r')
    # plt.scatter(iter_OR,W4,c='b')
    plt.title("Learned weights for OR logic")
    plt.xlabel("epochs")
    plt.ylabel("Weight values")
    plt.legend(('b','W1','W2'),loc = 'upper right')

    plt.subplot(2,2,3)
    plt.plot(iter_COMPLEMENT,W_COMPLEMENT[:,0:1],'g*',iter_COMPLEMENT,W_COMPLEMENT[:,1:2],'r^')
    # plt.scatter(iter_COMPLEMENT,W7,c='r')
    plt.title("Learned weights for COMPLEMENT logic")
    plt.xlabel("epochs")
    plt.ylabel("Weight values")
    plt.legend(('b','W1'),loc = 'upper right')

    plt.subplot(2,2,4)
    plt.plot(iter_NAND,W_NAND[:,0:1],'g*',iter_NAND,W_NAND[:,1:2],'r^',iter_NAND,W_NAND[:,2:3],'bs')
    plt.title("Learned weights for NAND logic")
    plt.xlabel("epochs")
    plt.ylabel("Weight values")
    plt.legend(('b','W1','W2'),loc = 'upper right')
    plt.show()



def compare_weights(W_AND,W_OR,W_NAND):
    X = np.arange(-0.2,2,0.5)
    y_AND_learn = -(W_AND[W_AND.shape[0]-1,1]*X + W_AND[W_AND.shape[0]-1,0])/W_AND[W_AND.shape[0]-1,2]
    y_AND_off = -X + 1.5

    y_OR_learn = -(W_OR[W_OR.shape[0]-1,1]*X + W_OR[W_OR.shape[0]-1,0])/W_OR[W_OR.shape[0]-1,2]
    y_OR_off = -X + 0.5

    y_NAND_learn = -(W_NAND[W_NAND.shape[0]-1,1]*X + W_NAND[W_NAND.shape[0]-1,0])/W_NAND[W_NAND.shape[0]-1,2]
    y_NAND_off = -X + 1.5

    plt.figure(num=3, figsize=(10, 3),)
    plt.subplots_adjust(wspace =0.2, hspace =0.3)
    plt.subplot(2,2,1)
    plt.plot([0,0,1],[0,1,0],'bo',[1],[1],'ro')
    plt.plot(X,y_AND_learn,label = "Learned")
    plt.plot(X,y_AND_off,label = "off line calculation")
    plt.title("Learned weights & off line weights for AND logic")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot([0],[0],'bo',[0,1,1],[1,0,1],'ro')
    plt.plot(X,y_OR_learn,label = "Learned")
    plt.plot(X,y_OR_off,label = "off line calculation")
    plt.title("Learned weights & off line weights for OR logic")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot([0,0,1],[0,1,0],'bo',[1],[1],'ro')
    plt.plot(X,y_NAND_learn,label = "Learned")
    plt.plot(X,y_NAND_off,label = "off line calculation")
    plt.title("Learned weights & off line weights for NAND logic")
    plt.xlabel("X1")
    plt.ylabel("x2")
    plt.legend()

    plt.show()



W_AND = train(AND_set)
W_OR = train(OR_set)
W_NAND = train(NAND_set)
W_COMPLEMENT = train(COMPLEMENT_set)
Plotweights = plotweights(W_AND,W_OR,W_NAND,W_COMPLEMENT)
# Compare_weights = compare_weights(W_AND,W_OR,W_NAND)
# X_OR =train(XOR_set)