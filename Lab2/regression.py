import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    mu = np.array([0,0])
    cov = np.array([[beta, 0], [0, beta]])

    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(a0, a1)
    x_set = np.dstack((X, Y))
    x_set = np.reshape(x_set, (len(X)*len(Y), 2))

    Z = util.density_Gaussian(mu, cov, x_set).reshape((100,100))

    plt.figure(1)
    plt.title("P(a)")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.contour(X, Y, Z)
    plt.plot(-0.1, -0.5, 'bx', label='true value')
    plt.legend()
    plt.savefig("prior.pdf")
    plt.show()

    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    ones = np.ones((1,np.size(x,0)))
    xt = np.vstack((ones, np.transpose(x)))
    xtx = np.matmul(xt,np.transpose(xt))
    var = np.array([[sigma2/beta, 0], [0, sigma2/beta]])
    mu = np.matmul(np.matmul(np.linalg.inv(xtx+var), xt),z)
    Cov = sigma2*np.linalg.inv(xtx+var)

    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(a0, a1)
    x_set = np.dstack((X, Y))
    x_set = np.reshape(x_set, (len(X)*len(Y), 2))

    Z = util.density_Gaussian(mu.transpose(), Cov, x_set).reshape((100,100))

    plt.figure(2)
    plt.title("p(a|x,z), N="+ str(x.shape[0]))
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.contour(X, Y, Z)
    plt.plot(-0.1, -0.5, 'bx', label='true value')
    plt.legend()
    plt.savefig("posterior"+str(x.shape[0])+".pdf")
    plt.show()
   
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    ones = np.ones((1,np.size(x,0)))
    xt = np.vstack((ones, np.transpose(x)))
    X_new = np.column_stack((np.ones((len(x), 1)), np.array(x)))

    newMu = np.matmul(xt.transpose(), mu)
    newCov = np.sqrt(np.diag(sigma2 + np.matmul(np.matmul(xt.transpose(), Cov), xt)))

    plt.figure(3)
    plt.title("p(z|x,z), N="+ str(x_train.shape[0]))
    plt.xlabel("x")
    plt.ylabel("z")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.errorbar(np.array(x), newMu, yerr=newCov, ecolor='k', color='b', label='Predictions')
    plt.scatter(x_train, z_train, color = 'g', label='Training Samples')
    plt.legend()
    plt.savefig("predict"+str(x_train.shape[0])+".pdf")
    plt.show()

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # prior distribution p(a)
    priorDistribution(beta)

    # number of training samples used to compute posterior
    ns  = [1, 5, 100]

    # for loop to try all required number of training samples
    for n in ns:
        # used samples
        x = x_train[0:n]
        z = z_train[0:n]
        
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
