import numpy as np
import matplotlib.pyplot as plt
import util

def lda(mu_female, mu_male, cov, x):
    cov_inv = np.linalg.inv(cov)
    mu_female_t = np.transpose(mu_female)
    mu_male_t = np.transpose(mu_male)

    # find lda boundary based on formula in part b
    lda_female = np.matmul(np.matmul(mu_female_t,cov_inv),np.transpose(x)) - 0.5*np.matmul(np.matmul(mu_female_t, cov_inv), mu_female)
    lda_male = np.matmul(np.matmul(mu_male_t,cov_inv),np.transpose(x)) - 0.5*np.matmul(np.matmul(mu_male_t, cov_inv), mu_male)
    return lda_female, lda_male

def qda(mu_female, mu_male, cov_female, cov_male, x):
    cov_female_inv = np.linalg.inv(cov_female)
    cov_male_inv = np.linalg.inv(cov_male)
    cov_female_det = np.linalg.det(cov_female)
    cov_male_det = np.linalg.det(cov_male)
    distance_female = x-mu_female
    distance_male = x-mu_male

    # find qda boundary based on formula in part b
    qda_female = -0.5*(np.log(cov_female_det)+np.matmul(np.matmul(np.transpose(distance_female),cov_female_inv),distance_female))
    qda_male = -0.5*(np.log(cov_male_det)+np.matmul(np.matmul(np.transpose(distance_male),cov_male_inv),distance_male))
    return qda_female, qda_male

def plot(females, males, mu_female, mu_male, cov, cov_female, cov_male, x, y, title):
    heights_m = [pair[0] for pair in males]
    weights_m = [pair[1] for pair in males]
    heights_f = [pair[0] for pair in females]
    weights_f = [pair[1] for pair in females]
    
    axes = plt.gca()
    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
    plt.scatter(heights_m, weights_m, color = 'blue')
    plt.scatter(heights_f, weights_f, color = 'red') 

    h, w = np.meshgrid(np.arange(50, 80, 1), np.arange(80, 280, 1))
    h_flat = h.flatten().reshape(-1, 1)
    w_flat = w.flatten().reshape(-1, 1)
    h_set = np.concatenate((h_flat, w_flat), axis = 1)
    z_m = util.density_Gaussian(np.transpose(mu_male), cov, h_set).reshape((200, 30))
    z_f = util.density_Gaussian(np.transpose(mu_female), cov, h_set).reshape((200, 30))

    plt.contour(h, w, z_m, colors = 'blue')
    plt.contour(h, w, z_f, colors = 'red')

    if title == 'LDA':
        lda_female, lda_male = lda(mu_female, mu_male, cov, h_set)
        bound = (lda_male - lda_female).reshape((200, 30))
    
    else:
        qda_female = np.zeros((h_set.shape[0], 1))
        qda_male = np.zeros((h_set.shape[0], 1))

        for i in range(h_set.shape[0]):
            qda_female[i], qda_male[i] = qda(mu_female, mu_male, cov_female, cov_male, h_set[i])
        bound = (qda_male - qda_female).reshape((200, 30))
    
    plt.contour(h, w, bound, 0) 
    plt.title(title)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.savefig(title+'.pdf')
    plt.show()

def discrimAnalysis(x, y):
    females = []
    males = []

    # separate male and female points
    for i in range(len(x)):
        if y[i] == 1:
            males.append(x[i])
        else:
            females.append(x[i])

    # find mu and sigma based on formula in part a
    mu_female = (1/len(females))*np.sum(females, axis=0)
    mu_male = (1/len(males))*np.sum(males, axis=0)

    cov_female = 0
    for f in females:
        cov_female += np.outer(f - mu_female, np.transpose(f - mu_female))

    cov_female = (1/len(females))*cov_female

    cov_male = 0
    for m in males:
        cov_male += np.outer(m - mu_male, np.transpose(m - mu_male))

    cov_male = (1/len(males))*cov_male

    cov = ((cov_female*len(females)) + (cov_male*len(males)))/len(y)

    # plot lda and qda
    plot(females, males, mu_female, mu_male, cov, cov_female, cov_male, x, y, 'LDA')
    plot(females, males, mu_female, mu_male, cov, cov_female, cov_male, x, y, 'QDA')
    
    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    mis_lda = 0
    mis_qda = 0

    for i in range(len(x)):
        lda_female, lda_male = lda(mu_female, mu_male, cov, x[i])
        
        prediction = 0
        if lda_female > lda_male:
            prediction = 2
        else:
            prediction = 1
        if y[i] != prediction:
            mis_lda += 1

        qda_female, qda_male = qda(mu_female, mu_male, cov_female, cov_male, x[i])

        prediction = 0
        if qda_female > qda_male:
            prediction = 2
        else:
            prediction = 1
        if y[i] != prediction:
            mis_qda += 1

    print(mis_lda/len(y), mis_qda/len(y))
    
    return (mis_lda/len(y), mis_qda/len(y))


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
  