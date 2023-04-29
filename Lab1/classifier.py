import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    spams = file_lists_by_category[0]
    hams = file_lists_by_category[1]

    spams_dic = util.get_word_freq(spams)
    hams_dic = util.get_word_freq(hams)
    all_dic = util.get_word_freq(spams+hams)

    alpha = 1
    spams_count = sum(spams_dic.values())
    hams_count = sum(hams_dic.values())
    distinct_count = len(all_dic)

    # find laplace smoothing based on part 1.a
    def calculate_condi_prob(word_count, spam):
        if spam:
            return (word_count+alpha)/(spams_count+distinct_count)
        else:
            return (word_count+alpha)/(hams_count+distinct_count)

    p_d = {}
    q_d = {}

    for word in all_dic:
        if(word in spams_dic):
            p_d[word] = calculate_condi_prob(spams_dic[word], True)
        else:
        	p_d[word] = calculate_condi_prob(0, True)
    
    for word in all_dic:
        if(word in hams_dic):
        	q_d[word] = calculate_condi_prob(hams_dic[word], False)
        else:
            q_d[word] = calculate_condi_prob(0, False)
    
    probabilities_by_category = (p_d, q_d)   
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    filename_list = (filename, 1)
    words_dic = util.get_word_freq(filename_list[0:1])

    spams = probabilities_by_category[0]
    hams = probabilities_by_category[1]

    log_likelihood_spam = 0
    log_likelihood_ham = 0

    # find log likelihood of part 2.a
    for word in words_dic:
        freq = words_dic[word]
        if(word in spams):
            log_likelihood_spam += np.log(spams[word])*freq
        if(word in hams):
            log_likelihood_ham += np.log(hams[word])*freq

    log_likelihood_spam += np.log(prior_by_category[0])
    log_likelihood_ham += np.log(prior_by_category[1])

    log_likelies = [log_likelihood_spam, log_likelihood_ham]

    if log_likelihood_spam > log_likelihood_ham:
        classification = 'spam'
    else: 
    	classification = 'ham'

    classify_result = (classification, log_likelies)
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    t1 = []
    t2 = []
    ratios = [1E-30, 1E-15, 1E-10, 1E-5, 1E0, 20, 50, 90, 1E2, 1E3]

    for ratio in ratios:
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            classification,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category)
            
            log_likelihood_spam = log_posterior[0]
            log_likelihood_ham = log_posterior[1]

            base = os.path.basename(filename)
            true_index = ('ham' in base) 

            if (log_likelihood_spam + np.log(ratio) > log_likelihood_ham):
                classification = 'spam'
            else:
            	classification = 'ham'
            guessed_index = (classification == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        
        correct = np.diag(performance_measures)
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0],totals[0],correct[1],totals[1]))

        t1_error = totals[0] - correct[0]
        t2_error = totals[1] - correct[1]

        t1.append(t1_error)
        t2.append(t2_error)
    
    plt.plot(t1, t2)
    plt.xlabel('Type 1 error')
    plt.ylabel('Type 2 error')
    plt.title('Type 1 vs Type 2 error trade off')
    plt.savefig("nbc.pdf")
    plt.show()
