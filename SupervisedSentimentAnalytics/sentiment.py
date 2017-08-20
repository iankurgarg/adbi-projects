import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets

if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)
    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    positive_list = {}
    negative_list = {}

    for l in train_pos:
        l2 = set(l)
        for word in l2:
            if not word in positive_list:
                positive_list[word] = 1
            else:
                positive_list[word] += 1

    for l in train_neg:
        l2 = set(l)
        for word in l2:
            if not word in negative_list:
                negative_list[word] = 1
            else:
                negative_list[word] += 1


    len_pos = int(len(train_pos)*0.01);
    len_neg = int(len(train_neg)*0.01);
    p_list = {k:v for (k,v) in positive_list.items() if (v >= len_pos and (v >= 2*negative_list[k] or negative_list[k] >= 2*v) and k not in stopwords)}
    n_list = {k:v for (k,v) in negative_list.items() if v >= len_neg and (v >= 2*positive_list[k] or positive_list[k] >= 2*v) and k not in stopwords}

    feature_list = list(set(p_list.keys() + n_list.keys()))
    print len(feature_list)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = []
    for l in train_pos:
        feature_dict = dict.fromkeys(feature_list, 0)
        for word in l:
            if word in feature_dict:
                feature_dict[word] = 1;
        train_pos_vec.append(feature_dict.values())

    train_neg_vec = []
    for l in train_neg:
        feature_dict = dict.fromkeys(feature_list, 0)
        for word in l:
            if word in feature_dict:
                feature_dict[word] = 1;
        train_neg_vec.append(feature_dict.values())

    test_pos_vec = []
    for l in test_pos:
        feature_dict = dict.fromkeys(feature_list, 0)
        for word in l:
            if word in feature_dict:
                feature_dict[word] = 1;
        test_pos_vec.append(feature_dict.values())

    test_neg_vec = []
    for l in test_neg:
        feature_dict = dict.fromkeys(feature_list, 0)
        for word in l:
            if word in feature_dict:
                feature_dict[word] = 1;
        test_neg_vec.append(feature_dict.values())

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = [None]*len(train_pos)
    labeled_train_neg = [None]*len(train_neg)
    labeled_test_pos = [None]*len(test_pos)
    labeled_test_neg = [None]*len(test_neg)

    i=0
    for s in train_pos:
        labeled_train_pos[i] = LabeledSentence(words=s, tags=["TRAIN_POS_"+str(i)])
        i = i+1

    i=0
    for s in train_neg:
        labeled_train_neg[i] = LabeledSentence(words=s, tags=["TRAIN_NEG_"+str(i)])
        i = i+1

    i=0
    for s in test_pos:
        labeled_test_pos[i] = LabeledSentence(words=s, tags=["TEST_POS_"+str(i)])
        i = i+1

    i=0
    for s in test_neg:
        labeled_test_neg[i] = LabeledSentence(words=s, tags=["TEST_NEG_"+str(i)])
        i = i+1



    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for tag in model.docvecs.doctags.keys():
        if "TRAIN_POS_" in tag:
            train_pos_vec.append(model.docvecs[tag])
        elif "TRAIN_NEG_" in tag:
            train_neg_vec.append(model.docvecs[tag])
        elif "TEST_POS_" in tag:
            test_pos_vec.append(model.docvecs[tag])
        elif "TEST_NEG_" in tag:
            test_neg_vec.append(model.docvecs[tag])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    nb_model = BernoulliNB()
    nb_model.fit(train_pos_vec+train_neg_vec, Y)


    lr_model = LogisticRegression()
    lr_model.fit(train_pos_vec+train_neg_vec, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    nb_model = GaussianNB()
    nb_model.fit(train_pos_vec+train_neg_vec, Y)


    lr_model = LogisticRegression()
    lr_model.fit(train_pos_vec+train_neg_vec, Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    predicted_pos = model.predict(test_pos_vec)
    tp = sum(predicted_pos == "pos")
    fn = sum(predicted_pos == "neg")
    predicted_neg = model.predict(test_neg_vec)
    fp = sum(predicted_neg == "pos")
    tn = sum(predicted_neg == "neg")

    accuracy = float(tp+tn)/(tp+tn+fp+fn)
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
