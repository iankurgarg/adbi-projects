#===============================================================
# Name: Data Fusion Project
# Vers: 1.0
# Description: In this project, we will have you
# implement the algorithm discussed in "paper citation"
# using Tensorflow to solve the issue of image classification. 
# In addition, we will also ask you to clean and filter the 
# training data to reduce training time. At the end of the 
# project, the following learning goals should be achieved:
#   1. How to take data of different modalities and combine
#       them to answer a question about the entire dataset
#   2. How to sample and reduce very large datasets to test
#       and validate your algorithm
#===============================================================

import os
import string
import collections as cl
import numpy as np
import numpy.random as npr
import tensorflow as tf
import networkx as nx
#import rawdatacleaner as rdc

def load_textcorpus():
    """
        Reads the bbc datafile and generates a dictionary, where the keys
        are the text categories, and the values are a list of the documents,
        with each document being just being the multiset of words in the document
    """
    textfiles = []
    datafile = "data/bbc"
    for l in filter(lambda x: "." not in x, os.listdir(datafile)):
        for f in os.listdir(datafile+os.sep+l):
            with open(datafile+os.sep+l+os.sep+f, "r") as of:
                document = []
                for line in of:
                    document.extend(line.lower().strip().strip(string.punctuation).split())
                textfiles.append(document)
    return textfiles

def load_imagecorpus():
    """
        Reads the saved numpy matrices that store 5 127-by-127 patches for each image in
        the NUS-WIDE dateset that contain a certain keyword (discussed in the project description)

        Also reads the image_tag file and generates the tags related to the image
    """
    matrix_dir = "data/Mats"
    image_tag_file = "data/imagelist.txt"

    images = {}
    tags = {}
    image_tag_file = open(image_tag_filename, "r")
    for irow in image_tag_file:
        row = irow.split()
        iname = row[0]
        if len(row[1:] ) > 0:
            tags[iname] = row[1:]
            images[iname] = np.load(os.path.join(matrix_dir, iname.split("\\")[0]+os.sep+iname.split("\\")[1].split(".")[0])+".npy")

    return (tags,images)

def build_tag_vocab(tags):
    """
        This file generates the one-hot-encoding (binary) representation
        of each tag in the image's tag corpus, and return it as a dictionary
        where the keys are the tags, and the values are their binary representation
        (as a numpy array of integers)
    """
    ohe_tags = {}
    vocab_size = len(tags)
    for (i,w) in enumerate(tags):
        ohe_tags[w] = i
    return ohe_tags

def build_text_vocab(textfiles):
    """
        This file generates the one-hot-encoding (binary) representation
        of each tag in the image's tag corpus, and return it as a dictionary
        where the keys are the tags, and the values are their binary representation
        (as a numpy array of integers)
    """
    ohe_textfiles = {}
    vocab_size = len(textfiles)
    for (i,w) in enumerate(textfiles):
        ohe_textfiles[w] = i
    return ohe_textfiles

def build_image_inputouput_set(tags, images, ohe_tags):
    """
        For each image in the training set, generate
        a tuple whose first position is a single 127-by-127
        patch, the the second position is a list of the indices
        representing the tags associated with the patch
    """
    image_inputoutput_set = []

    # YOUR CODE HERE

    return image_inputoutput_set

def build_textcorpus_inputoutput_set(textcorpus, ohe_textfiles):
    """
        Build the input/output label pairs by applying a sliding
        window over the text corpus, and extracting the center
        and context words, as dicussed in the lectures.

        Set the window size to 5
        Let the input be a list containing the index of the center element
        Let the output be a list containing the indices of the context elements
    """
    textcorpus_inputoutput_set = []

    # YOUR CODE HERE

    return textcorpus_inputoutput_set

def build_skipgram(text_input, text_output, vocabulary_size):
    """
        Implement the skipgram algorithm. Please read the tensorflow tutorial
        on skipgram, which will walk you through how to do this in tensorflow
        https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#vector-representations-of-words

        Set the number of hidden dimensions to 300

        Return the optimization operation, the loss and the embedding weight for text_input
    """
    embedding_size = 300
    # YOUR CODE HERE

    return (loss, optimizer, embedding)

def build_cnn(image_input, image_output, tag_size):
    """
        Implement a cnn to embed the images in relation to their tags. For a walkthough
        on how to build cnns, please refer to this tensorflow tutorial:
        https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

        For the purposes of this project, we want a CNN with one convolutional
        layers and two softmax layers, with the second serving as the output layer.

        For Convolution 1, use a 5-by-5 filter with a stride of 2. Adjust the number of convolutional filters to 32

        Set the embedding weight vector equal to the output of the first convolutional layer (without drop-out)
    """

    # YOUR CODE HERE

    #Include this as part of code.
    loss = -tf.reduce_sum(image_output*tf.log(y_conv2))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return (loss, optimizer, cnn_embedding)

def build_adjacency_graph(tags, textcorpus):
    """
        Build the adjacency graph used by the heterogeneous data fusion algorithm to determine
        if an image and a word should be embedding close to one another. An image and an word should
        have an edge between them if the word is contained in the image's tag list

        A good graph library:
        http://networkx.readthedocs.org/en/networkx-1.11/

        Hint1: Instead of using the images as nodes, you can give your nodes attributes to denote
        "image nodes" and "text nodes"
    """


    # YOUR CODE HERE

    return adjacency_graph

def build_noise_list(agraph):
    """
        Using the adjaceny graph, generate a list of 1000 pairs of non-adjacent images and words.
        This will be added to the learning set to make sure our algorithm can distinguish between
        words and images that should not be close.
    """

    # YOUR CODE HERE

    return noise_list

def data_fusion_loss_function(pdi, pdt, edge_exists):
    """
        This function should implement the loss function described by formula 18 in the paper
    """

    # Compute the dot-product of pdi and pdt and multiply by the negated edge_exists (-1 if 1, 1 if negative 1)
    # YOUR CODE HERE

    loss_function = tf.constant(1)

    return (dotprod, loss_function);


def build_data_fusion_layer(t_embedding, i_embedding, edge_exists):
    """
        Using the lecture notes and your previous word2vec implementation, implement the data fusion
        layer described by formula 16 from the paper that will take in the embeddings between an 
        image embedding and a text embedding and compute the embedding and loss for the embedding.

        Note: Because we are only optimizing the embedding with repect to images and text, and not text-to-text
        or image-to-image, the objective function simplfies to just the loss function of the image to text portion

        Set the embedding dimension for this new projection layer to be 150
    """

    # Incorporating dropout on the hidden layer:
    dropped_tembedding = tf.nn.dropout(t_embedding, keep_prob=.5)
    dropped_iembedding = tf.nn.dropout(i_embedding, keep_prob=.5)

    # Create the variables associated with the transformation weights
    #Using fromula 16 from the paper
    # YOUR CODE HERE
    d = 150

    # Transform the embeddings with Uz and Ut

    # 
    # Transform the embeddings with Uz and Ut
    # pdi = tf.constant(1)
    # pdt = tf.constant(1)



    (dropprod, loss_function) = data_fusion_loss_function(pdi, pdt, edge_exists)
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss_function)

    return (optimizer, loss_function, Ut, Ui, dropprod, pdi, pdt)


def query(sess, pdi, pdt, image, frequent_words):
    """
        Given an image, retreive the most frequent words that might be strongly
        associated with the image using the vectors generated by the heterogeneous
        data fusion method
    """
    # YOUR CODE HERE

    pass

def main():
    # Number of iterations for the heterogeneous data fusion algorithm to train
    num_iters = 5

    text = load_textcorpus()
    (image_tags, images) = load_imagecorpus()

    tags = set()
    count = Counter()

    #Add all words from text corpus
    for l in text:
        count.update(l)

    words = count.keys()

    vocabulary_size = len(words)
    print vocabulary_size, '   vocab size'
    freq_words = count.most_common(1000)

    for t in image_tags.itervalues():
        tags.update(t)

    ohe_text = build_text_vocab(words)
    ohe_tags = build_tag_vocab(tags)

    text_learning = build_textcorpus_inputoutput_set(text, ohe_text)
    image_learning = build_image_inputouput_set(image_tags, images, ohe_tags)
    print len(image_learning), '    image size'

    relationgraph = build_adjacency_graph(image_tags, text)

    learning_embedding = tf.Graph()
    with learning_embedding.as_default():
        image_input = tf.placeholder("float",shape=(127,127))
        blank_image = np.zeros(127)
        image_truth = tf.placeholder("float",shape=[None,1])

        text_input = tf.placeholder("int32",shape=[None])
        text_truth = tf.placeholder("float",shape=[None, 1])

        edge_exists = tf.placeholder("float",shape=[1])

        with tf.name_scope("skipgram"):
            (toptimizer, tloss, tembedding) = build_skipgram(text_input,text_truth)

        with tf.name_scope("cnn"):
            (toptimizer, iloss, iembedding) = build_cnn(image_input, image_truth)

        with tf.name_scope("data_fusion_layer"):
            (doptimizer, dloss, Ut, Uz, dropprod, pdi, pdt) = build_data_fusion_layer(tembedding, iembedding, edge_exists)

    with tf.Session(graph=comp) as sess:
        tf.initialize_all_variables().run()

        for batch in image_learning:
            _,loss = sess.run([toptimizer, tloss],{image_input:batch[0], \
                image_truth:batch[1], text_input:[[]], text_truth:[], edge_exists:[-1]})

        # Now, train the skipgram neural network. Look to the previous for loop
        # for suggestions on how to perform this

        # YOUR CODE HERE

        for i in range(0,num_iters):
            noise_list = build_noise_list(relationgraph)

            # Normally you would permute this. However, due to the large size of each list,
            # this permuation step would dominate training.

            for batch in relationgraph.edges():
                _,loss = sess.run([doptimizer, dloss], {image_input:batch[0], \
                    image_truth:[], text_input:batch[1], text_truth:[], edge_exists:[1]})

            # Now, train the data fusion layer by passing each image_word pair that should not be adjacent in the noise_list
            # by passing the tuples one at a time, with edge_exists equal to -1

            # YOUR CODE HERE

        edges_predicted = 0
        for batch in relationgraph.edges():
            # For each batch, get the value of the dropprod computed by the computational graph
            # If this value is above zero, increment edges_predicted by one

            # YOUR CODE HERE

        print "--------------------------------"
        print "Reconstruction Error: %d" % (relationgraph.num_of_edges/edges_predicts)
        print "--------------------------------"
        print "Image 100 Tags"
        print image_tags[image[100]]
        print query(sess, pdi, pdt, images[100], image_tags[images[100]], frequent_words)

print zip([1,0],["a","b","c"])