import sys
import random
from rec2vec import graph
from gensim.models import Word2Vec
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

def process(args):
    # Create a graph from the training set
    nodedict = graph.records_to_graph()

    # Build the model using DeepWalk and Word2Vec
    G = graph.load_adjacencylist("out.adj", undirected=True)
    # YOUR CODE HERE

    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks, path_length=args.walk_length)#, rand=random.Random(args.seed))

    model = Word2Vec(walks, size=args.representation_size, window=args.window_size, workers=args.workers, min_count=0, seed=args.seed)
#    mode.build_vocab()
                         
    # Perform some evaluation of the model on the test dataset
    with open("./data/test_user_ratings.dat") as fin:
        fin.next()
        groundtruth = [line.strip().split("\t")[:3] for line in fin]    # (user, movie, rating)
    tr = [int(round(float(g[2]))) for g in groundtruth]
    pr = [predict_rating(model, nodedict, "u"+g[0], "m"+g[1]) for g in groundtruth]

    print "MSE = %f" % mean_squared_error(tr, pr)
    print "accuracy = %f" % accuracy_score(tr, pr)
    cm = confusion_matrix(tr, pr, labels=range(1,6))
    print cm


def predict_rating(model, nodedict, user, movie):
    """
    Predicts the rating between a user and a movie by finding the movie-rating node with the highest
    similarity to the given user node.
    Loops through the five possible movie-rating nodes and finds the node with the highest similarity to the user.
    
    Returns an integer rating 1-5.
    """
    # YOUR CODE HERE

    user_id = nodedict[user].id
    
    ratingsRange = ['1','2','3','4','5']
    maxSim = 0.0
    maxRating = 0

    for r in ratingsRange:
		temp = movie + "_" + r
		movie_id = nodedict[temp].id
		s = model.similarity(user_id, movie_id)
		if (s > maxSim):
			maxSim = s
			maxRating = int(r)

    return maxRating

def main():
    parser = ArgumentParser("rec2vec", 
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('--number-walks', default=10, type=int,
        help='Number of random walks to start at each node')
    parser.add_argument('--walk-length', default=40, type=int,
        help='Length of the random walk started at each node')
    parser.add_argument('--seed', default=0, type=int,
        help='Seed for random walk generator.')   
    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
        help='Size to start dumping walks to disk, instead of keeping them in memory.')
    parser.add_argument('--window-size', default=5, type=int,
        help='Window size of skipgram model.')
    parser.add_argument('--workers', default=1, type=int,
        help='Number of parallel processes.')        
    parser.add_argument('--representation-size', default=64, type=int,
        help='Number of latent dimensions to learn for each node.')
    
    parser.set_defaults(csv_to_graph=True, loo=True)
    args = parser.parse_args()
    
    process(args)
    

if __name__=="__main__":
    sys.exit(main())
