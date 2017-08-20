import csv
import pandas as pd
import numpy as np
import random
import sys

# Setting a seed to ensure replicability of results
random.seed(0)

# Function that implements Greedy algorithm
def Greedy(budgets, bids, queries):
	revenue = 0.0
	for q in queries:
		bidder = find_bidder_greedy(bids[q], budgets)
		if bidder != -1:
			revenue += bids[q][bidder]
			budgets[bidder] -= bids[q][bidder]

	return revenue;

# Function that implements Balance algorithm
def Balance(budget, bids, queries):
	revenue = 0.0;
	for q in queries:
		bidder = find_bidder_balance(bids[q], budget)
		if bidder != -1:
			revenue += bids[q][bidder]
			budget[bidder] -= bids[q][bidder]

	return revenue;

# Function that implements MSVV algorithm
def MSVV(rembudget, budgets, bids, queries):
	revenue = 0.0;
	for q in queries:
		bidder = find_bidder_msvv(bids[q], rembudget, budgets)
		if bidder != -1:
			revenue += bids[q][bidder]
			rembudget[bidder] -= bids[q][bidder]

	return revenue;

# Helper function that check if all the bidders of a query have exhausted their budget or not.
# Returns -1 if they have all exhausted their budgets, 0 otherwise
def check_budget(b, budgets):
	keys = b.keys()
	for k in keys:
		if budgets[k] >= b[k]:
			return 0
	return -1

def psi (xu):
	return 1 - np.exp(xu-1);

# Function finds the best bidder for a query.
# Input: b: bids of all available bidders for the query, budgets: budgets of all bidders
# Output: id of the bidder that should be matched if possible, -1 if no bidder possible
# For each bidder checks if the bid is greater than max, if yes, update max.
# If equal to max (meaning more than one bidder can be selected), compares the bidder id and chooses the min
def find_bidder_greedy(b, budgets):
	keys = b.keys()
	maxBidder = keys[0]
	maxBid = -1;
	c = check_budget(b, budgets)
	if c == -1:
		return -1;
	for k in keys:
		if budgets[k] >= b[k]:
			if maxBid < b[k]:
				maxBidder = k
				maxBid = b[k]
			elif maxBid == b[k]:
				if maxBidder > k:
					maxBidder = k
					maxBid = b[k]
	return maxBidder

# Function finds the best bidder for a query.
# Input: b: bids of all available bidders for the query, budgets: budgets of all bidders
# Output: id of the bidder that should be matched if possible, -1 if no bidder possible
# For each bidder checks if the reamining budget is greater than max, if yes, update max.
# If equal to max (meaning more than one bidder can be selected), compares the bidder id and chooses the min
def find_bidder_balance(b, budgets):
	keys = b.keys()
	maxBidder = keys[0]
	c = check_budget(b, budgets)
	if c == -1:
		return -1;
	for k in keys:
		if budgets[k] >= b[k]:
			if budgets[maxBidder] < budgets[k]:
				maxBidder = k
			elif budgets[maxBidder] == budgets[k]:
				if maxBidder > k:
					maxBidder = k

	return maxBidder

# Scales the bid based on the remaining budget as per the MSVV algorithm
def scaledBid (bid, rembud, bud):
	xu = (bud-rembud)/bud
	return bid*psi(xu)

# Function finds the best bidder for a query.
# Input: b: bids of all available bidders for the query, budgets: budgets of all bidders
# Output: id of the bidder that should be matched if possible, -1 if no bidder possible
# For each bidder checks if the scaled bid is greater than max, if yes, update max.
# If equal to max (meaning more than one bidder can be selected), compares the bidder id and chooses the min
def find_bidder_msvv(b, rembudgets, budgets):
	keys = b.keys()
	maxBidder = keys[0]
	c = check_budget(b, rembudgets)
	if c == -1:
		return -1;
	for k in keys:
		if budgets[k] >= b[k]:
			m1 = scaledBid(b[maxBidder], rembudgets[maxBidder], budgets[maxBidder])
			m2 = scaledBid(b[k], rembudgets[k], budgets[k])
			if m1 < m2:
				maxBidder = k
			elif m1 == m2:
				if maxBidder > k:
					maxBidder = k

	return maxBidder

# Runs a particular algorithm 100 times and reports the average revenue
# INPUT: budget of all bidders, bids of all queries for all bidders, all queries, type
# type 1: Greedy, type 2: Balance, type 3: MSVV
# returns average revenue
def calculate_revenue(budget, bids, queries, type):
	total_revenue = 0.0;
	iters = 100;
	for i in range(0,iters):
		random.shuffle(queries)
		budget1 = dict(budget)
		if type ==1:
			revenue = Greedy(budget1, bids, queries);
		elif type == 2:
			revenue = Balance(budget1, bids, queries);
		elif type == 3:
			revenue = MSVV(budget1, dict(budget), bids, queries);
		else:
			revenue = 0.0
		total_revenue += revenue

	return total_revenue/iters


# The main function. takes input from the bidder_dataset.csv file, stores it into dictionaries
# Also, takes input queries from queries.txt.
def main(type):
	budget = dict();
	bids = dict();

	input = pd.read_csv('bidder_dataset.csv')

	for i in range(0, len(input)):
		a = input.iloc[i]['Advertiser']
		k = input.iloc[i]['Keyword']
		bv = input.iloc[i]['Bid Value']
		b = input.iloc[i]['Budget']
		if not (a in budget):
			budget[a] = b
		if not (k in bids):
			bids[k] = {}
		if not (a in bids[k]):
			bids[k][a] = bv


	with open('queries.txt') as f:
		queries = f.readlines()
	queries = [x.strip() for x in queries]

	r = calculate_revenue(budget, bids, queries, type)
	print r
	print (r/sum(budget.values()))

# checks runtime arguemnts and accordingly run algorithms
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Invalid Input"
	else:
		if sys.argv[1] == 'greedy':
			main(1)
		elif sys.argv[1] == 'balance':
			main(2)
		elif sys.argv[1] == 'msvv':
			main(3)
		else:
			print 'Invalid Input'
