from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans
import Levenshtein
import spacy
from collections import defaultdict
import re
from sklearn.cluster import spectral_clustering
import math

en_nlp = spacy.load('en')
def part_of_speech_features(words_array):
	tags = []
	for word in words_array:
		word = word.strip()
		if bool(re.search(r'-', word)):
			word = word.split('-')
			word = word[0]
		curr_tag = en_nlp(word)
		curr_tag = curr_tag[0]
		tags.append(curr_tag)
	#print("Tags: " + str(len(tags)))
	seen_tag_indices = {}
	tag_counter = 0
	# print('tags starting')
	for i in range(len(tags)):
		curr_tag = tags[i].tag_
		# print(curr_tag)
		if curr_tag not in seen_tag_indices:
			seen_tag_indices[curr_tag] = tag_counter
			tag_counter += 1
	# print(seen_tag_indices)
	tag_matrix = np.zeros((len(tags), len(seen_tag_indices)))
	for i in range(len(tags)):
		curr_tag = tags[i].tag_
		tag_matrix[i, seen_tag_indices[curr_tag]] = 1
	return tag_matrix

def part_of_speech_test():
	test_arr = ['this', 'is', 'a', 'words', 'array', 'made', 'of', 
	'only', 'the', 'best', 'phrases', 'arrays' 'have' 'to ''offer', 
	'in', 'the', 'office']
	test_output = part_of_speech_features(test_arr)
	print(test_output)

# testing part of speech features
# part_of_speech_test()

def create_PPMI_matrix(term_context_matrix):
	def divide_array(matrix_array, marginal_array):
		return matrix_array - marginal_array
	
	term_context_matrix = term_context_matrix + pow(10, -3)
	
	#print(str(term_context_matrix))
	context_sum = np.sum(term_context_matrix, axis=0)
	#print(context_sum)
	word_sum = np.sum(term_context_matrix, axis=1)
	#print(word_sum) 	
	
	total_sum1 = np.sum(context_sum)
	
	context_sum = np.log2(context_sum/total_sum1)
	word_sum = np.log2(word_sum / total_sum1)

	term_context_matrix = np.log2(term_context_matrix / total_sum1)

	term_context_matrix = np.apply_along_axis(divide_array, 1, term_context_matrix, context_sum) 
	term_context_matrix = np.apply_along_axis(divide_array, 0, term_context_matrix, word_sum)
	term_context_matrix = np.clip(term_context_matrix, 0, None)
	return term_context_matrix

def compute_cosine_similarity(vector1, vector2):
	'''Computes the cosine similarity of the two input vectors.

	Inputs:
	  vector1: A nx1 numpy array
	  vector2: A nx1 numpy array

	eturns:
	  A scalar similarity value.
	'''
  
	products = vector1 * vector2

	vector1 = np.square(vector1)
	vector2 = np.square(vector2)

	mag1 = sqrt(np.sum(vector1))
	mag2 = sqrt(np.sum(vector2))

	mags = mag1 * mag2

	return np.sum(products)/mags

def create_norm_matrix(term_context_matrix):
	def divide_array(matrix_array, marginal_array):
		return matrix_array / marginal_array
	
	term_context_matrix = term_context_matrix + 1
	
	#print(str(term_context_matrix))

	#print(context_sum)
	word_sum = np.sum(term_context_matrix, axis=1)
	#print(word_sum) 	

	term_context_matrix = np.apply_along_axis(divide_array, 0, term_context_matrix, word_sum)
	term_context_matrix = np.clip(term_context_matrix, 0, None)
	return term_context_matrix

def findClosest(word, cands):
	minDist = 400
	ret = ""
	for val in cands:
		if val == word: continue
		if Levenshtein.distance(word, val) < minDist:
			minDist = Levenshtein.distance(word, val)
			ret = val
	return ret

def create_sim_matrix(matrix):
	def get_mag(row_vector):
		row_vector = np.square(row_vector)
		row_vector = math.sqrt(np.sum(row_vector))
		return row_vector
	def divide_by_mag(mat_vector, mag_vector):
		return mat_vector / mag_vector

	trans_matrix = matrix.transpose()
	sim_matrix = np.matmul(matrix, trans_matrix)
	magnitudes = np.apply_along_axis(get_mag, 1, matrix)
	sim_matrix = np.apply_along_axis(divide_by_mag, 1, sim_matrix, magnitudes)
	sim_matrix = np.apply_along_axis(divide_by_mag, 0, sim_matrix, magnitudes)
	return sim_matrix


def weight_pos(pos_features, matrix):
	row_mean = np.mean(matrix)
	return row_mean * pos_features

# This maps from word  -> list of candidates
word2cands = {}

# This maps from word  -> number of clusters
word2num = {}

# Read the words file.
with open("data/dev_input.txt") as f:
	for line in f:
		word, numclus, cands = line.split(" :: ")
		cands = cands.split()
		word2num[word] = int(numclus)
		word2cands[word] = cands


# Load cooccurrence vectors (question 2)
vec = KeyedVectors.load_word2vec_format("data/coocvec-500mostfreq-window-3.vec.filter")
# Load dense vectors (uncomment for question 3)
# vec = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.filter")    

filename = "dev_output_features.txt"
f = open(filename, "w")
j = 0
for word in word2cands:
	cands = word2cands[word]
	numclusters = word2num[word]
	dic = {}
	for k in range (1, numclusters + 1):
		dic[k] = []
	pos_features = part_of_speech_features(cands)
	#print(len(cands))
	#for cand in cands: 
	#	print(cand)
	matrix = np.zeros((len(cands), 500))
	i = 0
	for val in cands:
		try:
			matrix[i, :] = (vec[val])
		except:
			# print (matrix[i, :])
			continue
			# matrix[i, :] = (vec[findClosest(val, cands)])
		i += 1
	matrix = np.append(matrix, weight_pos(pos_features, matrix), axis=1)
	# print(str(matrix.max()))
	matrix = create_sim_matrix(matrix)
	matrix = 1 - matrix
	kmeans = KMeans(n_clusters = numclusters).fit(matrix)
	# kmeans = spectral_clustering(matrix, n_clusters=numclusters)
	for l in range (0, len(kmeans.labels_)):
		dic[kmeans.labels_[l] + 1].append(cands[l])
	for key, value in dic.items():
		f.write(word + " :: " + str(key) + " :: " + ' '.join(dic[key]) + "\n")
	# print (dic)
	# print(word)
	# print(len(cands))
	# print(len(kmeans.labels_))
	j += 1

f.close()

	# TODO: get word vectors from vec
	# Cluster them with k-means
	# Write the clusters to file.
