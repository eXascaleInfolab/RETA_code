import argparse, sys, json, pickle, torch, operator, random, os, time
from model import *
import numpy as np
from random import randint, randrange, sample
#import cPickle as pickle

from sklearn.metrics import ndcg_score
from torch.optim.lr_scheduler import StepLR

from timeit import default_timer as timer
from datetime import timedelta
from random import shuffle
from multiprocessing import Pool

import torch.multiprocessing as mp

from copy import deepcopy
from pytictoc import TicToc

from batching_reta_plus import *

from multiprocessing import JoinableQueue, Queue, Process

from scipy.sparse import csr_matrix

test_batch_size = 8192

def add_sparsified_types_to_negative_samples (head_entity_id, tail_entity_id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type):
	current_head_types = []
	current_tail_types = []

	# get top head types
	if head_entity_id in entityId2entityTypes:
		current_head_types = entityId2entityTypes[head_entity_id]
		headType2freq = {}
		for h_type in current_head_types:
			h_type_id = type2id[h_type]
			if h_type_id in typeId2frequency:
				headType2freq[h_type] = typeId2frequency[h_type_id]
		sorted_headType2freq = sorted(headType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier]
		current_head_types = [type2id[item[0]] for item in sorted_headType2freq]

	# get top tail types
	if tail_entity_id in entityId2entityTypes:
		current_tail_types = entityId2entityTypes[tail_entity_id]
		tailType2freq = {}
		for t_type in current_tail_types:
			t_type_id = type2id[t_type]
			if t_type_id in typeId2frequency:
				tailType2freq[t_type] = typeId2frequency[t_type_id]
		sorted_tailType2freq = sorted(tailType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier]
		current_tail_types = [type2id[item[0]] for item in sorted_tailType2freq]

	if len(current_head_types)==0:
		current_head_types = [unk_type_id]

	if len(current_tail_types)==0:
		current_tail_types = [unk_type_id]

	headType_tailType_pairs = []
	for h_type_id in current_head_types:
		for t_type_id in current_tail_types:
			headType_tailType_pairs.append(h_type_id)
			headType_tailType_pairs.append(t_type_id)

	for h_type_id in headType_tailType_pairs[0::2]:
		if h_type_id != unk_type_id:
			h_type_name = id2type[h_type_id]
			if head_entity_id!=len(id2type) and h_type_name not in entityId2entityTypes[head_entity_id]:
				print("ERROR head type")


		if t_type_id != unk_type_id:
			t_type_name = id2type[t_type_id]
			if tail_entity_id!=len(id2type) and t_type_name not in entityId2entityTypes[tail_entity_id]:
				print("ERROR tail type")

	return headType_tailType_pairs

def build_entity2sparsifiedTypes (typeId2frequency, entityId2entityTypes, type2id, sparsifier, unk_type_id, id2type):

	entity2sparsifiedTypes = {}
	for entity_id in entityId2entityTypes:
		current_entity_types = entityId2entityTypes[entity_id]
		entityType2freq = {}
		for e_type in current_entity_types:
			e_type_id = type2id[e_type]
			if e_type_id in typeId2frequency:
				entityType2freq[e_type] = typeId2frequency[e_type_id]
		sorted_entityType2freq = sorted(entityType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier]
		current_entity_types = [type2id[item[0]] for item in sorted_entityType2freq]

		entity2sparsifiedTypes[entity_id] = current_entity_types

		for e_type_id in entity2sparsifiedTypes[entity_id]:
			if e_type_id == unk_type_id:
				continue
			e_type_name = id2type[e_type_id]
			if e_type_name not in entityId2entityTypes[entity_id]:
				print("ERROR entity type", e_type_name)

	return entity2sparsifiedTypes

def sort_testing_facts_according_to_arity_fast (new_all_tiled_fact, r_e_tuples):

	arities = np.array([len(x)//2 for x in new_all_tiled_fact])
	unique_arity = np.unique(arities)

	test_facts_by_arities = []
	if len(unique_arity)==1:
		test_facts_by_arities.append(np.asarray(new_all_tiled_fact))
		idx_iterator = range(0, len(r_e_tuples))
		idx2relation_tail_in_scores = dict(zip(idx_iterator, r_e_tuples))
		return test_facts_by_arities, idx2relation_tail_in_scores
	else:
		inds_sort = np.argsort(arities)
		# print(inds_sort)
		sorted_arities = arities[inds_sort]
		# sorted_new_all_tiled_fact = new_all_tiled_fact[inds_sort]
		sorted_new_all_tiled_fact = [new_all_tiled_fact[i] for i in inds_sort]
		sorted_r_e_tuples = [r_e_tuples[i] for i in inds_sort]
		test_facts_by_arities = []
		for i in unique_arity:
			inds = np.where(sorted_arities==i)
			# print(inds)
			# if (i+1) in unique_arity:
			# 	end = np.where(sorted_arities==i+1)
			# else:
			# 	end = len(sorted_new_all_tiled_fact)
			# print("start",start,"end", end)
			tmp_facts = [sorted_new_all_tiled_fact[j] for j in inds[0]]
			# print("#########fact arity ##########", i)
			# print(tmp_facts)
			test_facts_by_arities.append(np.asarray(tmp_facts))

		idx_iterator = range(0, len(sorted_r_e_tuples))
		idx2relation_tail_in_scores = dict(zip(idx_iterator, sorted_r_e_tuples))
		return test_facts_by_arities, idx2relation_tail_in_scores

def sort_testing_facts_according_to_arity (new_all_tiled_fact, relation_tail_in_scores2idx, idx2relation_tail_in_scores):

	arity2testFacts = {}
	arity2relation_tail_pairs = {}
	for fact_it, test_fact in enumerate(new_all_tiled_fact):
		arity = len(test_fact)//2
		if arity not in arity2testFacts:
			arity2testFacts[arity] = []
			arity2relation_tail_pairs[arity] = []
		arity2testFacts[arity].append(test_fact)
		current_rel_tail_pair = idx2relation_tail_in_scores[fact_it] # get r-t pair
		arity2relation_tail_pairs[arity].append(current_rel_tail_pair)

	test_facts_by_arities = []
	new_relation_tail_in_scores2idx = {}
	new_idx2relation_tail_in_scores = {}
	idx = 0
	for arity in arity2testFacts:
		test_facts_by_arities.append(np.asarray(arity2testFacts[arity]))
		all_r_t_pairs_with_current_arity = arity2relation_tail_pairs[arity]
		for r_t_pair in all_r_t_pairs_with_current_arity:
			new_idx2relation_tail_in_scores[idx] = r_t_pair
			new_relation_tail_in_scores2idx[r_t_pair] = idx
			idx += 1
	test_facts_by_arities = np.asarray(test_facts_by_arities)

	return test_facts_by_arities, new_relation_tail_in_scores2idx, new_idx2relation_tail_in_scores

def build_entity2types_dictionaries (dataset_name, entity2id):

	entityName2entityTypes = {}
	entityId2entityTypes = {}
	entityType2entityNames = {}
	entityType2entityIds = {}

	entity2type_file = open(dataset_name + "/entity2types_ttv.txt", "r")

	for line in entity2type_file:
		splitted_line = line.strip().split("\t")
		entity_name = splitted_line[0]
		entity_type = splitted_line[1]

		if entity_name not in entityName2entityTypes:
			entityName2entityTypes[entity_name] = []
		if entity_type not in entityName2entityTypes[entity_name]:
			entityName2entityTypes[entity_name].append(entity_type)

		if entity_type not in entityType2entityNames:
			entityType2entityNames[entity_type] = []
		if entity_name not in entityType2entityNames[entity_type]:
			entityType2entityNames[entity_type].append(entity_name)

		entity_id = entity2id[entity_name]
		if entity_id not in entityId2entityTypes:
			entityId2entityTypes[entity_id] = []
		if entity_type not in entityId2entityTypes[entity_id]:
			entityId2entityTypes[entity_id].append(entity_type)

		if entity_type not in entityType2entityIds:
			entityType2entityIds[entity_type] = []
		if entity_id not in entityType2entityIds[entity_type]:
			entityType2entityIds[entity_type].append(entity_id)

	entity2type_file.close()

	return entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds

def build_head2relation2tails (inputData, entity2id, relation2id, entityId2entityTypes, entitiesEvaluated):

	head2relation2tails = {}
	with open(inputData + "/n-ary_test.json") as f:
		for line in f:
			line = json.loads(line.strip().replace("'", "\""))
			h = list(line.values())[0][0]
			r = list(line.keys())[0]
			t = list(line.values())[0][1]

			hId = entity2id[h]
			rId = relation2id[r]
			tId = entity2id[t]

			if entitiesEvaluated == 'both':
				if hId not in entityId2entityTypes:
					continue
				if tId not in entityId2entityTypes:
					continue

			elif entitiesEvaluated == 'one':
				if hId not in entityId2entityTypes and tId not in entityId2entityTypes:
					continue
				if hId in entityId2entityTypes and tId in entityId2entityTypes:
					continue

			elif entitiesEvaluated == 'none':
				if hId in entityId2entityTypes or tId in entityId2entityTypes:
					continue

			if hId not in head2relation2tails:
				head2relation2tails[hId] = {}

			if rId not in head2relation2tails[hId]:
				head2relation2tails[hId][rId] = []

			if tId not in head2relation2tails[hId][rId]:
				head2relation2tails[hId][rId].append(tId)

	f.close()

	return head2relation2tails

def build_type2relationType2frequency (dataset_name, buildTypeDictionaries):

	type_relation_type_file = open(dataset_name + "/type2relation2type_ttv.txt", "r")

	type2relationType2frequency = {}
	for line in type_relation_type_file:
		splitted_line = line.strip().split("\t")
		head_type = splitted_line[0]
		relation = splitted_line[1]
		tail_type = splitted_line[2]

		relationType = (relation, tail_type)

		if head_type not in type2relationType2frequency:
			type2relationType2frequency[head_type] = {}

		if relationType not in type2relationType2frequency[head_type]:
			type2relationType2frequency[head_type][relationType] = 1
		else:
			type2relationType2frequency[head_type][relationType] += 1

	type_relation_type_file.close()

	return type2relationType2frequency

def build_tensor_matrix (inputData, entity2id, relation2id, entityName2entityTypes, topNfilters, type2relationType2frequency, entityId2typeIds_with_sparsifier, type2id, id2type, device, entitiesEvaluated):

	type_head_tail_entity_matrix = torch.zeros((len(type2id), len(entity2id)), requires_grad=False).to(device)

	train_test_valid = ["train", "test"]

	for ttv in train_test_valid:
		with open(inputData + "/" + ttv + ".txt") as train_file:
			for line in train_file:
				splitted_line = line.strip().split()
				head_entity = splitted_line[0]
				tail_entity = splitted_line[1]

				if head_entity in entityName2entityTypes:
					head_entity_id = entity2id[head_entity]
					head_types = entityName2entityTypes[head_entity]
					for head_type in head_types:
						head_type_id = type2id[head_type]
						type_head_tail_entity_matrix[head_type_id][head_entity_id] = 1

				if tail_entity in entityName2entityTypes:
					tail_entity_id = entity2id[tail_entity]
					tail_types = entityName2entityTypes[tail_entity]
					for tail_type in tail_types:
						tail_type_id = type2id[tail_type]
						type_head_tail_entity_matrix[tail_type_id][tail_entity_id] = 1

		train_file.close()


	for i in range (len(entity2id)):
		current_column = type_head_tail_entity_matrix[:,i]
		non_zero_entries = torch.nonzero(current_column).tolist()
		if len(non_zero_entries)==0:
			type_head_tail_entity_matrix[:,i] = 1

	tailType_relation_headType_tensor = torch.zeros((len(type2id), len(relation2id), len(type2id)), requires_grad=False).to(device)

	list_of_type_relation_type = []
	for h_type in type2relationType2frequency:
		for relation_type in type2relationType2frequency[h_type]:
			relation = relation_type[0]
			t_type = relation_type[1]
			list_of_type_relation_type.append((h_type, relation, t_type))


	list_of_filtered_type_relation_type = []
	for h_type in type2relationType2frequency:
		if topNfilters <= 0:
			sorted_relation_tailType = sorted(type2relationType2frequency[h_type].items(), key=operator.itemgetter(1), reverse=True)
			for list_idx, relationType_frequency in reversed(list(enumerate(sorted_relation_tailType))):
				freq = relationType_frequency[1]
				if freq <= (topNfilters*-1):
					del sorted_relation_tailType[list_idx]

		for relationType_frequency in sorted_relation_tailType:
			relation = relationType_frequency[0][0]
			t_type = relationType_frequency[0][1]
			list_of_filtered_type_relation_type.append((h_type, relation, t_type))
	list_of_type_relation_type = list_of_filtered_type_relation_type

	for trt in list_of_type_relation_type:
		head_type = trt[0]
		relation = trt[1]
		tail_type = trt[2]

		head_type_id = type2id[head_type]
		relation_id = relation2id[relation]
		tail_type_id = type2id[tail_type]

		tailType_relation_headType_tensor[tail_type_id][relation_id][head_type_id] = 1

	return type_head_tail_entity_matrix, tailType_relation_headType_tensor

def chunks (L, n):
	""" Yield successive n-sized chunks from L."""
	for i in range(0, len(L), n):
		yield L[i:i+n]

def my_precision_and_recall_at_k_fast (scores, gt_head_id, head2relation2tails, all_k, id2entity, id2relation, idx2relation_tail_in_scores):

	final_results = []

	# t.tic()
	sorted_scores_idx_all = (-scores).argsort() #descending sort
	# t.toc(("1. sort"))

	sorted_entity_ids = [idx2relation_tail_in_scores[i] for i in sorted_scores_idx_all]

	num_of_total_gt = 0
	inds_true = []
	for r_id in head2relation2tails[gt_head_id]:
		num_of_total_gt += len(head2relation2tails[gt_head_id][r_id])
		for tail_id in head2relation2tails[gt_head_id][r_id]:
			try:
				ind = sorted_entity_ids.index((r_id,tail_id))
			except ValueError:
				ind = -1

			if ind!=-1:
				inds_true.append(ind)

	inds_true.sort()
	inds_true = np.asarray(inds_true)


	for k in all_k:
		if k == 'all':
			if len(inds_true)>0:
				num_of_gt_in_top_k = len(inds_true)
				precision = num_of_gt_in_top_k/(len(scores))
				map_ = 0
				for i,ind in enumerate(inds_true):
					map_ += ((i+1)/(ind+1))/len(inds_true)
					# print("((i+1)/(ind+1)), len(inds_true)", ((i+1)/(ind+1)), len(inds_true))
			else:
				num_of_gt_in_top_k=0
				precision=0
				map_=0

		else:
			tmp_inds = np.where(inds_true<k)[0]
			if len(tmp_inds)>0:
				tmp_inds_true = inds_true[tmp_inds]
				# print("tmp_inds_true", tmp_inds_true)
				num_of_gt_in_top_k = len(tmp_inds_true)
				precision = num_of_gt_in_top_k/k
				map_ = 0
				for i,ind in enumerate(tmp_inds_true):
					map_ += ((i+1)/(ind+1))/len(tmp_inds_true)
			else:
				num_of_gt_in_top_k=0
				precision=0
				map_=0

		recall = num_of_gt_in_top_k/num_of_total_gt

		final_results.append(precision)
		final_results.append(recall)
		final_results.append(map_)


	##NDCG
	for k in all_k:
		if k == 'all':
			if len(inds_true)>0:
				ndcg = np.sum(1/np.log2(inds_true+2))/np.sum(1/np.log2(np.arange(len(inds_true))+2))
			# print(np.sum(1/np.log2(np.arange(len(inds_true))+2)))
			else:
				ndcg=0
		else:
			tmp_inds = np.where(inds_true<k)[0]
			if len(tmp_inds)>0:
				tmp_inds_true = inds_true[tmp_inds]
				ndcg = np.sum(1/np.log2(tmp_inds_true+2))/np.sum(1/np.log2(np.arange(np.minimum(len(inds_true),k))+2))
			else:
				ndcg=0

		final_results.append(ndcg)

	return final_results


def evaluate_all_relation_tail_pairs_v2 (head_id, id2entity, relation2id, id2relation, column, fact, model, arity, device, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, type2id, id2type):

	list_of_scores_per_head_with_filter = []
	# relation_tail_in_scores2idx = {}
	r_e_tuples = []
	idx_iterator = 0

	all_tiled_fact = np.array([])
	new_tiled_fact = []

	for r_id in id2relation:
		r_id = int(r_id)

		all_entities = []

		for e in id2entity:
			e = int(e)
			# relation_tail_in_scores2idx[(r_id, e)] = idx_iterator
			all_entities.append(e)
			r_e_tuples.append((r_id, e))
			idx_iterator += 1

		tiled_fact = np.array(fact*len(all_entities)).reshape(len(all_entities),-1)
		tiled_fact[:,column] = all_entities

		replicated_relation = np.repeat(r_id, len(all_entities))
		tiled_fact[:,0] = replicated_relation
		tiled_fact[:,2] = replicated_relation

		for current_fact in tiled_fact:
			current_head_entity = current_fact[1]
			current_tail_entity = current_fact[3]
			headType_tailType_pairs = add_sparsified_types_to_negative_samples (current_head_entity, current_tail_entity, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type)
			current_fact = current_fact[:4]
			current_fact = np.append(current_fact, headType_tailType_pairs)
			new_tiled_fact.append(current_fact)

	all_tiled_fact_group_by_arity, idx2relation_tail_in_scores = sort_testing_facts_according_to_arity_fast(new_tiled_fact, r_e_tuples)
	# all_tiled_fact_group_by_arity, relation_tail_in_scores2idx, idx2relation_tail_in_scores = sort_testing_facts_according_to_arity(new_tiled_fact, relation_tail_in_scores2idx, idx2relation_tail_in_scores)
	pred = None
	for facts_with_same_arities in all_tiled_fact_group_by_arity:
		batch_of_facts_with_same_arities = list(chunks(facts_with_same_arities, test_batch_size))
		arity = len(batch_of_facts_with_same_arities[0][0])//2
		if pred == None:
			pred = model(batch_of_facts_with_same_arities[0], arity, "testing", device)
		else:
			pred_tmp = model(batch_of_facts_with_same_arities[0], arity, "testing", device)
			pred = torch.cat((pred, pred_tmp))
		for batch_it in range(1, len(batch_of_facts_with_same_arities)):
			pred_tmp = model(batch_of_facts_with_same_arities[batch_it], arity, "testing", device)
			pred = torch.cat((pred, pred_tmp))

	score_with_filter = pred.view(-1).detach().cpu().numpy()
	list_of_scores_per_head_with_filter.append(score_with_filter)

	return list_of_scores_per_head_with_filter, idx2relation_tail_in_scores

def get_reta_filtered_results(head_id, id2entity, type2relationType2frequency, topNfilters, column, fact, model, arity, device, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, type2id, relation2id, atLeast, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, id2type, entity2sparsifiedTypes, entitiesEvaluated, rt_negative_candidates):
	list_of_scores_per_head_with_filter = []
	idx2relation_tail_in_scores = {}

	h = id2entity[head_id]

	tiled_fact = np.array([])

	all_head_types = type_head_tail_entity_matrix[:,head_id]
	tailType_relation_matrix = torch.matmul(tailType_relation_headType_tensor, all_head_types)
	tailType_relation_matrix = torch.transpose(tailType_relation_matrix, 0, 1)
	relation_entity_matrix = torch.matmul(tailType_relation_matrix, type_head_tail_entity_matrix)

	relation_entity_matrix[relation_entity_matrix < atLeast] = 0

	if torch.nonzero(relation_entity_matrix).shape[0] != 0:
		filtered_relation_tail_pairs = torch.nonzero(relation_entity_matrix)

		all_relations = filtered_relation_tail_pairs[:,0].tolist()
		entities_without_duplicates = filtered_relation_tail_pairs[:,1].tolist()
		r_e_tuples = list(zip(all_relations, entities_without_duplicates))

		if len(entities_without_duplicates) > 0:
			tiled_fact = np.array(fact*len(entities_without_duplicates)).reshape(len(entities_without_duplicates),-1)
			tiled_fact[:,column] = entities_without_duplicates

			tiled_fact[:,0] = all_relations
			tiled_fact[:,2] = all_relations

			head_sparsified_types = []
			current_head_entity = tiled_fact[0][1]
			if current_head_entity in entity2sparsifiedTypes:
				head_sparsified_types = entity2sparsifiedTypes[current_head_entity]
			else:
				head_sparsified_types = [unk_type_id]

			if tiled_fact.shape[0] > 0:
				tmp_t_types = [entity2sparsifiedTypes.get(x,[]) for x in list(tiled_fact[:,3])]
				inds = [i for i, x in enumerate(tmp_t_types) if len(x) == 0]
				for i in inds:
					tmp_t_types[i] = [unk_type_id]

				tmp_t_types_0 = np.array([x[0] for x in tmp_t_types])

				tmp_array = tiled_fact[:,:4]
				for i in range( len(head_sparsified_types) ):
					tmp_array = np.c_[tmp_array, np.full((tiled_fact.shape[0],1),np.array(head_sparsified_types[i])), tmp_t_types_0.T]
				new_tiled_fact = list(tmp_array)

				inds_t = [i for i, x in enumerate(tmp_t_types) if len(x) > 1]
				for i in inds_t:
					for j in range(1, len(tmp_t_types[i])):
						for k in range( len(head_sparsified_types) ):
							new_tiled_fact[i] = np.concatenate((new_tiled_fact[i] , np.array([head_sparsified_types[k], tmp_t_types[i][j]]) ))

				new_tiled_fact, idx2relation_tail_in_scores = sort_testing_facts_according_to_arity_fast(new_tiled_fact, r_e_tuples)
				return new_tiled_fact, idx2relation_tail_in_scores
	return [], []

def get_scores_from_reta_filtered_results (new_tiled_fact, model, device):
	list_of_scores_per_head_with_filter = []
	# print(len(new_tiled_fact))
	if len(new_tiled_fact)>0:
		pred = None
		for facts_with_same_arities in new_tiled_fact:
			batch_of_facts_with_same_arities = list(chunks(facts_with_same_arities, test_batch_size))
			arity = len(batch_of_facts_with_same_arities[0][0])//2
			if pred == None:
				pred = model(batch_of_facts_with_same_arities[0], arity, "testing", device)
			else:
				pred_tmp = model(batch_of_facts_with_same_arities[0], arity, "testing", device)
				pred = torch.cat((pred, pred_tmp))
			for batch_it in range(1, len(batch_of_facts_with_same_arities)):
				pred_tmp = model(batch_of_facts_with_same_arities[batch_it], arity, "testing", device)
				pred = torch.cat((pred, pred_tmp))

		score_with_filter = pred.view(-1).detach().cpu().numpy()
		list_of_scores_per_head_with_filter.append(score_with_filter)

	return list_of_scores_per_head_with_filter


def get_filtered_relations_and_tails_filter1_filter2_hybrid (head_id, id2entity, type2relationType2frequency, topNfilters, column, fact, model, arity, device, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, type2id, relation2id, atLeast, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, id2type, entity2sparsifiedTypes, entitiesEvaluated, rt_negative_candidates):

	list_of_scores_per_head_with_filter = []
	idx2relation_tail_in_scores = {}

	h = id2entity[head_id]

	tiled_fact = np.array([])

	# filtered_relation_tail_pairs = rt_negative_candidates[head_id]
	# t =  TicToc()
	# t.tic()
	all_head_types = type_head_tail_entity_matrix[:,head_id]
	tailType_relation_matrix = torch.matmul(tailType_relation_headType_tensor, all_head_types)
	tailType_relation_matrix = torch.transpose(tailType_relation_matrix, 0, 1)
	relation_entity_matrix = torch.matmul(tailType_relation_matrix, type_head_tail_entity_matrix)

	relation_entity_matrix[relation_entity_matrix < atLeast] = 0
	# t.toc(("filtering time (tensor multiplication)"))

	if torch.nonzero(relation_entity_matrix).shape[0] != 0:
		filtered_relation_tail_pairs = torch.nonzero(relation_entity_matrix)

		all_relations = filtered_relation_tail_pairs[:,0].tolist()
		entities_without_duplicates = filtered_relation_tail_pairs[:,1].tolist()
		r_e_tuples = list(zip(all_relations, entities_without_duplicates))
		# idx_iterator = range(0, len(r_e_tuples))
		# relation_tail_in_scores2idx = dict(zip(r_e_tuples, idx_iterator))
		# idx2relation_tail_in_scores = dict(zip(idx_iterator, r_e_tuples))

		if len(entities_without_duplicates) > 0:
			tiled_fact = np.array(fact*len(entities_without_duplicates)).reshape(len(entities_without_duplicates),-1)
			tiled_fact[:,column] = entities_without_duplicates

			tiled_fact[:,0] = all_relations
			tiled_fact[:,2] = all_relations

			head_sparsified_types = []
			current_head_entity = tiled_fact[0][1]
			if current_head_entity in entity2sparsifiedTypes:
				head_sparsified_types = entity2sparsifiedTypes[current_head_entity]
			else:
				head_sparsified_types = [unk_type_id]


			if tiled_fact.shape[0] > 0:

				tmp_t_types = [entity2sparsifiedTypes.get(x,[]) for x in list(tiled_fact[:,3])]
				inds = [i for i, x in enumerate(tmp_t_types) if len(x) == 0]
				for i in inds:
					tmp_t_types[i] = [unk_type_id]

				tmp_t_types_0 = np.array([x[0] for x in tmp_t_types])

				tmp_array = tiled_fact[:,:4]
				for i in range( len(head_sparsified_types) ):
					tmp_array = np.c_[tmp_array, np.full((tiled_fact.shape[0],1),np.array(head_sparsified_types[i])), tmp_t_types_0.T]
				new_tiled_fact = list(tmp_array)

				inds_t = [i for i, x in enumerate(tmp_t_types) if len(x) > 1]
				for i in inds_t:
					for j in range(1, len(tmp_t_types[i])):
						for k in range( len(head_sparsified_types) ):
							new_tiled_fact[i] = np.concatenate((new_tiled_fact[i] , np.array([head_sparsified_types[k], tmp_t_types[i][j]]) ))

				new_tiled_fact, idx2relation_tail_in_scores = sort_testing_facts_according_to_arity_fast(new_tiled_fact, r_e_tuples) # new_tiled_fact size: (num of mini batches, num of facts, arity)

				# t.tic()
				pred = None
				for facts_with_same_arities in new_tiled_fact:
					batch_of_facts_with_same_arities = list(chunks(facts_with_same_arities, test_batch_size))
					arity = len(batch_of_facts_with_same_arities[0][0])//2
					if pred == None:
						pred = model(batch_of_facts_with_same_arities[0], arity, "testing", device)
					else:
						pred_tmp = model(batch_of_facts_with_same_arities[0], arity, "testing", device)
						pred = torch.cat((pred, pred_tmp))
					for batch_it in range(1, len(batch_of_facts_with_same_arities)):
						pred_tmp = model(batch_of_facts_with_same_arities[batch_it], arity, "testing", device)
						pred = torch.cat((pred, pred_tmp))

				score_with_filter = pred.view(-1).detach().cpu().numpy()
				list_of_scores_per_head_with_filter.append(score_with_filter)
				# t.toc("scoring time (forward pass)")

	return list_of_scores_per_head_with_filter, idx2relation_tail_in_scores

def build_headTail2hTypetType (inputData, entity2id, type2id, entity2types, sparsifier, typeId2frequency, buildTypeDictionaries):

	if buildTypeDictionaries == "True":

		headTail2hTypetType = {}
		entityId2typeIds_with_sparsifier = {}

		train_test_valid = ["train", "test"]

		for ttv in train_test_valid:
			with open(inputData + "/n-ary_" + ttv + ".json") as f:
				for line_number, line in enumerate(f):
					line = json.loads(line.strip().replace("'", "\""))
					h = list(line.values())[0][0]
					t = list(line.values())[0][1]

					if h in entity2types and t in entity2types: # both head and tail have types
						h_types = entity2types[h]
						t_types = entity2types[t]

						h_id = entity2id[h]
						t_id = entity2id[t]

						h_t_tuple = (h_id, t_id)
						if h_t_tuple not in headTail2hTypetType:
							headTail2hTypetType[h_t_tuple] = []

						for h_type in h_types:
							h_type_id = type2id[h_type]
							for t_type in t_types:
								t_type_id = type2id[t_type]

								types_tuple = (h_type_id, t_type_id)

								if types_tuple not in headTail2hTypetType[h_t_tuple]:
									headTail2hTypetType[h_t_tuple].append(types_tuple)

					elif h in entity2types and t not in entity2types:
						h_types = entity2types[h]
						h_id = entity2id[h]
						if h_id not in entityId2typeIds_with_sparsifier:
							entityId2typeIds_with_sparsifier[h_id] = []
							for h_type in h_types:
								h_type_id = type2id[h_type]

								if h_type_id not in entityId2typeIds_with_sparsifier[h_id]:
									entityId2typeIds_with_sparsifier[h_id].append(h_type_id)

					elif h not in entity2types and t in entity2types:
						t_types = entity2types[t]
						t_id = entity2id[t]
						if t_id not in entityId2typeIds_with_sparsifier:
							entityId2typeIds_with_sparsifier[t_id] = []
							for t_type in t_types:
								t_type_id = type2id[t_type]

								if t_type_id not in entityId2typeIds_with_sparsifier[t_id]:
									entityId2typeIds_with_sparsifier[t_id].append(t_type_id)

			f.close()

		if sparsifier >= 0:

			for k in headTail2hTypetType:
				current_headType2freq = {}
				current_tailType2freq = {}

				current_head_types = [item[0] for item in headTail2hTypetType[k]]
				current_tail_types = [item[1] for item in headTail2hTypetType[k]]

				current_unique_head_types = list(dict.fromkeys(current_head_types))
				current_unique_tail_types = list(dict.fromkeys(current_tail_types))

				current_headType2freq = {}
				for h_type in current_unique_head_types:
					current_headType2freq[h_type] = typeId2frequency[h_type]
				current_tailType2freq = {}
				for t_type in current_unique_tail_types:
					current_tailType2freq[t_type] = typeId2frequency[t_type]

				sorted_current_headType2freq = sorted(current_headType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier]
				sorted_current_tailType2freq = sorted(current_tailType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier]

				topNheadTypes = [item[0] for item in sorted_current_headType2freq]
				topNtailTypes = [item[0] for item in sorted_current_tailType2freq]

				new_headType_tailType_list = []
				for hType_tTypy_tuple in headTail2hTypetType[k]:
					if hType_tTypy_tuple[0] in topNheadTypes and hType_tTypy_tuple[1] in topNtailTypes:
						new_headType_tailType_list.append(hType_tTypy_tuple)

				headTail2hTypetType[k] = new_headType_tailType_list

			for k in headTail2hTypetType:
				if len(headTail2hTypetType[k]) > (sparsifier*sparsifier):
					print("ERROR in the sparsification: the maximum length of each list must be", sparsifier*sparsifier)


			# entityId2typeIds_with_sparsifier
			for k in entityId2typeIds_with_sparsifier:
				current_entityType2freq = {}

				current_entity_types = entityId2typeIds_with_sparsifier[k] # get current head types

				current_unique_entity_types = list(dict.fromkeys(current_entity_types)) # get current unique head types

				# get entityType2freq
				current_entityType2freq = {}
				for e_type in current_unique_entity_types:
					if e_type in typeId2frequency:
						current_entityType2freq[e_type] = typeId2frequency[e_type]
					else:
						pass


				sorted_current_entityType2freq = sorted(current_entityType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier]

				topNentityTypes = [item[0] for item in sorted_current_entityType2freq]

				new_entiyType_list = []
				for eType in entityId2typeIds_with_sparsifier[k]:
					if eType in topNentityTypes:
						new_entiyType_list.append(eType)

				entityId2typeIds_with_sparsifier[k] = new_entiyType_list

			for k in entityId2typeIds_with_sparsifier:
				if len(entityId2typeIds_with_sparsifier[k]) > (sparsifier*sparsifier):
					print("ERROR in the sparsification: the maximum length of each list must be", sparsifier*sparsifier)

		for k in headTail2hTypetType:
			headTail2hTypetType[k] = np.asarray(headTail2hTypetType[k])
		for k in entityId2typeIds_with_sparsifier:
			entityId2typeIds_with_sparsifier[k] = np.asarray(entityId2typeIds_with_sparsifier[k])

		with open(inputData + "/headTail2hTypetType.pickle", "wb") as handle:
			pickle.dump(headTail2hTypetType, handle, protocol=pickle.HIGHEST_PROTOCOL)
		handle.close()
		with open(inputData + "/entityId2typeIds_with_sparsifier.pickle", "wb") as handle:
			pickle.dump(entityId2typeIds_with_sparsifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
		handle.close()

	elif buildTypeDictionaries == "False":
		with open(inputData + "/headTail2hTypetType.pickle", "rb") as handle:
			headTail2hTypetType = pickle.load(handle)
		handle.close()
		with open(inputData + "/entityId2typeIds_with_sparsifier.pickle", "rb") as handle:
			entityId2typeIds_with_sparsifier = pickle.load(handle)
		handle.close()

	return headTail2hTypetType, entityId2typeIds_with_sparsifier

def build_type2id_v2 (inputData):
	type2id = {}
	id2type = {}
	type_counter = 0
	with open(inputData + "/entity2types_ttv.txt") as entity2type_file:
		for line in entity2type_file:
			splitted_line = line.strip().split("\t")
			entity_type = splitted_line[1]

			if entity_type not in type2id:
				type2id[entity_type] = type_counter
				id2type[type_counter] = entity_type
				type_counter += 1

	entity2type_file.close()
	return type2id, id2type

def build_typeId2frequency (dataset_name, type2id):
	typeId2frequency = {}
	type_relation_type_file = open(dataset_name + "/type2relation2type_ttv.txt", "r")

	for line in type_relation_type_file:
		splitted_line = line.strip().split("\t")
		head_type = splitted_line[0]
		tail_type = splitted_line[2]
		head_type_id = type2id[head_type]
		tail_type_id = type2id[tail_type]

		if head_type_id not in typeId2frequency:
			typeId2frequency[head_type_id] = 0
		if tail_type_id not in typeId2frequency:
			typeId2frequency[tail_type_id] = 0

		typeId2frequency[head_type_id] += 1
		typeId2frequency[tail_type_id] += 1

	type_relation_type_file.close()

	return typeId2frequency

def add_type_pair_to_fact (train, test, valid, headTail2hTypetType, entityId2typeIds_with_sparsifier, unk_type_id):

	for it, ttv in enumerate([train, test, valid]):
		if it == 0:
			current_ttv = train
		elif it == 1:
			current_ttv = test
		elif it == 2:
			current_ttv = valid

		arity2fact = {}
		for d in current_ttv:
			for fact_iterator, current_fact in enumerate(d):
				current_head = current_fact[1]
				current_tail = current_fact[3]
				if (current_head, current_tail) in headTail2hTypetType and len(headTail2hTypetType[(current_head, current_tail)]) > 0:
					list_of_indexes = headTail2hTypetType[(current_head, current_tail)]
					current_fact = np.append(current_fact, list_of_indexes)

				elif current_head in entityId2typeIds_with_sparsifier and current_tail not in entityId2typeIds_with_sparsifier:
					list_of_indexes = entityId2typeIds_with_sparsifier[current_head]

					for current_h_type in list_of_indexes:
						current_fact = np.append(current_fact, [current_h_type, unk_type_id])

				elif current_head not in entityId2typeIds_with_sparsifier and current_tail in entityId2typeIds_with_sparsifier:
					list_of_indexes = entityId2typeIds_with_sparsifier[current_tail]

					for current_t_type in list_of_indexes:
						current_fact = np.append(current_fact, [unk_type_id, current_t_type])

				elif current_head not in entityId2typeIds_with_sparsifier and current_tail not in entityId2typeIds_with_sparsifier:
					current_fact = np.append(current_fact, [unk_type_id, unk_type_id])
				else:
					print("ERROR: it should go in one of the previous cases")

				current_fact = tuple(current_fact)

				new_arity = len(current_fact)
				if new_arity not in arity2fact:
					arity2fact[new_arity] = []
				arity2fact[new_arity].append(current_fact)

		new_list = []
		for new_arity in sorted(arity2fact.keys()):
			new_dict = {}
			for fact in arity2fact[new_arity]:
				new_dict[fact] = [1]
			new_list.append(new_dict)

		if it == 0:
			train = new_list
		elif it == 1:
			test = new_list
		elif it == 2:
			valid = new_list

	return train, test, valid

def evaluate_model_v2 (model, test, id2entity, type2relationType2frequency, topNfilters, atLeast, device, type2id, id2type, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, relation2id, head2relation2tails, id2relation, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, entity2sparsifiedTypes, indir, entitiesEvaluated, testing_head_ids_subset, rt_negative_candidates):

	number_of_actual_heads_evaluated = 0
	number_of_heads_not_evaluated_1 = 0
	number_of_heads_not_evaluated_3 = 0
	number_of_heads_not_evaluated_4 = 0
	number_of_heads_not_evaluated_5 = 0

	my_precision10 = 0
	my_recall10 = 0
	my_map10 = 0

	my_precision5 = 0
	my_recall5 = 0
	my_map5 = 0

	my_precision3 = 0
	my_recall3 = 0
	my_map3 = 0

	my_precision1 = 0
	my_recall1 = 0
	my_map1 = 0

	my_precisionAll = 0
	my_recallAll = 0
	my_mapAll = 0

	my_ndcg10 = 0
	my_ndcg5 = 0
	my_ndcg3 = 0
	my_ndcg1 = 0
	my_ndcgAll = 0

	visited_head = {}

	print("evaluate_model")
	model.eval()

	t3 = TicToc()
	t3.tic()

	# reta_filtered_results = {}

	with torch.no_grad():

		list_of_testing_facts = []
		for test_fact_grouped_by_arity in test:
			for test_fact in test_fact_grouped_by_arity:
				list_of_testing_facts.append(test_fact)

		# pool=Pool(1)
		para_results=[]
		for fact_progress, fact in enumerate(list_of_testing_facts):

			list_of_scores_per_head_with_filter = []
			fact = list(fact)
			arity = int(len(fact)/2)
			head_id = fact[1]

			# if "FB15k" in indir and entitiesEvaluated!="none":
			# 	if head_id not in testing_head_ids_subset:
			# 		continue

			if fact_progress%1000==0:
				t3.toc("EVALUATING 1000 FACTS:")
				print("evaluation progress:", fact_progress, "/", len(list_of_testing_facts), " | number_of_actual_heads_evaluated:", number_of_actual_heads_evaluated)
				sys.stdout.flush()
				t3.tic()

			column = 3

			# correct_index = fact[column]

			if head_id not in head2relation2tails:
				number_of_heads_not_evaluated_5 += 1
				continue

			if head_id not in id2entity:
				continue

			head_name = id2entity[head_id]

			if head_name in visited_head:
				number_of_heads_not_evaluated_3 += 1
				continue

			visited_head[head_name] = 1
			# print(len(visited_head))

			if head_name in entityName2entityTypes:

				new_tiled_fact, idx2relation_tail_in_scores = get_reta_filtered_results(head_id, id2entity, type2relationType2frequency, topNfilters, column, fact, model, arity, device, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, type2id, relation2id, atLeast, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, id2type, entity2sparsifiedTypes, entitiesEvaluated, rt_negative_candidates)
				list_of_scores_per_head_with_filter = get_scores_from_reta_filtered_results (new_tiled_fact, model, device)
				# t3.toc("filtering time:")

			if head_name not in entityName2entityTypes:
				number_of_heads_not_evaluated_4 += 1
				if entitiesEvaluated=="none":
					list_of_scores_per_head_with_filter, idx2relation_tail_in_scores = evaluate_all_relation_tail_pairs_v2 (head_id, id2entity, relation2id, id2relation, column, fact, model, arity, device, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, type2id, id2type)
			elif len(list_of_scores_per_head_with_filter)==0 or len(list_of_scores_per_head_with_filter[0])<10:
				number_of_heads_not_evaluated_1 += 1
				if entitiesEvaluated=="none":
					list_of_scores_per_head_with_filter, idx2relation_tail_in_scores = evaluate_all_relation_tail_pairs_v2 (head_id, id2entity, relation2id, id2relation, column, fact, model, arity, device, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, type2id, id2type)


			if len(list_of_scores_per_head_with_filter)>0:

				number_of_actual_heads_evaluated += 1
				score_with_with_filter = np.concatenate(list_of_scores_per_head_with_filter).ravel()

				scores = score_with_with_filter
				# para_results.append(pool.apply_async(my_precision_and_recall_at_k_fast,args = (scores, head_id, head2relation2tails, [10,5,3,1,'all'], id2entity, id2relation, idx2relation_tail_in_scores)))
				para_results.append(my_precision_and_recall_at_k_fast (scores, head_id, head2relation2tails, [10,5,3,1,'all'], id2entity, id2relation, idx2relation_tail_in_scores))

			if "FB15k" in indir and fact_progress>10000: # for quick test
				break;

		# pool.close()
		# pool.join()
		# para_results =[r.get() for r in para_results]

		for final_results in para_results:
			my_precision10 += final_results[0]
			my_recall10 += final_results[1]
			my_map10 += final_results[2]

			my_precision5 += final_results[3]
			my_recall5 += final_results[4]
			my_map5 += final_results[5]

			my_precision3 += final_results[6]
			my_recall3 += final_results[7]
			my_map3 += final_results[8]

			my_precision1 += final_results[9]
			my_recall1 += final_results[10]
			my_map1 += final_results[11]

			my_precisionAll += final_results[12]
			my_recallAll += final_results[13]
			my_mapAll += final_results[14]

			my_ndcg10 += final_results[15]
			my_ndcg5 += final_results[16]
			my_ndcg3 += final_results[17]
			my_ndcg1 += final_results[18]
			my_ndcgAll += final_results[19]

		print("number_of_actual_heads_evaluated:", number_of_actual_heads_evaluated)
		print("number of heads not evaluated (because the filter was too strict and it did not generated any relation-tail pairs):", number_of_heads_not_evaluated_1)
		print("number of heads not evaluated (because the head was already evaluated):", number_of_heads_not_evaluated_3)
		print("number of heads not evaluated (because the head did not have a type):", number_of_heads_not_evaluated_4)
		print("number of heads not evaluated (because the head did not have a type):", number_of_heads_not_evaluated_4)
		print("number of heads not evaluated (because the head is not in the GT):", number_of_heads_not_evaluated_5)

		if number_of_actual_heads_evaluated != 0:
			print("my_precision10:", my_precision10, "/", number_of_actual_heads_evaluated, "=", my_precision10/number_of_actual_heads_evaluated)
			print("my_recall10:", my_recall10, "/", number_of_actual_heads_evaluated, "=", my_recall10/number_of_actual_heads_evaluated)
			print("my_map10:", my_map10, "/", number_of_actual_heads_evaluated, "=", my_map10/number_of_actual_heads_evaluated)
			print("my_precision5:", my_precision5, "/", number_of_actual_heads_evaluated, "=", my_precision5/number_of_actual_heads_evaluated)
			print("my_recall5:", my_recall5, "/", number_of_actual_heads_evaluated, "=", my_recall5/number_of_actual_heads_evaluated)
			print("my_map5:", my_map5, "/", number_of_actual_heads_evaluated, "=", my_map5/number_of_actual_heads_evaluated)
			print("my_precision3:", my_precision3, "/", number_of_actual_heads_evaluated, "=", my_precision3/number_of_actual_heads_evaluated)
			print("my_recall3:", my_recall3, "/", number_of_actual_heads_evaluated, "=", my_recall3/number_of_actual_heads_evaluated)
			print("my_map3:", my_map3, "/", number_of_actual_heads_evaluated, "=", my_map3/number_of_actual_heads_evaluated)
			print("my_precision1:", my_precision1, "/", number_of_actual_heads_evaluated, "=", my_precision1/number_of_actual_heads_evaluated)
			print("my_recall1:", my_recall1, "/", number_of_actual_heads_evaluated, "=", my_recall1/number_of_actual_heads_evaluated)
			print("my_map1:", my_map1, "/", number_of_actual_heads_evaluated, "=", my_map1/number_of_actual_heads_evaluated)
			print("my_precisionAll:", my_precisionAll, "/", number_of_actual_heads_evaluated, "=", my_precisionAll/number_of_actual_heads_evaluated)
			print("my_recallAll:", my_recallAll, "/", number_of_actual_heads_evaluated, "=", my_recallAll/number_of_actual_heads_evaluated)
			print("my_mapAll:", my_mapAll, "/", number_of_actual_heads_evaluated, "=", my_mapAll/number_of_actual_heads_evaluated)
			print("my_ndcg10:", my_ndcg10, "/", number_of_actual_heads_evaluated, "=", my_ndcg10/number_of_actual_heads_evaluated)
			print("my_ndcg5:", my_ndcg5, "/", number_of_actual_heads_evaluated, "=", my_ndcg5/number_of_actual_heads_evaluated)
			print("my_ndcg3:", my_ndcg3, "/", number_of_actual_heads_evaluated, "=", my_ndcg3/number_of_actual_heads_evaluated)
			print("my_ndcg1:", my_ndcg1, "/", number_of_actual_heads_evaluated, "=", my_ndcg1/number_of_actual_heads_evaluated)
			print("my_ndcgAll:", my_ndcgAll, "/", number_of_actual_heads_evaluated, "=", my_ndcgAll/number_of_actual_heads_evaluated)

			print("\n")
			my_precision10 = my_precision10/number_of_actual_heads_evaluated
			my_precision5 = my_precision5/number_of_actual_heads_evaluated
			my_precision3 = my_precision3/number_of_actual_heads_evaluated
			my_precision1 = my_precision1/number_of_actual_heads_evaluated
			my_precisionAll = my_precisionAll/number_of_actual_heads_evaluated

			my_recall10 = my_recall10/number_of_actual_heads_evaluated
			my_recall5 = my_recall5/number_of_actual_heads_evaluated
			my_recall3 = my_recall3/number_of_actual_heads_evaluated
			my_recall1 = my_recall1/number_of_actual_heads_evaluated
			my_recallAll = my_recallAll/number_of_actual_heads_evaluated

			my_map10 = my_map10/number_of_actual_heads_evaluated
			my_map5 = my_map5/number_of_actual_heads_evaluated
			my_map3 = my_map3/number_of_actual_heads_evaluated
			my_map1 = my_map1/number_of_actual_heads_evaluated
			my_mapAll = my_mapAll/number_of_actual_heads_evaluated

			my_ndcg10 = my_ndcg10/number_of_actual_heads_evaluated
			my_ndcg5 = my_ndcg5/number_of_actual_heads_evaluated
			my_ndcg3 = my_ndcg3/number_of_actual_heads_evaluated
			my_ndcg1 = my_ndcg1/number_of_actual_heads_evaluated
			my_ndcgAll = my_ndcgAll/number_of_actual_heads_evaluated

			print(str(my_recall10) + "\t" + str(my_recall5) + "\t" + str(my_mapAll) + "\t" + str(my_ndcgAll))

		else:
			print("The number of actual heads evaluated is 0, meaning that the filter is too strict!")

def evaluate_model_validation (model, test, id2entity, type2relationType2frequency, topNfilters, atLeast, device, type2id, id2type, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, relation2id, head2relation2tails, id2relation, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, entity2sparsifiedTypes, indir, entitiesEvaluated, testing_head_ids_subset, rt_negative_candidates):

	number_of_actual_heads_evaluated = 0
	number_of_heads_not_evaluated_1 = 0
	number_of_heads_not_evaluated_3 = 0
	number_of_heads_not_evaluated_4 = 0
	number_of_heads_not_evaluated_5 = 0

	my_precision10 = 0
	my_recall10 = 0
	my_map10 = 0

	my_precision5 = 0
	my_recall5 = 0
	my_map5 = 0

	my_precision3 = 0
	my_recall3 = 0
	my_map3 = 0

	my_precision1 = 0
	my_recall1 = 0
	my_map1 = 0

	my_precisionAll = 0
	my_recallAll = 0
	my_mapAll = 0

	my_ndcg10 = 0
	my_ndcg5 = 0
	my_ndcg3 = 0
	my_ndcg1 = 0
	my_ndcgAll = 0

	visited_head = {}

	print("evaluate_model (validation on 1000 heads only)")
	model.eval()

	t3 = TicToc()
	t3.tic()

	with torch.no_grad():

		list_of_testing_facts = []
		for test_fact_grouped_by_arity in test:
			for test_fact in test_fact_grouped_by_arity:
				list_of_testing_facts.append(test_fact)

		# pool=Pool(1)
		para_results=[]
		for fact_progress, fact in enumerate(list_of_testing_facts):
			list_of_scores_per_head_with_filter = []
			fact = list(fact)
			arity = int(len(fact)/2)
			head_id = fact[1]

			# if "FB15k" in indir and entitiesEvaluated!="none":
			# 	if head_id not in testing_head_ids_subset:
			# 		continue

			if fact_progress%1000==0:
				t3.toc("EVALUATING 100 FACTS:")
				print("evaluation progress:", fact_progress, "/", len(list_of_testing_facts), " | number_of_actual_heads_evaluated:", number_of_actual_heads_evaluated)
				sys.stdout.flush()
				t3.tic()

			column = 3

			# correct_index = fact[column]

			if head_id not in head2relation2tails:
				# final_results = []
				# return final_results
				number_of_heads_not_evaluated_5 += 1
				continue

			if head_id not in id2entity:
				continue
				# final_results = []
				# return final_results

			head_name = id2entity[head_id]

			if head_name in visited_head:
				# final_results = []
				# return final_results
				number_of_heads_not_evaluated_3 += 1
				continue

			visited_head[head_name] = 1

			if head_name in entityName2entityTypes:

				new_tiled_fact, idx2relation_tail_in_scores = get_reta_filtered_results(head_id, id2entity, type2relationType2frequency, topNfilters, column, fact, model, arity, device, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, type2id, relation2id, atLeast, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, id2type, entity2sparsifiedTypes, entitiesEvaluated, rt_negative_candidates)

				list_of_scores_per_head_with_filter = get_scores_from_reta_filtered_results (new_tiled_fact, model, device)

				# print("new_tiled_fact size",new_tiled_fact[0].size)
				# print("new_tiled_fact",new_tiled_fact[0])

			if head_name not in entityName2entityTypes:
				number_of_heads_not_evaluated_4 += 1
				if entitiesEvaluated=="none":
					list_of_scores_per_head_with_filter, idx2relation_tail_in_scores = evaluate_all_relation_tail_pairs_v2 (head_id, id2entity, relation2id, id2relation, column, fact, model, arity, device, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, type2id, id2type)
			elif len(list_of_scores_per_head_with_filter)==0 or len(list_of_scores_per_head_with_filter[0])<10:
				number_of_heads_not_evaluated_1 += 1
				if entitiesEvaluated=="none":
					list_of_scores_per_head_with_filter, idx2relation_tail_in_scores = evaluate_all_relation_tail_pairs_v2 (head_id, id2entity, relation2id, id2relation, column, fact, model, arity, device, sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, type2id, id2type)


			if len(list_of_scores_per_head_with_filter)>0:

				number_of_actual_heads_evaluated += 1
				score_with_with_filter = np.concatenate(list_of_scores_per_head_with_filter).ravel()

				scores = score_with_with_filter
				# final_results = my_precision_and_recall_at_k_fast (scores, head_id, head2relation2tails, [10,5,3,1,'all'], id2entity, id2relation, idx2relation_tail_in_scores)
				# para_results.append(pool.apply_async(my_precision_and_recall_at_k_fast,args = (scores, head_id, head2relation2tails, [10,5,3,1,'all'], id2entity, id2relation, idx2relation_tail_in_scores)))
				para_results.append(my_precision_and_recall_at_k_fast (scores, head_id, head2relation2tails, [10,5,3,1,'all'], id2entity, id2relation, idx2relation_tail_in_scores))

			if fact_progress>1000:
				break;

		# pool.close()
		# pool.join()
		# para_results =[r.get() for r in para_results]

		for final_results in para_results:
			my_precision10 += final_results[0]
			my_recall10 += final_results[1]
			my_map10 += final_results[2]

			my_precision5 += final_results[3]
			my_recall5 += final_results[4]
			my_map5 += final_results[5]

			my_precision3 += final_results[6]
			my_recall3 += final_results[7]
			my_map3 += final_results[8]

			my_precision1 += final_results[9]
			my_recall1 += final_results[10]
			my_map1 += final_results[11]

			my_precisionAll += final_results[12]
			my_recallAll += final_results[13]
			my_mapAll += final_results[14]

			my_ndcg10 += final_results[15]
			my_ndcg5 += final_results[16]
			my_ndcg3 += final_results[17]
			my_ndcg1 += final_results[18]
			my_ndcgAll += final_results[19]

		if number_of_actual_heads_evaluated != 0:

			my_precision10 = my_precision10/number_of_actual_heads_evaluated
			my_precision5 = my_precision5/number_of_actual_heads_evaluated
			my_precision3 = my_precision3/number_of_actual_heads_evaluated
			my_precision1 = my_precision1/number_of_actual_heads_evaluated
			my_precisionAll = my_precisionAll/number_of_actual_heads_evaluated

			my_recall10 = my_recall10/number_of_actual_heads_evaluated
			my_recall5 = my_recall5/number_of_actual_heads_evaluated
			my_recall3 = my_recall3/number_of_actual_heads_evaluated
			my_recall1 = my_recall1/number_of_actual_heads_evaluated
			my_recallAll = my_recallAll/number_of_actual_heads_evaluated

			my_map10 = my_map10/number_of_actual_heads_evaluated
			my_map5 = my_map5/number_of_actual_heads_evaluated
			my_map3 = my_map3/number_of_actual_heads_evaluated
			my_map1 = my_map1/number_of_actual_heads_evaluated
			my_mapAll = my_mapAll/number_of_actual_heads_evaluated

			my_ndcg10 = my_ndcg10/number_of_actual_heads_evaluated
			my_ndcg5 = my_ndcg5/number_of_actual_heads_evaluated
			my_ndcg3 = my_ndcg3/number_of_actual_heads_evaluated
			my_ndcg1 = my_ndcg1/number_of_actual_heads_evaluated
			my_ndcgAll = my_ndcgAll/number_of_actual_heads_evaluated

			print(str(my_recall10) + "\t" + str(my_recall5) + "\t" + str(my_mapAll) + "\t" + str(my_ndcgAll))

		else:
			print("The number of actual heads evaluated is 0, meaning that the filter is too strict!")

def sort_new_batch_according_to_arity_2 (new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity):

	list_of_arities_in_pos_and_neg_facts = []
	x_by_arities = []
	y_by_arities = []

	arity2positiveFacts = {}
	for pos_fact in new_positive_facts_indexes_with_different_arity:
		arity = len(pos_fact)//2

		if arity not in list_of_arities_in_pos_and_neg_facts:
			list_of_arities_in_pos_and_neg_facts.append(arity)

		if arity not in arity2positiveFacts:
			arity2positiveFacts[arity] = []
		arity2positiveFacts[arity].append(pos_fact)

	arity2negativeFacts = {}
	for neg_fact in new_negative_facts_indexes_with_different_arity:
		arity = len(neg_fact)//2
		if arity not in list_of_arities_in_pos_and_neg_facts:
			list_of_arities_in_pos_and_neg_facts.append(arity)

		if arity not in arity2negativeFacts:
			arity2negativeFacts[arity] = []
		arity2negativeFacts[arity].append(neg_fact)

	list_of_arities_in_pos_and_neg_facts.sort()

	x_by_arities = []
	y_by_arities = []
	for arity in list_of_arities_in_pos_and_neg_facts:
		current_x_by_arities = []
		current_y_by_arities = []
		if arity in arity2positiveFacts:
			list_of_ones = np.ones(len(arity2positiveFacts[arity]))
			list_of_ones = np.reshape(list_of_ones, (len(list_of_ones), 1))
			current_x_by_arities += arity2positiveFacts[arity]
			current_y_by_arities += list(list_of_ones)

		if arity in arity2negativeFacts:
			list_of_ones = np.ones(len(arity2negativeFacts[arity])) * -1
			list_of_ones = np.reshape(list_of_ones, (len(list_of_ones), 1))
			current_x_by_arities += arity2negativeFacts[arity]
			current_y_by_arities += list(list_of_ones)

		x_by_arities.append(np.asarray(current_x_by_arities))
		y_by_arities.append(np.asarray(current_y_by_arities))

	return x_by_arities, y_by_arities

def build_posEntity2negEntities (entityId2entityTypes, entityType2entityIds, whole_train):
	posEntity2negEntities = {}

	list_of_train_entity_ids = {}
	for d in whole_train:
		for triplet in d:
			list_of_train_entity_ids[int(triplet[1])] = 1
			list_of_train_entity_ids[int(triplet[3])] = 1

	for train_entity in list_of_train_entity_ids: # get a train entity
		if train_entity in entityId2entityTypes: # if train entity has types
			train_entity_types = entityId2entityTypes[train_entity] # get all types of train entity
			for train_entity_type in train_entity_types: # for each type of train_entity
				all_entities_with_current_type = entityType2entityIds[train_entity_type] # get all entities with the same type of train_entity
				for current_entity in all_entities_with_current_type: # for each entity with the same type of train_entity
					if current_entity in list_of_train_entity_ids and current_entity != train_entity: # check that current_entity is in the train as well
						if train_entity not in posEntity2negEntities:
							posEntity2negEntities[train_entity] = {}
						if current_entity not in posEntity2negEntities[train_entity]:
							posEntity2negEntities[train_entity][current_entity] = 1
	for k in posEntity2negEntities:
		posEntity2negEntities[k] = list(posEntity2negEntities[k].keys())
	return posEntity2negEntities

def main():

	#parse input arguments
	parser = argparse.ArgumentParser(description="Model's hyperparameters")
	parser.add_argument('--indir', type=str, help='Input dir of train, test and valid data')
	parser.add_argument('--withTypes', type=str, default="True")
	parser.add_argument('--epochs', default=1000, help='Number of epochs (default: 10)' )
	parser.add_argument('--batchsize', type=int, default=128, help='Batch size (default: 128)' )
	parser.add_argument('--num_filters', type=int, default=100, help='number of filters CNN' )
	parser.add_argument('--embsize', default=100, help='Embedding size (default: 100)' )
	parser.add_argument('--learningrate', default=0.0001, help='Learning rate (default: 0.00005)' )
	parser.add_argument('--outdir', type=str, help='Output dir of model')
	parser.add_argument('--load', default='False', help='If true, it loads a saved model in dir outdir and evaluate it (default: False). If preload, it load an existing model and keep training it' )
	parser.add_argument('--modelToBeTrained', default='', help='path of the pretrained model to be loaded. It works with --load=preload' )
	parser.add_argument('--gpu_ids', default='0', help='Comma-separated gpu id used to paralellize the evaluation' )
	parser.add_argument('--num_negative_samples', type=int, default=1, help='number of negative samples for each positive sample' )
	parser.add_argument('--atLeast', type=int, help='2' ) # beta in  paper
	parser.add_argument('--topNfilters', type=int, help='2' ) # filter h_type-relation-t_type according to its frequency, keep frequent ones only
	parser.add_argument('--buildTypeDictionaries', type=str, default='False', help='True OR False' )
	parser.add_argument('--sparsifier', type=int, default=-1, help='if type frequency is less than K in ranking, set its entry to 0 in the img. If its value is <=0 then it will not sparsify the matrix' )
	parser.add_argument('--entitiesEvaluated', default='both', type=str, help='both, one, none' )
	parser.add_argument('--negative_strategy', default=0, type=float, help='0 is default strategy, 1 replacing entity with others with the same type' )
	args = parser.parse_args()
	print("\n\n************************")
	for e in vars(args):
		print (e, getattr(args, e))
	print("************************\n\n")

	# if args.load == 'True':

	with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
		data_info = pickle.load(fin)
	test = data_info['test_facts']

	relation2id = data_info['roles_indexes']
	entity2id = data_info['values_indexes']
	key_val = data_info['role_val']

	id2entity = {}
	for tmpkey in entity2id:
		id2entity[entity2id[tmpkey]] = tmpkey
	id2relation = {}
	for tmpkey in relation2id:
		id2relation[relation2id[tmpkey]] = tmpkey

	n_entities = len(entity2id)
	n_relations = len(relation2id)
	# print("Unique number of relations and head types:", n_relations)
	# print("Unique number of entities and tail types:", n_entities)

	# with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
	# 	data_info1 = pickle.load(fin)
	# whole_train = data_info["train_facts"]
	# whole_valid = data_info["valid_facts"]
	# whole_test = data_info['test_facts']

	type2id, id2type = build_type2id_v2(args.indir)
	type2id["UNK"] = len(type2id)
	id2type[len(type2id)] = "UNK"
	unk_type_id = type2id["UNK"]

	entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds = build_entity2types_dictionaries(args.indir, entity2id)

	head2relation2tails_both = build_head2relation2tails(args.indir, entity2id, relation2id, entityId2entityTypes, 'both')
	head2relation2tails_one = build_head2relation2tails(args.indir, entity2id, relation2id, entityId2entityTypes, 'one')
	head2relation2tails_none = build_head2relation2tails(args.indir, entity2id, relation2id, entityId2entityTypes, 'none')
	print("unique testing heads (both/one):", len(head2relation2tails_both), len(head2relation2tails_one))

	typeId2frequency = build_typeId2frequency (args.indir, type2id)
	headTail2hTypetType, entityId2typeIds_with_sparsifier = build_headTail2hTypetType(args.indir, entity2id, type2id, entityName2entityTypes, args.sparsifier, typeId2frequency, args.buildTypeDictionaries)

	_, test, _ = add_type_pair_to_fact ([], test, [], headTail2hTypetType, entityId2typeIds_with_sparsifier, unk_type_id)
	# whole_train, whole_test, whole_valid = add_type_pair_to_fact (whole_train, whole_test, whole_valid, headTail2hTypetType, entityId2typeIds_with_sparsifier, unk_type_id)

	device = "cuda:"+str(args.gpu_ids)

	type2relationType2frequency = build_type2relationType2frequency(args.indir, args.buildTypeDictionaries)
	type_head_tail_entity_matrix, tailType_relation_headType_tensor = build_tensor_matrix(args.indir, entity2id, relation2id, entityName2entityTypes, args.topNfilters, type2relationType2frequency, entityId2typeIds_with_sparsifier, type2id, id2type, device, args.entitiesEvaluated)

	entity2sparsifiedTypes = build_entity2sparsifiedTypes (typeId2frequency, entityId2entityTypes, type2id, args.sparsifier, unk_type_id, id2type)

	rt_negative_candidates = []



	testing_head_ids_subset = {}
	if args.load == 'True':
		# testing_head_ids_subset = {}
		# if "FB15k" in args.indir:
		# 	with open(args.indir + '/2000_random_testing_heads.txt') as input_file:
		# 		for line in input_file:
		# 			line = line.strip()
		# 			head_id = entity2id[line]
		# 			testing_head_ids_subset[head_id] = 1
		# 	input_file.close()

		epoch = args.outdir.split("/")[-1].split("_")[2]
		model = torch.load(args.outdir,map_location=device)
		t2 = TicToc()
		t2.tic()

		print("model.emb_types:", model.emb_types)

		if args.entitiesEvaluated == 'both':
			head2relation2tails = head2relation2tails_both
		if args.entitiesEvaluated == 'one':
			head2relation2tails = head2relation2tails_one
		if args.entitiesEvaluated == 'none':
			head2relation2tails = head2relation2tails_none

		evaluate_model_v2 (model, test, id2entity, type2relationType2frequency, args.topNfilters, args.atLeast, device, type2id, id2type, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, relation2id, head2relation2tails, id2relation, args.sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, entity2sparsifiedTypes, args.indir, args.entitiesEvaluated, testing_head_ids_subset, rt_negative_candidates)
		t2.toc()
		print("Evaluation last epoch ", epoch, "- running time (seconds):", t2.elapsed)


		print("END OF SCRIPT!")

		sys.stdout.flush()

	else:

		schema = tailType_relation_headType_tensor.cpu().detach().numpy().astype(int)
		# print(schema[0,0,0])
		print(schema.shape)
		# print(whole_train)

		print("Preprocessing neg sample lists, for fast training process")
		neg_list_per_head = {}
		# print(type_head_tail_entity_matrix.size(dim=1))
		candidate_size_both=0
		candidate_size_one=0
		for head_id in range(type_head_tail_entity_matrix.size(dim=1)):
			if(head_id%1000==0):
				print("progress ",head_id," over ", type_head_tail_entity_matrix.size(dim=1))
			all_head_types = type_head_tail_entity_matrix[:,head_id]
			tailType_relation_matrix = torch.matmul(tailType_relation_headType_tensor, all_head_types)
			tailType_relation_matrix = torch.transpose(tailType_relation_matrix, 0, 1)
			relation_entity_matrix = torch.matmul(tailType_relation_matrix, type_head_tail_entity_matrix)
			relation_entity_matrix[relation_entity_matrix < args.atLeast] = 0
			# print("head_id is : ",head_id, torch.nonzero(relation_entity_matrix).shape[0])
			neg_list_per_head[head_id] = []
			if torch.nonzero(relation_entity_matrix).shape[0] != 0:
				filtered_relation_tail_pairs = torch.nonzero(relation_entity_matrix).cpu().detach().numpy().astype(int)
				if head_id in head2relation2tails_one:
					candidate_size_one = candidate_size_one+len(filtered_relation_tail_pairs)/len(head2relation2tails_one)
				if head_id in head2relation2tails_both:
					candidate_size_both = candidate_size_both+len(filtered_relation_tail_pairs)/len(head2relation2tails_both)

				if len(filtered_relation_tail_pairs)>100*3000: # cache negative sample candidates, for quick batch with the expense of more RAM (we have 128GB RAM)
					rand_indices = np.random.permutation(len(filtered_relation_tail_pairs))
					rand_indices = rand_indices[:300*1000]
					sampled_pairs = filtered_relation_tail_pairs[rand_indices]
				else:
					sampled_pairs = filtered_relation_tail_pairs
				neg_list_per_head[head_id]=sampled_pairs

		print('Average candidate size both/one:', candidate_size_both, candidate_size_one)


		with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
			data_info = pickle.load(fin)
		train = data_info["train_facts"]
		valid = data_info["valid_facts"]
		test = data_info['test_facts']
		relation2id = data_info['roles_indexes']
		entity2id = data_info['values_indexes']

		key_val = data_info['role_val']

		id2entity = {}
		for tmpkey in entity2id:
			id2entity[entity2id[tmpkey]] = tmpkey
		id2relation = {}
		for tmpkey in relation2id:
			id2relation[relation2id[tmpkey]] = tmpkey

		n_entities = len(entity2id)
		n_relations = len(relation2id)
		print("Unique number of relations:", n_relations)
		print("Unique number of entities:", n_entities)

		# with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
		# 	data_info1 = pickle.load(fin)
		# whole_train = data_info1["train_facts"]
		# whole_valid = data_info1["valid_facts"]
		# whole_test = data_info1['test_facts']

		mp.set_start_method('spawn')
		t1 = TicToc()
		t2 = TicToc()

		entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds = build_entity2types_dictionaries(args.indir, entity2id)
		# print(len(entity2id))
		# print(entity2id[0])

		## img matrix
		type2id, id2type = build_type2id_v2(args.indir)
		unk_type_id = len(type2id)
		typeId2frequency = build_typeId2frequency (args.indir, type2id)
		headTail2hTypetType, entityId2typeIds_with_sparsifier = build_headTail2hTypetType(args.indir, entity2id, type2id, entityName2entityTypes, args.sparsifier, typeId2frequency, args.buildTypeDictionaries)


		train, test, valid = add_type_pair_to_fact (train, test, valid, headTail2hTypetType, entityId2typeIds_with_sparsifier, unk_type_id)
		# whole_train, whole_test, whole_valid = add_type_pair_to_fact (whole_train, whole_test, whole_valid, headTail2hTypetType, entityId2typeIds_with_sparsifier, unk_type_id)
		n_types = len(type2id) + 1 #number of types including the UNK

		# build posEntity2negEntities
		posEntity2negEntities = build_posEntity2negEntities (entityId2entityTypes, entityType2entityIds, train)
				# posEntity2negEntities = build_posEntity2negEntities (entityId2entityTypes, entityType2entityIds, whole_train)


		n_batches_per_epoch = []
		for i in train:
			ll = len(i)
			if ll == 0:
				n_batches_per_epoch.append(0)
			else:
				n_batches_per_epoch.append(int((ll - 1) / args.batchsize) + 1)

		device = "cuda:"+str(args.gpu_ids.split(",")[0])
		print("device:", device)

		if args.load == "preload":
			model = torch.load(args.modelToBeTrained, map_location=device)
			starting_epoch = int(args.modelToBeTrained.rsplit('/', 1)[-1].split("_")[7].replace("epoch", "")) + 1
			print("Model pre-loaded. The training will start at epoch", starting_epoch)
		elif args.load == "False":
			if args.withTypes == "True":
				model = RETA(len(relation2id), len(entity2id), len(type2id)+1, int(args.embsize), int(args.num_filters)).cuda()
			elif args.withTypes == "False":
				model = RETA_NO_TYPES(len(relation2id), len(entity2id), len(type2id)+1, int(args.embsize), int(args.num_filters)).cuda()
			model.init()
			starting_epoch = 1

		for name, param in model.named_parameters():
			if param.requires_grad:
				print("param:", name, param.size())

		opt = torch.optim.Adam(model.parameters(), lr=float(args.learningrate))
		# scheduler = StepLR(opt, step_size=10, gamma=0.99)

		for epoch in range(starting_epoch, int(args.epochs)+1):
			t1.tic()
			model.train()
			model.to(device)
			train_loss = 0
			rel = 0

			arity2numOfPos = {}
			arity2numOfNeg = {}

			for i in range(len(train)):

				train_i_indexes = np.array(list(train[i].keys())).astype(np.int32)
				train_i_values = np.array(list(train[i].values())).astype(np.float32)

				for batch_num in range(n_batches_per_epoch[i]):

					arity = len(train_i_indexes[0])//2

					if arity < 3:
						print("ERROR: arity < 3")

					new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity = Batch_Loader_think_slow_instance_completion(train_i_indexes, train_i_values, n_entities, n_relations, key_val, args.batchsize, arity, train[i], id2entity, id2relation, args.num_negative_samples, type2id, args.sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, id2type, entityName2entityTypes, args.negative_strategy, posEntity2negEntities, schema, entity2sparsifiedTypes, args.indir, neg_list_per_head)

					x_by_arities, y_by_arities = sort_new_batch_according_to_arity_2 (new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity)

					loss = 0
					for j in range (len(x_by_arities)):
						arity = len(x_by_arities[j][0])//2
						if arity < 3:
							print("ERROR: arity < 3")
						# print(x_by_arities[j])
						# print(y_by_arities[j])
						pred = model(x_by_arities[j], arity, "training", device, id2relation, id2entity)
						pred = pred * torch.FloatTensor(y_by_arities[j]).cuda(device) * (-1)
						# loss += model.loss(pred).mean()
						loss += model.loss(pred).sum()
					loss = loss/(args.batchsize*2)

					opt.zero_grad()
					loss.backward()
					opt.step()
					train_loss += loss.item()

			# if(epoch%50==0):
			#
			# 	evaluate_model_validation(model, test, id2entity, type2relationType2frequency, args.topNfilters, args.atLeast, device, type2id, id2type, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, relation2id, head2relation2tails_both, id2relation, args.sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, entity2sparsifiedTypes, args.indir, 'both' , testing_head_ids_subset, rt_negative_candidates)
			# 	evaluate_model_validation(model, test, id2entity, type2relationType2frequency, args.topNfilters, args.atLeast, device, type2id, id2type, type_head_tail_entity_matrix, tailType_relation_headType_tensor, entityName2entityTypes, relation2id, head2relation2tails_one, id2relation, args.sparsifier, typeId2frequency, entityId2entityTypes, unk_type_id, entity2sparsifiedTypes, args.indir, 'one' , testing_head_ids_subset, rt_negative_candidates)
			#
			# 	if args.withTypes == "True":
			# 		file_name = "RETA_batchSize" + str(args.batchsize) + "_epoch" + str(epoch) + "_embSize" + args.embsize + "_lr" + args.learningrate + "_sparsifier" + str(args.sparsifier) + "_numFilters" + str(args.num_filters)
			# 	elif args.withTypes == "False":
			# 		file_name = "RETA_with_NO_types_batchSize" + str(args.batchsize) + "_epoch" + str(epoch) + "_embSize" + args.embsize + "_lr" + args.learningrate + "_sparsifier" + str(args.sparsifier) + "_numFilters" + str(args.num_filters)
			# 	print("Saving the model trained at epoch", epoch, "in:", args.outdir + '/' + file_name)
			# 	if not os.path.exists(args.outdir):
			# 		os.makedirs(args.outdir)
			# 	torch.save(model, args.outdir + '/' + file_name)
			# 	print("Model saved")


			# scheduler.step()
			t1.toc()
			print("End of epoch", epoch, "- train_loss:", train_loss, "- training time (seconds):", t1.elapsed) # "- lr", optimizer.param_groups[0]['lr'],

			sys.stdout.flush()

		print("END OF EPOCHS")

		#SAVE THE LAST MODEL
		if args.withTypes == "True":
			file_name = "RETA_plus_batchSize" + str(args.batchsize) + "_epoch" + str(epoch) + "_embSize" + args.embsize + "_lr" + args.learningrate + "_sparsifier" + str(args.sparsifier) + "_numFilters" + str(args.num_filters)
		elif args.withTypes == "False":
			file_name = "RETA_plus_with_NO_types_batchSize" + str(args.batchsize) + "_epoch" + str(epoch) + "_embSize" + args.embsize + "_lr" + args.learningrate + "_sparsifier" + str(args.sparsifier) + "_numFilters" + str(args.num_filters)
		print("Saving the model trained at epoch", epoch, "in:", args.outdir + '/' + file_name)
		if not os.path.exists(args.outdir):
			os.makedirs(args.outdir)
		torch.save(model, args.outdir + '/' + file_name)
		print("Model saved")

		print("END OF SCRIPT!")

		sys.stdout.flush()


if __name__ == '__main__':
	main()
