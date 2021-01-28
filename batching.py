import numpy as np
import random
import math, torch
from itertools import repeat

def add_sparsified_types_to_negative_samples (head_entity_id, tail_entity_id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id):
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
		sorted_headType2freq = sorted(headType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier] # get the N most frequent head types
		current_head_types = [type2id[item[0]] for item in sorted_headType2freq] # get top head type ids

	# get top tail types
	if tail_entity_id in entityId2entityTypes:
		current_tail_types = entityId2entityTypes[tail_entity_id]
		tailType2freq = {}
		for t_type in current_tail_types:
			t_type_id = type2id[t_type]
			if t_type_id in typeId2frequency:
				tailType2freq[t_type] = typeId2frequency[t_type_id]
		sorted_tailType2freq = sorted(tailType2freq.items(), key=lambda kv: kv[1], reverse=True)[:sparsifier] # get the N most frequent tail types
		current_tail_types = [type2id[item[0]] for item in sorted_tailType2freq] # get top tail type ids

	if len(current_head_types)==0:
		current_head_types = [unk_type_id]

	if len(current_tail_types)==0:
		current_tail_types = [unk_type_id]

	headType_tailType_pairs = []
	for h_type_id in current_head_types:
		for t_type_id in current_tail_types:
			headType_tailType_pairs.append(h_type_id)
			headType_tailType_pairs.append(t_type_id)

	return headType_tailType_pairs

def update_the_types (current_fact, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id):

	current_fact = current_fact[:4]
	current_head = current_fact[1]
	current_tail = current_fact[3]

	headType_tailType_pairs = add_sparsified_types_to_negative_samples (current_fact[1], current_fact[3], sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)

	current_fact = np.append(current_fact, headType_tailType_pairs)

	return current_fact

def replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity):
	"""
	Replace values randomly to get negative samples
	"""
	rmd_dict = key_val

	new_range = (last_idx*num_negative_samples)

	arity = 2

	for cur_idx in range(new_range): #loop batch size times
		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
		tmp_key = new_facts_indexes[last_idx + cur_idx, key_ind]
		tmp_len = len(rmd_dict[tmp_key])
		rdm_w = np.random.randint(0, tmp_len)

		times = 1
		tmp_array = new_facts_indexes[last_idx + cur_idx]
		tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
		tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
		while (tuple(tmp_array) in whole_train_facts):
			if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
				tmp_array[key_ind+1] = np.random.randint(0, n_values)
			else:
				rdm_w = np.random.randint(0, tmp_len)
				tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
			times = times + 1
			tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)

		new_facts_indexes[last_idx + cur_idx, key_ind+1] = tmp_array[key_ind+1]
		new_negative_facts_indexes_with_different_arity.append(tmp_array)
		new_facts_values[last_idx + cur_idx] = [-1]

def replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity):
	"""
	Replace keys randomly to get negative samples
	"""

	new_range = (last_idx*num_negative_samples)

	rdm_ws = np.random.randint(0, n_keys, new_range)

	arity = 2

	for cur_idx in range(new_range):
		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
		tmp_array = new_facts_indexes[last_idx + cur_idx]

		if key_ind==0 or key_ind==2:
			tmp_array[0] = rdm_ws[cur_idx]
			tmp_array[2] = rdm_ws[cur_idx]
		else:
			tmp_array[key_ind] = rdm_ws[cur_idx]

		while (tuple(tmp_array) in whole_train_facts):

			rnd_key = np.random.randint(0, n_keys)
			if key_ind==0 or key_ind==2:
				tmp_array[0] = rnd_key
				tmp_array[2] = rnd_key
			else:
				tmp_array[key_ind] = rnd_key

		new_facts_indexes[last_idx + cur_idx, key_ind] = tmp_array[key_ind]
		new_facts_values[last_idx + cur_idx] = [-1]

		new_negative_facts_indexes_with_different_arity.append(tmp_array)

def Batch_Loader(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, num_negative_samples, type2id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, id2type, entityName2entityTypes):

	new_positive_facts_indexes_with_different_arity = []
	new_negative_facts_indexes_with_different_arity = []

	new_facts_indexes = np.empty((batch_size+(batch_size*num_negative_samples), 2*arity)).astype(np.int32)
	new_facts_values = np.empty((batch_size+(batch_size*num_negative_samples), 1)).astype(np.float32)

	idxs = np.random.randint(0, len(train_i_values), batch_size)

	new_facts_indexes[:batch_size, :] = train_i_indexes[idxs, :]

	for positive_fact in train_i_indexes[idxs, :]:
		new_positive_facts_indexes_with_different_arity.append(positive_fact)

	new_facts_values[:batch_size] = train_i_values[idxs, :]
	last_idx = batch_size

	new_facts_indexes[last_idx:last_idx+(last_idx*num_negative_samples), :] = np.tile(new_facts_indexes[:last_idx, :], (num_negative_samples, 1))

	new_facts_values[last_idx:last_idx+(last_idx*num_negative_samples)] = np.tile(new_facts_values[:last_idx], (num_negative_samples, 1))

	val_key = random.uniform(0, 1)
	if val_key < 0.5:
		replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity)
	else:
		replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity)

	last_idx += batch_size

	return new_facts_indexes, new_facts_values, new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity
