import numpy as np
import random, os.path
from random import sample

import math, torch, pickle
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

def update_the_types_think_slow (current_fact, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes):

	current_fact = current_fact[:4]
	current_head = current_fact[1]
	current_tail = current_fact[3]

	if current_head in entity2sparsifiedTypes:
		current_head_types = entity2sparsifiedTypes[current_head]
	else:
		current_head_types = [unk_type_id]

	if current_tail in entity2sparsifiedTypes:
		current_tail_types = entity2sparsifiedTypes[current_tail]
	else:
		current_tail_types = [unk_type_id]

	headType_tailType_pairs = []
	for h_type_id in current_head_types:
		for t_type_id in current_tail_types:
			headType_tailType_pairs.append(h_type_id)
			headType_tailType_pairs.append(t_type_id)

	current_fact = np.append(current_fact, headType_tailType_pairs)

	return current_fact


def update_the_types (current_fact, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id):

	current_fact = current_fact[:4]
	current_head = current_fact[1]
	current_tail = current_fact[3]

	headType_tailType_pairs = add_sparsified_types_to_negative_samples (current_fact[1], current_fact[3], sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)

	current_fact = np.append(current_fact, headType_tailType_pairs)

	return current_fact
# 
# def replace_val_think_slow(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity, negative_strategy, posEntity2negEntities, schema, entity2sparsifiedTypes):
# 	"""
# 	Replace values randomly to get negative samples
# 	"""
# 	rmd_dict = key_val
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	arity = 2
#
# 	for cur_idx in range(new_range): #loop batch size times
# 		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
# 		tmp_key = new_facts_indexes[last_idx + cur_idx, key_ind]
#
# 		times = 1
# 		tmp_array = new_facts_indexes[last_idx + cur_idx]
# 		# print("org", tmp_array)
# 		# for i in range(int(len(tmp_array)/2-2)):
# 		# 	print(schema[tmp_array[i*2+5], tmp_array[0],tmp_array[i*2+4]])
#
# 		# random_negative_strategy = random.uniform(0, 1)
# 		# current_pos_entity = tmp_array[key_ind+1]
#
# 		# if current_pos_entity in posEntity2negEntities:
# 		# 	neg_dict_for_key = list(set(posEntity2negEntities[current_pos_entity]) & set(rmd_dict[tmp_key]))
# 		# 	print("neg_dict_for_key size",len(posEntity2negEntities[current_pos_entity]),len(rmd_dict[tmp_key]),len(neg_dict_for_key))
# 		# 	if len(neg_dict_for_key)==0:
# 		# 		neg_dict_for_key = rmd_dict[tmp_key]
# 		# else:
# 		# 	neg_dict_for_key = rmd_dict[tmp_key]
#
# 		neg_dict_for_key = rmd_dict[tmp_key]
# 		tmp_len = len(neg_dict_for_key)
# 		rdm_w = np.random.randint(0, tmp_len)
#
# 		tmp_array[key_ind+1] = neg_dict_for_key[rdm_w] # corrupting the fact
#
# 		tmp_array = update_the_types_think_slow (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes)
# 		flag = 0
# 		# print("neg", tmp_array)
#
# 		for i in range(int(len(tmp_array)/2-2)):
# 			if (tmp_array[i*2+5]==unk_type_id | tmp_array[i*2+4]==unk_type_id):
# 				flag=1
# 				break
# 			# print(schema[tmp_array[i*2+5], tmp_array[0],tmp_array[i*2+4]])
# 			if schema[tmp_array[i*2+5], tmp_array[0],tmp_array[i*2+4]]>0:
# 				flag=1
# 				break
#
#
# 		# print(tmp_array)
# 		while ((tuple(tmp_array) in whole_train_facts) | (flag==0) ): #
# 			# if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
# 			if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100): # too many attempts. Random choose a negative
# 				tmp_array[key_ind+1] = np.random.randint(0, n_values)
# 				tmp_array = update_the_types_think_slow (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes)
# 				flag=1
# 				# print("too many attampt, break")
# 				# break;
# 			else:
# 				rdm_w = np.random.randint(0, tmp_len)
# 				tmp_array[key_ind+1] = neg_dict_for_key[rdm_w] # corrupting the fact
# 				tmp_array = update_the_types_think_slow (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes)
# 				flag = 0
# 				for i in range(int(len(tmp_array)/2-2)):
# 					if (tmp_array[i*2+5]==unk_type_id | tmp_array[i*2+4]==unk_type_id):
# 						flag=1
# 						break
# 					if schema[tmp_array[i*2+5], tmp_array[0],tmp_array[i*2+4]]>0:
# 						flag=1
# 						break
#
# 			times = times + 1
# 			# print(times)
#
#
# 		# if(flag==0):
# 		# 	print("too many attampt, break")
#
# 		new_facts_indexes[last_idx + cur_idx, key_ind+1] = tmp_array[key_ind+1]
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array)
# 		new_facts_values[last_idx + cur_idx] = [-1]
		# print(new_facts_indexes)

# def replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity, negative_strategy, posEntity2negEntities):
# 	"""
# 	Replace values randomly to get negative samples
# 	"""
# 	rmd_dict = key_val
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	arity = 2
#
# 	for cur_idx in range(new_range): #loop batch size times
# 		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
# 		tmp_key = new_facts_indexes[last_idx + cur_idx, key_ind]
# 		tmp_len = len(rmd_dict[tmp_key])
#
# 		rdm_w = np.random.randint(0, tmp_len)
#
# 		times = 1
# 		tmp_array = new_facts_indexes[last_idx + cur_idx]
#
# 		random_negative_strategy = random.uniform(0, 1)
# 		current_pos_entity = tmp_array[key_ind+1]
# 		if random_negative_strategy < negative_strategy and current_pos_entity in posEntity2negEntities: # new strategy (replace an entity with another entity with the same type, e.g., replace obama with biden)
# 			tmp_array[key_ind+1] = random.choice(posEntity2negEntities[current_pos_entity])
# 		else: # default strategy (replace the entity with a random entity)
# 			tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w] # corrupting the fact
#
# 		tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
# 		while (tuple(tmp_array) in whole_train_facts):
# 			# if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
# 			if times > 100: # too many attempts. Random choose a negative
# 				tmp_array[key_ind+1] = np.random.randint(0, n_values)
# 			else:
# 				if random_negative_strategy < negative_strategy and current_pos_entity in posEntity2negEntities: # new strategy (replace an entity with another entity with the same type, e.g., replace obama with biden)
# 					tmp_array[key_ind+1] = random.choice(posEntity2negEntities[current_pos_entity])
# 				else: # default strategy (replace the entity with a random entity)
# 					if (tmp_len == 1) or (times > 2*tmp_len): # too many attempts. Random choose a negative
# 						tmp_array[key_ind+1] = np.random.randint(0, n_values)
# 					else: # default strategy (replace the entity with a random entity)
# 						rdm_w = np.random.randint(0, tmp_len)
# 						tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w] # corrupting the fact
# 			times = times + 1
# 			tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
#
# 		new_facts_indexes[last_idx + cur_idx, key_ind+1] = tmp_array[key_ind+1]
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array)
# 		new_facts_values[last_idx + cur_idx] = [-1]


#
# def replace_key_think_slow(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity, schema, entity2sparsifiedTypes):
# 	"""
# 	Replace keys randomly to get negative samples
# 	"""
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	rdm_ws = np.random.randint(0, n_keys, new_range)
#
# 	arity = 2
#
# 	for cur_idx in range(new_range):
# 		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
# 		tmp_array = new_facts_indexes[last_idx + cur_idx]
#
# 		if key_ind==0 or key_ind==2:
# 			tmp_array[0] = rdm_ws[cur_idx]
# 			tmp_array[2] = rdm_ws[cur_idx]
# 		else:
# 			tmp_array[key_ind] = rdm_ws[cur_idx]
#
# 		flag = 0
# 		for i in range(int(len(tmp_array)/2-2)):
# 			if schema[tmp_array[i*2+5], tmp_array[0],tmp_array[i*2+4]]>0:
# 				flag=1
# 				break
# 		times = 1
#
# 		while ((tuple(tmp_array) in whole_train_facts) | (flag==0)): #
# 			# if times > 100: # too many attempts. Random choose a negative
# 			# 	break;
#
# 			rnd_key = np.random.randint(0, n_keys)
# 			if key_ind==0 or key_ind==2:
# 				tmp_array[0] = rnd_key
# 				tmp_array[2] = rnd_key
# 			else:
# 				tmp_array[key_ind] = rnd_key
#
# 			times = times + 1
# 			flag = 0
# 			for i in range(int(len(tmp_array)/2-2)):
# 				if schema[tmp_array[i*2+5], tmp_array[0],tmp_array[i*2+4]]>0:
# 					flag=1
# 					break
# 			if times>100:
# 				flag=1
#
# 		new_facts_indexes[last_idx + cur_idx, key_ind] = tmp_array[key_ind]
# 		new_facts_values[last_idx + cur_idx] = [-1]
#
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array)

# def replace_types(n_types, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity): # randomly replace one type in each fact
# 	"""
# 	fast
# 	"""
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	rdm_ws = np.random.randint(0, n_types, new_range) #generate new_range random type ids. last_idx==batch_size
#
# 	arity = len(new_facts_indexes[0]) // 2 # arity of the current batch of facts
#
# 	for cur_idx in range(new_range): #loop batch size times
# 		type_ind = np.random.randint(4, high=arity*2, dtype=int) #generate a number between 4 and arity-1. This is the column of a type that we want to replace
#
# 		# Sample a random key
# 		tmp_array = new_facts_indexes[last_idx + cur_idx] #get a fact in position last_idx+cur_idx. In the first half of new_facts_indexes there are positive facts, and in the second half negative facts (that's why the row number is always last_idx (which is == batch size) + another index)
#
# 		tmp_array[type_ind] = rdm_ws[cur_idx]
#
# 		while (tuple(tmp_array) in whole_train_facts): # check if the corrupted fact exists in the whole_train_facts (which contains also the permute facts)
#
# 			rnd_type = np.random.randint(0, n_types)
# 			tmp_array[type_ind] = rnd_type
#
# 		new_facts_indexes[last_idx + cur_idx, type_ind] = tmp_array[type_ind] #store the corrupted key in new_facts_indexes (row:last_idx+cur_idx, column:type_ind) # THIS IS LIKE IN THE ORIGINAL IMPLEMENTATION WHERE WE ONLY CHANGE ONE VALUE OF THE FACT.
# 		new_facts_values[last_idx + cur_idx] = [-1] #set -1 because we have generated a false fact
#
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array) #store the corrupted fact in new_facts_indexes (row:last_idx+cur_idx). IN THIS NEW IMPLEMENTATION WE REPLACE THE WHOLE FACT WITH THE CORRUPTED FACT (AND NOT ONLY THE CURREPTED ELEMENT LIKE IN THE ORGINAL IMPLEMENTATION)
#
# def replace_types_2(n_types, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity): # randomly replace all types in each fact
# 	"""
# 	slow
# 	"""
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	rdm_ws = np.random.randint(0, n_types, new_range) #generate new_range random type ids. last_idx==batch_size
#
# 	arity = len(new_facts_indexes[0]) // 2 # arity of the current batch of facts
#
# 	for cur_idx in range(new_range): #loop batch size times
# 		type_ind = np.random.randint(4, high=arity*2, dtype=int) #generate a number between 4 and arity-1. This is the column of a type that we want to replace
#
# 		# Sample a random key
# 		tmp_array = new_facts_indexes[last_idx + cur_idx] #get a fact in position last_idx+cur_idx. In the first half of new_facts_indexes there are positive facts, and in the second half negative facts (that's why the row number is always last_idx (which is == batch size) + another index)
#
# 		# tmp_array[type_ind] = rdm_ws[cur_idx]
#
# 		# r h r t, h1 t1 h1 t2 h1 t3, h2 t1 h2 t2 h2 t3, h3 t1 h3 t2 h3 t3
# 		if cur_idx%2==0: # head type
# 			type_indexes = list(range(4, arity*2, 2))
# 		else:
# 			type_indexes = list(range(5, arity*2, 2))
# 		new_type = rdm_ws[cur_idx]
# 		old_type = tmp_array[type_ind]
# 		for type_idx in type_indexes:
# 			if tmp_array[type_idx] == old_type:
# 				tmp_array[type_idx] = new_type
#
# 		while (tuple(tmp_array) in whole_train_facts): # check if the corrupted fact exists in the whole_train_facts (which contains also the permute facts)
#
# 			# rnd_type = np.random.randint(0, n_types)
# 			# tmp_array[type_ind] = rnd_type
# 			old_type = new_type # now the old_type is new_type (because it has been modified before the while)
# 			new_type = np.random.randint(0, n_types)
# 			for type_idx in type_indexes:
# 				if tmp_array[type_idx] == old_type:
# 					tmp_array[type_idx] = new_type
#
# 		new_facts_indexes[last_idx + cur_idx, type_ind] = tmp_array[type_ind] #store the corrupted key in new_facts_indexes (row:last_idx+cur_idx, column:type_ind) # THIS IS LIKE IN THE ORIGINAL IMPLEMENTATION WHERE WE ONLY CHANGE ONE VALUE OF THE FACT.
# 		new_facts_values[last_idx + cur_idx] = [-1] #set -1 because we have generated a false fact
#
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array) #store the corrupted fact in new_facts_indexes (row:last_idx+cur_idx). IN THIS NEW IMPLEMENTATION WE REPLACE THE WHOLE FACT WITH THE CORRUPTED FACT (AND NOT ONLY THE CURREPTED ELEMENT LIKE IN THE ORGINAL IMPLEMENTATION)


#
# def Batch_Loader_think_slow(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, num_negative_samples, type2id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, id2type, entityName2entityTypes, negative_strategy, posEntity2negEntities, schema, entity2sparsifiedTypes):
#
# 	new_positive_facts_indexes_with_different_arity = []
# 	new_negative_facts_indexes_with_different_arity = []
#
# 	new_facts_indexes = np.empty((batch_size+(batch_size*num_negative_samples), 2*arity)).astype(np.int32)
# 	new_facts_values = np.empty((batch_size+(batch_size*num_negative_samples), 1)).astype(np.float32)
#
# 	idxs = np.random.randint(0, len(train_i_values), batch_size)
# 	# idxs = range(5)
#
# 	new_facts_indexes[:batch_size, :] = train_i_indexes[idxs, :]
#
# 	for positive_fact in train_i_indexes[idxs, :]:
# 		new_positive_facts_indexes_with_different_arity.append(positive_fact)
#
# 	new_facts_values[:batch_size] = train_i_values[idxs, :]
# 	last_idx = batch_size
#
# 	new_facts_indexes[last_idx:last_idx+(last_idx*num_negative_samples), :] = np.tile(new_facts_indexes[:last_idx, :], (num_negative_samples, 1))
#
# 	new_facts_values[last_idx:last_idx+(last_idx*num_negative_samples)] = np.tile(new_facts_values[:last_idx], (num_negative_samples, 1))
#
# 	val_key = random.uniform(0, 1)
# 	if val_key < 0.5:
# 		replace_val_think_slow(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity, negative_strategy, posEntity2negEntities, schema, entity2sparsifiedTypes)
# 	else:
# 		replace_key_think_slow(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity, schema, entity2sparsifiedTypes)
#
# 	last_idx += batch_size
#
# 	return new_facts_indexes, new_facts_values, new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity



def Batch_Loader_think_slow_instance_completion(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, num_negative_samples, type2id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, id2type, entityName2entityTypes, negative_strategy, posEntity2negEntities, schema, entity2sparsifiedTypes, indir, neg_list_per_head):

	new_positive_facts_indexes_with_different_arity = []
	new_negative_facts_indexes_with_different_arity = []

	idxs = np.random.randint(0, len(train_i_values), batch_size)
	# idxs = range(5)
	for positive_fact in train_i_indexes[idxs, :]:
		new_positive_facts_indexes_with_different_arity.append(positive_fact)
		# print("positive_fact", positive_fact)

	# print(type(new_positive_facts_indexes_with_different_arity))
	# tmp_new_positive_facts_indexes_with_different_arity = new_positive_facts_indexes_with_different_arity.copy()


	for cur_idx in range(batch_size*num_negative_samples):
		tmp_array = new_positive_facts_indexes_with_different_arity[cur_idx].copy()
		# print(type(tmp_array))
		# print("pos1111111111:", new_positive_facts_indexes_with_different_arity[cur_idx])
		head_id = tmp_array[1]
		neg_list = neg_list_per_head[head_id]
		if len(neg_list)>0:
			random_index = np.random.randint(0, len(neg_list),1)
			# print("neg_list: ",neg_list)
			neg = neg_list[random_index][0]
			# print("tmp_array, neg before: ",tmp_array, neg)
			tmp_array[0] = neg[0]
			tmp_array[2] = neg[0]
			tmp_array[3] = neg[1] # corrupting the fact
			# print("tmp_array, neg after: ",tmp_array, neg)
		else:
			random_key = np.random.randint(0, n_keys, 1)
			random_value = np.random.randint(0, n_values, 1)
			tmp_array[0] = random_key
			tmp_array[2] = random_key
			tmp_array[3] = random_value # corrupting the fact
		# random_key = np.random.randint(0, n_keys, 1)
		# random_value = np.random.randint(0, n_values, 1)
		# tmp_array[0] = random_key
		# tmp_array[2] = random_key
		# tmp_array[3] = random_value # corrupting the fact
		# print("pos2:", new_positive_facts_indexes_with_different_arity[cur_idx])
		tmp_array = update_the_types_think_slow (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes)
		# print("neg",tmp_array)
		new_negative_facts_indexes_with_different_arity.append(tmp_array)
	# 	print("pos3:", new_positive_facts_indexes_with_different_arity[cur_idx])
	# print("new_positive_facts_indexes_with_different_arity", new_positive_facts_indexes_with_different_arity)
	# print("new_negative_facts_indexes_with_different_arity", new_negative_facts_indexes_with_different_arity)

		#
		# with open(path, 'rb') as inp:
		# 	tmp = pickle.load(inp)
		# 	new_tiled_fact = tmp[0]
		# 	if len(new_tiled_fact)==0:
		# 		continue
		# 	# print("new_tiled_fact length",len(new_tiled_fact))
		# 	# print("new_tiled_fact",new_tiled_fact)
		# 	rdm_idx = np.random.randint(0, len(new_tiled_fact))
		# 	neg_list = new_tiled_fact[rdm_idx] # corrupting the fact
		# 	# print("neg_list length",len(neg_list))
		# 	rdm_idx = np.random.randint(0, len(neg_list))
		# 	tmp_array[[0,2,3]] = neg_list[rdm_idx][[0,2,3]]
		# 	# print("neg:", tmp_array)
		# 	tmp_array = update_the_types_think_slow (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes)
		# 	times = 1
		# 	while (tuple(tmp_array) in whole_train_facts):
		# 		rdm_idx = np.random.randint(0, len(neg_list))
		# 		tmp_array[[0,2,3]] = neg_list[rdm_idx][[0,2,3]] # corrupting the fact
		# 		tmp_array = update_the_types_think_slow (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, entity2sparsifiedTypes)
		# 		times = times + 1
		# 		if times>100:
		# 			# print("too many attampts")
		# 			break

			# new_negative_facts_indexes_with_different_arity.append(tmp_array)

	return new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity



# def Batch_Loader_think_slow(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, num_negative_samples, type2id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, id2type, entityName2entityTypes, negative_strategy, posEntity2negEntities, type_head_tail_entity_matrix, tailType_relation_headType_tensor, rt_negative_candidates):
#
# 	new_positive_facts_indexes_with_different_arity = []
# 	new_negative_facts_indexes_with_different_arity = []
#
# 	new_facts_indexes = np.empty((batch_size+(batch_size*num_negative_samples), 2*arity)).astype(np.int32)
# 	new_facts_values = np.empty((batch_size+(batch_size*num_negative_samples), 1)).astype(np.float32)
#
# 	idxs = np.random.randint(0, len(train_i_values), batch_size)
# 	# idxs = range(5)
#
# 	new_facts_indexes[:batch_size, :] = train_i_indexes[idxs, :]
#
# 	for positive_fact in train_i_indexes[idxs, :]:
# 		new_positive_facts_indexes_with_different_arity.append(positive_fact)
#
# 	new_facts_values[:batch_size] = train_i_values[idxs, :]
# 	last_idx = batch_size
#
# 	new_facts_indexes[last_idx:last_idx+(last_idx*num_negative_samples), :] = np.tile(new_facts_indexes[:last_idx, :], (num_negative_samples, 1))
#
# 	new_facts_values[last_idx:last_idx+(last_idx*num_negative_samples)] = np.tile(new_facts_values[:last_idx], (num_negative_samples, 1))
# 	#
# 	# print(new_facts_indexes)
# 	# print(new_facts_values)
# 	# replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity, negative_strategy, posEntity2negEntities)
#
#
#
# 	rmd_dict = key_val
# 	new_range = (last_idx*num_negative_samples)
#
# 	arity = 2
#
# 	for cur_idx in range(new_range): #loop batch size times
# 		tmp_array = new_facts_indexes[last_idx + cur_idx]
# 		# print(tmp_array)
#
# 		head_id = tmp_array[1]
# 		# all_head_types = type_head_tail_entity_matrix[:,head_id]
# 		# print(type_head_tail_entity_matrix.shape)
# 		# print(type_head_tail_entity_matrix)
# 		# tailType_relation_matrix = torch.matmul(tailType_relation_headType_tensor, all_head_types)
# 		# tailType_relation_matrix = torch.transpose(tailType_relation_matrix, 0, 1)
# 		# relation_entity_matrix = torch.matmul(tailType_relation_matrix, type_head_tail_entity_matrix)
# 		# relation_entity_matrix[relation_entity_matrix < atLeast] = 0
# 		filtered_relation_tail_pairs = rt_negative_candidates[head_id]
#
# 		if filtered_relation_tail_pairs.shape[0]!=0:
# 			# print(filtered_relation_tail_pairs.shape[0])
# 			rnd_index = np.random.randint(0, filtered_relation_tail_pairs.shape[0])
# 			# print(rnd_index)
# 			neg_rt = filtered_relation_tail_pairs[rnd_index,:]
# 			# print(neg_rt)
# 			tmp_array[0] = neg_rt[0]
# 			tmp_array[2] = neg_rt[0]
# 			tmp_array[3] = neg_rt[1]
# 		else:
# 			tmp_len = len(rmd_dict[tmp_array[0]])
# 			rdm_w = np.random.randint(0, tmp_len)
# 			tmp_array[1] = rmd_dict[tmp_array[0]][rdm_w]
#
# 		times = 1
# 		tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
#
# 		while (tuple(tmp_array) in whole_train_facts):
# 			if filtered_relation_tail_pairs.shape[0]!=0:
# 				# print(filtered_relation_tail_pairs.shape[0])
# 				rnd_index = np.random.randint(0, filtered_relation_tail_pairs.shape[0])
# 				# print(rnd_index)
# 				neg_rt = filtered_relation_tail_pairs[rnd_index,:]
# 				# print(neg_rt)
# 				tmp_array[0] = neg_rt[0]
# 				tmp_array[2] = neg_rt[0]
# 				tmp_array[3] = neg_rt[1]
# 			else:
# 				if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
# 					tmp_array[1] = np.random.randint(0, n_values)
#
# 				tmp_len = len(rmd_dict[tmp_array[0]])
# 				rdm_w = np.random.randint(0, tmp_len)
# 				tmp_array[1] = rmd_dict[tmp_array[0]][rdm_w]
#
# 			times = times + 1
# 			tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
#
#
# 		new_facts_indexes[last_idx + cur_idx, :] = tmp_array
# 		new_facts_values[last_idx + cur_idx] = -1
#
#
# 	last_idx += batch_size
#
# 	return new_facts_indexes, new_facts_values, new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity



# def replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity):
# 	"""
# 	Replace values randomly to get negative samples
# 	"""
# 	rmd_dict = key_val
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	arity = 2
#
# 	for cur_idx in range(new_range): #loop batch size times
# 		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
# 		tmp_key = new_facts_indexes[last_idx + cur_idx, key_ind]
# 		tmp_len = len(rmd_dict[tmp_key])
# 		rdm_w = np.random.randint(0, tmp_len)
#
# 		times = 1
# 		tmp_array = new_facts_indexes[last_idx + cur_idx]
# 		tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
# 		tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
# 		while (tuple(tmp_array) in whole_train_facts):
# 			if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
# 				tmp_array[key_ind+1] = np.random.randint(0, n_values)
# 				print("too many attampt, break")
# 			else:
# 				rdm_w = np.random.randint(0, tmp_len)
# 				tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
# 			times = times + 1
# 			tmp_array = update_the_types (tmp_array, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id)
#
# 		new_facts_indexes[last_idx + cur_idx, key_ind+1] = tmp_array[key_ind+1]
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array)
# 		new_facts_values[last_idx + cur_idx] = [-1]
#
# def replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity):
# 	"""
# 	Replace keys randomly to get negative samples
# 	"""
#
# 	new_range = (last_idx*num_negative_samples)
#
# 	rdm_ws = np.random.randint(0, n_keys, new_range)
#
# 	arity = 2
#
# 	for cur_idx in range(new_range):
# 		key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
# 		tmp_array = new_facts_indexes[last_idx + cur_idx]
#
# 		if key_ind==0 or key_ind==2:
# 			tmp_array[0] = rdm_ws[cur_idx]
# 			tmp_array[2] = rdm_ws[cur_idx]
# 		else:
# 			tmp_array[key_ind] = rdm_ws[cur_idx]
#
# 		while (tuple(tmp_array) in whole_train_facts):
#
# 			rnd_key = np.random.randint(0, n_keys)
# 			if key_ind==0 or key_ind==2:
# 				tmp_array[0] = rnd_key
# 				tmp_array[2] = rnd_key
# 			else:
# 				tmp_array[key_ind] = rnd_key
#
# 		new_facts_indexes[last_idx + cur_idx, key_ind] = tmp_array[key_ind]
# 		new_facts_values[last_idx + cur_idx] = [-1]
#
# 		new_negative_facts_indexes_with_different_arity.append(tmp_array)
#
# def Batch_Loader(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, num_negative_samples, type2id, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, id2type, entityName2entityTypes):
#
# 	new_positive_facts_indexes_with_different_arity = []
# 	new_negative_facts_indexes_with_different_arity = []
#
# 	new_facts_indexes = np.empty((batch_size+(batch_size*num_negative_samples), 2*arity)).astype(np.int32)
# 	new_facts_values = np.empty((batch_size+(batch_size*num_negative_samples), 1)).astype(np.float32)
#
# 	idxs = np.random.randint(0, len(train_i_values), batch_size)
#
# 	new_facts_indexes[:batch_size, :] = train_i_indexes[idxs, :]
#
# 	for positive_fact in train_i_indexes[idxs, :]:
# 		new_positive_facts_indexes_with_different_arity.append(positive_fact)
#
# 	new_facts_values[:batch_size] = train_i_values[idxs, :]
# 	last_idx = batch_size
#
# 	new_facts_indexes[last_idx:last_idx+(last_idx*num_negative_samples), :] = np.tile(new_facts_indexes[:last_idx, :], (num_negative_samples, 1))
#
# 	new_facts_values[last_idx:last_idx+(last_idx*num_negative_samples)] = np.tile(new_facts_values[:last_idx], (num_negative_samples, 1))
#
# 	val_key = random.uniform(0, 1)
# 	if val_key < 0.5:
# 		replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, sparsifier, typeId2frequency, entityId2entityTypes, id2entity, unk_type_id, type2id, id2type, entityName2entityTypes, new_negative_facts_indexes_with_different_arity)
# 	else:
# 		replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples, new_negative_facts_indexes_with_different_arity)
#
# 	last_idx += batch_size
#
# 	return new_facts_indexes, new_facts_values, new_positive_facts_indexes_with_different_arity, new_negative_facts_indexes_with_different_arity
