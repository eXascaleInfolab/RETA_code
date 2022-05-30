# RETA/RETA++

RETA (as it suggests the Relation and Tail for a given head entity) is an end-to-end solution tackling instance completion problem over KGs. It consists of two components: a RETA-Filter and RETA-Grader. More precisely, our RETA-Filter first generates candidate relation-tail pairs for a given head by extracting and leveraging the schema of a KG; our RETA-Grader then evaluates and ranks the candidate relation-tail pairs considering the plausibility of both the candidate triplet and its corresponding schema using a newly-designed KG embedding model.

RETA++ is an the extension of RETA. It systematically integrates the two components by training RETA-Grader on the reduced solution space output by RETA-Filter via a customized negative sampling process, so as to fully benefit from the efficiency of RETA-Filter in solution space reduction and the deliberation of RETA-Grader in scoring candidate triplets.
​
## How to run RETA
###### Datasets link: http://bit.ly/3t2WFTE

###### Train and Evaluate JF17k (use the pre-processed dataset: http://bit.ly/3tksFCK)
```
python main.py --indir data/JF17k --withTypes True --epochs 3000 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k --gpu_ids 0 --num_negative_samples 1 --buildTypeDictionaries True --sparsifier 2

python main.py --indir data/JF17k --withTypes True --epochs 3000 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k/RETA_batchSize128_epoch3000_embSize100_lr0.0002_sparsifier1_numFilters50 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2 --entitiesEvaluated both

python main.py --indir data/JF17k --withTypes True --epochs 3000 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k/RETA_batchSize128_epoch3000_embSize100_lr0.0002_sparsifier1_numFilters50 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 1 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2 --entitiesEvaluated one

python main.py --indir data/JF17k --withTypes True --epochs 3000 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k/RETA_batchSize128_epoch3000_embSize100_lr0.0002_sparsifier1_numFilters50 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2 --entitiesEvaluated none
```

###### Train and Evaluate FB15k (use the pre-processed dataset: http://bit.ly/3cHacuq)
```
python main.py --indir data/FB15k --withTypes True --epochs 200 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k --gpu_ids 7 --num_negative_samples 1 --buildTypeDictionaries True --sparsifier 1

python main.py --indir data/FB15k --withTypes True --epochs 200 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k/RETA_batchSize128_epoch200_embSize100_lr0.0002_sparsifier1_numFilters200 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 --entitiesEvaluated both

python main.py --indir data/FB15k --withTypes True --epochs 200 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k/RETA_batchSize128_epoch200_embSize100_lr0.0002_sparsifier1_numFilters200 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 1 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 --entitiesEvaluated one

python main.py --indir data/FB15k --withTypes True --epochs 200 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k/RETA_batchSize128_epoch200_embSize100_lr0.0002_sparsifier1_numFilters200 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 --entitiesEvaluated none
```

###### Train and Evaluate humans_wikidata (use the pre-processed dataset: http://bit.ly/2M1S1ER)

```
python main.py --indir data/humans_wikidata --withTypes True --epochs 3000 --batchsize 128 --num_filters 100 --embsize 100 --learningrate 0.0002 --outdir data/humans_wikidata --gpu_ids 2 --num_negative_samples 1 --buildTypeDictionaries True --sparsifier 4

python main.py --indir data/humans_wikidata --withTypes True --epochs 3000 --batchsize 128 --num_filters 100 --embsize 100 --learningrate 0.0002 --outdir data/humans_wikidata/RETA_batchSize128_epoch3000_embSize100_lr0.0002_sparsifier4_numFilters100 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 1 --topNfilters -80 --buildTypeDictionaries True --sparsifier 4 --entitiesEvaluated both

python main.py --indir data/humans_wikidata --withTypes True --epochs 3000 --batchsize 128 --num_filters 100 --embsize 100 --learningrate 0.0002 --outdir data/humans_wikidata/RETA_batchSize128_epoch3000_embSize100_lr0.0002_sparsifier4_numFilters100 --load True --gpu_ids 3 --num_negative_samples 1 --atLeast 1 --topNfilters -80 --buildTypeDictionaries True --sparsifier 4 --entitiesEvaluated one
```

###### Parameter setting:
In `main.py`, you can set:

`--indir`: input file directory

`--withTypes`: True trains RETA, False trains RETA no type

`--epochs`: number of training epochs

`--batchsize`: batch size of training set

`--num_filters`: number of filters used in the CNN

`--embsize`: embedding size

`--load`: load a pre-trained RETA model and evaluate

`--learningrate`: learning rate

`--outdir`: where to store RETA model

`--num_negative_samples`: number of negative samples

`--gpu_ids`: gpu to be used for train and test the model

`--atLeast`: beta parameter

`--topNfilters`: alpha parameter

`--buildTypeDictionaries`: store dictionaries to speed-up the code

`--sparsifier`: top-k types for each entity

`--entitiesEvaluated`: both (evaluate facts where both h and t have types), one (evaluate facts where either h or t have types), none (evaluate facts where both h and t don't have types). In our paper we have computed the weighted average between these three settings.

## How to run RETA++
###### Train and Evaluate JF17k (use the pre-processed dataset: http://bit.ly/3tksFCK)
```
python main_reta_plus.py --indir data/JF17k --withTypes True --epochs 2300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k  --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2

python main_reta_plus.py --indir data/JF17k --withTypes True --epochs 2300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k/RETA_plus_batchSize128_epoch2300_embSize100_lr0.0002_sparsifier1_numFilters50  --load True  --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2 --entitiesEvaluated both 

python main_reta_plus.py --indir data/JF17k --withTypes True --epochs 2300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k/RETA_plus_batchSize128_epoch2300_embSize100_lr0.0002_sparsifier1_numFilters50  --load True  --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2 --entitiesEvaluated one 

python main_reta_plus.py --indir data/JF17k --withTypes True --epochs 2300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/JF17k/RETA_plus_batchSize128_epoch2300_embSize100_lr0.0002_sparsifier1_numFilters50  --load True  --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 2 --entitiesEvaluated none 
```


###### Train and Evaluate FB15k (use the pre-processed dataset: http://bit.ly/3cHacuq)
```
python main_reta_plus.py --indir data/FB15k --withTypes True --epochs 300 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 

python main_reta_plus.py --indir data/FB15k --withTypes True --epochs 300 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k/RETA_plus_batchSize128_epoch300_embSize100_lr0.0002_sparsifier1_numFilters200  --load True --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 --entitiesEvaluated both

python main_reta_plus.py --indir data/FB15k --withTypes True --epochs 300 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k/RETA_plus_batchSize128_epoch300_embSize100_lr0.0002_sparsifier1_numFilters200  --load True --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 --entitiesEvaluated one

python main_reta_plus.py --indir data/FB15k --withTypes True --epochs 300 --batchsize 128 --num_filters 200 --embsize 100 --learningrate 0.0002 --outdir data/FB15k/RETA_plus_batchSize128_epoch300_embSize100_lr0.0002_sparsifier1_numFilters200  --load True --num_negative_samples 1 --atLeast 2 --topNfilters -10 --buildTypeDictionaries True --sparsifier 1 --entitiesEvaluated none
```

###### Train and Evaluate humans_wikidata (use the pre-processed dataset: http://bit.ly/2M1S1ER)

```
python main_reta_plus.py --indir data/humans_wikidata --withTypes True --epochs 1300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/humans_wikidata/ --num_negative_samples 1 --atLeast 1 --topNfilters -80 --buildTypeDictionaries True --sparsifier 4 

python main_reta_plus.py --indir data/humans_wikidata --withTypes True --epochs 1300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/humans_wikidata/RETA_plus_batchSize128_epoch1300_embSize100_lr0.0002_sparsifier4_numFilters50  --load True --num_negative_samples 1 --atLeast 1 --topNfilters -80 --buildTypeDictionaries True --sparsifier 4 --entitiesEvaluated both

python main_reta_plus.py --indir data/humans_wikidata --withTypes True --epochs 1300 --batchsize 128 --num_filters 50 --embsize 100 --learningrate 0.0002 --outdir data/humans_wikidata/RETA_plus_batchSize128_epoch1300_embSize100_lr0.0002_sparsifier4_numFilters50  --load True --num_negative_samples 1 --atLeast 1 --topNfilters -80 --buildTypeDictionaries True --sparsifier 4 --entitiesEvaluated one
```


## Data preprocessing example
```
python build_dataset.py --indir FB15k --atLeast 1000
python builddata.py --data_dir data/FB15k/
```

# Reference
If you use our code or datasets, please cite:
```
Rosso, P., Yang, D., Ostapuk, N., Cudré-Mauroux, P. (2021, April). RETA: A Schema-Aware, End-to-End Solution for Instance Completion in Knowledge Graphs. In Proceedings of The Web Conference 2021.
```
