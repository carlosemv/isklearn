import itertools
import pandas as pd
from random import sample, shuffle

def triple_instances(folds, samples=500):
	instances = list(itertools.combinations(range(folds), 3))
	if samples and samples < len(instances):
		instances = sample(instances, samples)
	return instances

def single_instances(folds):
	instances = list(range(folds))
	shuffle(instances)
	return instances


instances = triple_instances(20)
df = pd.DataFrame(instances)
df.to_csv("new-instances.txt", header=False, index=False)