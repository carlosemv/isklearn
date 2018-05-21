import pandas as pd
from random import sample

num_instances = 10000
instances = []

for i in range(num_instances):
	instances.append(sample(range(1,10), 3))

df = pd.DataFrame(instances)
df.to_csv("instances.txt", header=False, index=False)