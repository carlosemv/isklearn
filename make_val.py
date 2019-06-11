import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run_output', type=str,
	help="Run output that best configs are extracted from")
parser.add_argument('validation', type=str,
	help="Python file containing validation function")
parser.add_argument('val_script', type=str,
	help="Name of slurm script to be generated")
args = parser.parse_known_args()[0]

def get_configs(source):
	with open(source, 'r') as src:
		for line in src.readlines():
			if "--dataset" in line:
				yield line

header = (
	"#!/bin/bash",
	"#SBATCH --time=0-00:00",
	"#SBATCH --job-name=val_isklearn",
	"#SBATCH --cpus-per-task=32",
	"#SBATCH --output=mnist_val_%j.out"
)

validation = Path(args.validation).resolve()

with open(args.val_script, 'w') as out:
	print(*header, sep='\n', file=out)
	for config in get_configs(args.run_output):
		run = "python {} {}".format(
			validation, config)
		out.write(run)