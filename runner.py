#!/usr/bin/env python3
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os
import os.path
import re
import subprocess
import sys

assert sys.version_info >= (3, 6), "Python >= 3.6 required"

def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

if len(sys.argv) < 7:
    print("Insufficient arguments")
    sys.exit(1)

# Get the parameters as command line arguments.
candidate_id = sys.argv[1]
instance_id = sys.argv[2]
seed = sys.argv[3]
instance = sys.argv[4]
dataset = sys.argv[6]
cand_params = sys.argv[7:]

# Define the stdout and stderr files.
output_prefix = "output/" + dataset
os.makedirs(output_prefix)
out_file = output_prefix + "/c" + str(candidate_id) + "-" + str(instance_id) + ".stdout"
err_file = output_prefix + "/c" + str(candidate_id) + "-" + str(instance_id) + ".stderr"

exe = dataset + "/target.py"
cutoff = 30*60

if not os.path.isfile(exe):
    target_runner_error(str(exe) + " not found")
if not os.access(exe, os.X_OK):
    target_runner_error(str(exe) + " is not executable")

command = [exe] + ["--instance"] + [instance] + \
    ["--config_id"] + [candidate_id] + cand_params

outf = open(out_file, "w")
errf = open(err_file, "w")
try:
    completed = subprocess.run(command, stdout=outf, stderr=errf, timeout=cutoff)
except subprocess.TimeoutExpired:
    outf.close()
    errf.close()

    # cutoff_log = "output/cutoff_" + dataset + ".out"
    # with open(cutoff_log, "a") as log:
    #     log.write(" ".join(command)+'\n')

    print(2**32, cutoff)
else:
    outf.close()
    errf.close()

    if completed.returncode != 0:
        target_runner_error("command returned code " + str(completed.returncode))

    if not os.path.isfile(out_file):
        target_runner_error("output file "+ out_file +" not found.")

    lastline = [line.rstrip('\n') for line in open(out_file)][-1]
    print(lastline)

os.remove(out_file)
os.remove(err_file)

sys.exit(0)
