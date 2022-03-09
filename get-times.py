import subprocess
from datetime import timedelta
import re

n_jobs = 10
jobs = list(range(1554129, 1554129+5))+list(range(1586343, 1586343+5))
print(len(jobs)==n_jobs)

time_cmd = 'sacct -j {} -n --format=elapsed'
delta_params = ('days', 'hours', 'minutes', 'seconds')
time_re = re.compile(r'(?:(\d)-)?(\d+):(\d+):(\d+)')

δs = timedelta()
for job in jobs:
	_, elapsed = subprocess.getstatusoutput(
		time_cmd.format(job))

	match = time_re.match(elapsed.strip())
	args = map(int, [a or 0 for a in match.group(*range(1,5))])
	δ = timedelta(**dict(zip(delta_params, args)))
	print(δ)
	δs += δ
print(round((δs/n_jobs).total_seconds()))

# 68529 -d cifar10 -m 3 -e 2000 -c 15 -r 10
# 44229 -d cifar10 -m 1 -e 2000 -c 10 -r 10
# 56625 -d svhn -m 3 -e 2000 -c 15 -r 10
# 42357 -d svhn -m 1 -e 2000 -c 10 -r 10
# 54041 -d fashion-mnist -m 3 -e 2000 -c 15 -r 10
# 35813 -d fashion-mnist -m 1 -e 2000 -c 10 -r 10
# 66089 -d cifar100 -m 3 -e 2000 -c 15 -r 10
# 40766 -d cifar100 -m 1 -e 2000 -c 10 -r 10