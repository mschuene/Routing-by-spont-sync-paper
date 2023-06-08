#!/bin/bash
# run sh /sge_run.sh jobname queue task_ids script
# script should be relative to the project dir
# script is python file that is run in the project dir
# task_id is defined when script runs
cd "$(dirname "$0")"
jobname=$1
queue=$2
job_ids=$3
script=$4
mkdir $jobname
cd $jobname

jobid=$(qsub -terse -N $jobname -q $queue -cwd -pe smp 2 -j y -p 0 -t $job_ids <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
export jobname=$jobname
export script=$script

ipython --HistoryManager.enabled=False <<EOF
# common setup code
import matplotlib
matplotlib.use('Agg')
import pickle 
import os
import sh
project_dir = "/home/maik/attmod"
res_dir = "/0/maik/attmod"
jobname = os.environ.get("jobname")
sh.cd(project_dir)
outdir = res_dir + "/" + jobname
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ,flush=True)
task_id = int(os.environ.get('SGE_TASK_ID'))
script = os.environ.get('script')
print("Task Id:",task_id)
%run -i -t $script
pickle.dump(res,open(outdir+"/res"+str(task_id)+".pickle",'wb'))
print('done')
EOF
EOT
)


echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N $jobname -q $queue -cwd  -p 0   -hold_jid $jid <<EOT
echo 'done'
EOT

qstat
