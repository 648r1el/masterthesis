#!/bin/bash
#
# determin the best lr and bs
#
# $1 is the experiment name
# $2 is the file which contains the information for the downstream task
# $3 are the GPU IDs to use

OPTS=$(getopt -o '' -l p1:,p2: -- "$@")
eval set -- ${OPTS}
# set the default values for --p1 and --p2
p1=50
p2=50

# check whether the options are parsed correctly
while :
do
	case $1 in
		--p1) 
			p1=$2 
			shift 2
			;;
		--p2) 
			p2=$2
			shift 2
			;;
		--) 
			shift
			break
			;;
		*)
			>&2 echo Wrong order of the arguments
			exit 1
			;;
	esac
done

# check the correct values
re_p1='^x[0-9]+$'
if [[ ! x$p1 =~ $re_p1 ]]; then
	>&2 echo $p1 is not a valid integer for flag --p1
	exit 1;
fi
if [[ ! x$p2 =~ $re_p1 ]]; then
	>&2 echo $p2 is not a valid integer for flag --p2
	exit 1;
fi
if [[ x$1 = "x" ]]; then
	>&2 echo No experiment name parsed
	exit 1
fi

if [[ ! -f data/$2 ]]; then
	>&2 echo Downstream task file $2 does not exist
	exit 1
fi
re_gpu='^x$|^x[0-9]+(,[0-9]+)*$'
if [[ ! x$3 =~ $re_gpu ]]; then
	>&2 echo GPU Ids must be either empty or a comma separated list of integers without whitespaces
	exit 1
fi
if [[ x$3 = "x" ]]; then
	gpu_id=-1
else
	gpu_id=$3
fi

# start the grid search
> grid_search_hyperparameters/train_stats_grid_search_lr_bs_$1.tsv
> grid_search_hyperparameters/test_stats_grid_search_lr_bs_$1.tsv
for i in {2..7}
do
	lr=$(printf "0.%0*d" "$i" 1)
	for j in {5..8}
	do
		bs=$((2**$j))
		echo Performance metrics for lr $lr and batch size $bs >> grid_search_hyperparameters/train_stats_grid_search_lr_bs_$1.tsv
		echo Performance metrics for lr $lr and batch size $bs >> grid_search_hyperparameters/test_stats_grid_search_lr_bs_$1.tsv
		python train_test.py --omics-mode abcd --save-model --use-sample-list --detect-na --net-VAE fc --latent-space 512 --epoch-num-p1 $p1 --epoch-num-p2 $p1 --epoch-num-p3 $((p1+p2)) --down-task-file $2 --label Tumor_codes --batch-size $bs --gpu-ids $gpu_id --lr $lr --experiment-name $1\_lr_$i\_bs_$bs
		tail -n 1 checkpoints/$1\_lr_$i\_bs_$bs/train_summary.txt >> grid_search_hyperparameters/train_stats_grid_search_lr_bs_$1.tsv
		tail -n 1 checkpoints/$1\_lr_$i\_bs_$bs/test_summary.txt >> grid_search_hyperparameters/test_stats_grid_search_lr_bs_$1.tsv
	done
done
