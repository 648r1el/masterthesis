#! /usr/bin/bash

# run the experiments for the COAD analysis with omiembed
#
# $1 is the number of experiments

# link the samples file
ln -fT data/all_samples.tsv data/sample_list.tsv
for ((i=0;i<$1;i++))
do
	echo Running experiment all_cancers_classification_v05_$i
	# train the model
	python train_test.py --omics-mode abcd --save-model --use-sample-list --detect-na --net-VAE fc --latent-space 512 --epoch-num-p1 20 --epoch-num-p2 30 --epoch-num-p3 50 --label Tumor_codes --batch-size 256 --gpu-ids 0 --lr 0.001 --experiment-name all_cancer_classification_v05_$i
	echo calculating the shap values for all_cancers_classification_$i
	python explain_model.py -e 100 -g -d abcd -c data/clinical.tsv COAD all_cancer_classification_v05_$i
done
# link the samples file
ln -fT data/cancer_coad_samples.tsv data/sample_list.tsv
for ((i=0;i<$1;i++))
do
	echo Running experiment coad_normal_classification_$i
	# train the model
	python train_test.py --omics-mode abcd --save-model --use-sample-list --detect-na --net-VAE fc --latent-space 512 --epoch-num-p1 10 --epoch-num-p2 15 --epoch-num-p3 15 --label Tumor_codes --batch-size 256 --gpu-ids 0 --lr 0.0001 --down-task-file clinical_coad_normal.tsv --experiment-name coad_normal_classification_v05_$i
	echo calculating the shap values for coad_normal_classification_$i
	python explain_model.py -e 40 -g -d abcd -c data/clinical_coad_normal.tsv COAD coad_normal_classification_v05_$i
done
