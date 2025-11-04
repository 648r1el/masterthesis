from mofapy2.run.entry_point import entry_point
import pandas as pd
import numpy as np
from argparse import ArgumentParser

def set_parser():
    parser = ArgumentParser()
    parser.add_argument('-g', '--groups', default=None, type=str,
                        help='The name of the groups which shall be specified. If not given, no groups are specified')
    return parser

if __name__ == '__main__':

    p = set_parser()
    args = p.parse_args()

    num_omics = 4

    # read the data into separate variables
    mirna_data = pd.read_csv('mofa_mirna.tsv', sep='\t', index_col=0)#.join(
    #   pd.read_csv('mofa_mirna.tsv', sep='\t', index_col=0)
    #)
    lncrna_data = pd.read_csv('mofa_lncrna.tsv', sep='\t', index_col=0)#.join(
        #   pd.read_csv('mofa_lncrna.tsv', sep='\t', index_col=0)
    #)
    methyl_data = pd.read_csv('mofa_dna_methylation.tsv', sep='\t', index_col=0).dropna(axis=0, how='all')#.join(
    #    pd.read_csv('mofa_dna_methylation.tsv', sep='\t', index_col=0)
    #).dropna(axis=0, how='all')
    rnaseq_data = pd.read_csv('mofa_rna_seq.tsv', sep='\t', index_col=0)#.join(
        #    pd.read_csv('mofa_rna_seq.tsv', sep='\t', index_col=0)
    #)

    # filter the number of samples such that only patients are used which occur in all omics layers
    mirna_samples = {*mirna_data.columns}
    lncrna_samples = {*lncrna_data.columns}
    met_samples = {*methyl_data.columns}
    rna_seq_samples = {*rnaseq_data.columns}

    # use the set operation '&' to determine the intersection between all samples
    # columns = [*mirna_samples & lncrna_samples & met_samples ]
    columns = [*mirna_samples & lncrna_samples & met_samples & rna_seq_samples]
    sample_names = []

    groups = pd.Series(['a'] * len(columns))
    save_file = 'model_coad_4_omics_fpkm_2_percent_r_square.hdf5'

    if args.groups is not None:
        """
        group_data = pd.concat([pd.read_csv('mofa_read_groups.tsv', sep='\t', index_col=0),
            pd.read_csv('mofa_groups.tsv', sep='\t', index_col=0)], axis=0
        )
        """
        group_data = pd.read_csv('mofa_coad_groups.tsv', sep='\t', index_col=0)
        columns = [*{*group_data.index} & {*columns}]
        groups = group_data.loc[columns, args.groups]  # columns corresponds to the samples, which are the index of groups
        groups.index = range(len(groups))
        save_file = f'test_model_coad_4_omics_fpkm_{args.groups}.hdf5'

    group_values = [*groups.drop_duplicates().dropna().values]

    # prepare the data for the training model
    data_mat = [[None] * len(group_values) for i in range(num_omics)]

    for j in range(len(group_values)):
        group_samples = pd.Series(columns).loc[groups == group_values[j]].values

        # select all patients whose data were measured for all omics and remove the features with missing values
        data_mat[0][j] = mirna_data.loc[:, group_samples]
        data_mat[1][j] = lncrna_data.loc[:, group_samples]
        data_mat[2][j] = methyl_data.loc[:, group_samples]
        data_mat[3][j] = rnaseq_data.loc[:, group_samples]

        # set the feature names
        features = [[*data_mat[i][j].index] for i in range(len(data_mat))]

        # transpose the matrices so that they can fit the model
        data_mat[0][j] = data_mat[0][j].values.T
        data_mat[1][j] = data_mat[1][j].values.T
        data_mat[2][j] = data_mat[2][j].values.T
        data_mat[3][j] = data_mat[3][j].values.T

        sample_names.append(group_samples)

    # train the model
    ent = entry_point()

    ent.set_data_options(
        scale_views = True
    )

    ent.set_data_matrix(data_mat, likelihoods=['gaussian'] * num_omics, views_names=['miRNA', 'lncRNA', 'Methylation', 'RNASeq'],
                        samples_names=sample_names, features_names=features)

    ent.set_model_options(
        factors=10,
        spikeslab_weights=True,
        ard_weights=True
    )
    ent.set_train_options(
        convergence_mode='fast',
        dropR2=0.02,
        gpu_mode=False,
        seed=1
    )

    ent.build()

    ent.run()

    ent.save(save_file)

