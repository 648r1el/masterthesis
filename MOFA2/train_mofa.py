from mofapy2.run.entry_point import entry_point
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

def set_parser():
    parser = ArgumentParser()
    parser.add_argument('-g', '--groups', default=None, type=str,
                        help='The name of the groups which shall be specified. If not given, no groups are specified')
    parser.add_argument('-v', '--variance-filter', default=0, type=float,
                        help='Only consider columns with the given amount of variance')
    parser.add_argument('-c', '--cancer-type', default='coad', type=str, help='The name of the caner type to investigate')
    parser.add_argument('-n', '--num-omics', default=4, type=int, help='Number of omics layers')
    parser.add_argument('-o', '--outfile-suffix', default='_fpkm_ucsc', type=str,
                        help='The suffix of the output file to specify the output a bit more. E.g., the default suffix '
                             'refers to the used normalization method (FPKM) and the source of the dataset (UCSC). '
                             'You can specify the name of the output file in any way you want')
    parser.add_argument('-l', '--log-base', default=None, type=str,
                        help='Indicates the log base of the normalized data of transcriptomics and mirna expression '
                             'if it was calculated to log odds. The data is then reversed to its original normalized '
                             'count data to perform the analysis. Log-odds are '
                             'usually calculated with the base 2, but any other number can be parsed; the Eulerian '
                             'number must be parsed as "e". If the data is already normalized, this flag can be '
                             'ignored.')
    return parser

if __name__ == '__main__':

    p = set_parser()
    args = p.parse_args()

    num_omics = args.num_omics

    # read the data into separate variables
    # this section is hardcoded, but can be easily augmented or reduced by adding or removing a line with the respective
    # data.
    print('Reading the input data')
    mirna_data = pd.read_csv(f'Input/mofa_{args.cancer_type}_mirna.tsv', sep='\t', index_col=0)
    lncrna_data = pd.read_csv(f'Input/mofa_{args.cancer_type}_lncrna.tsv', sep='\t', index_col=0)
    methyl_data = pd.read_csv(f'Input/mofa_{args.cancer_type}_dna_methylation.tsv.bz2', sep='\t', index_col=0).dropna(axis=0, how='all')
    rnaseq_data = pd.read_csv(f'Input/mofa_{args.cancer_type}_rna_seq.tsv.bz2', sep='\t', index_col=0)
    print('Data read successfully')

    if args.log_base is not None:
        if args.log_base == 'e':
            mirna_data = np.exp(mirna_data) - 1
            lncrna_data = np.exp(lncrna_data) - 1
            rnaseq_data = np.exp(rnaseq_data) - 1
        else:
            log_base = int(args.log_base)
            mirna_data = np.power(log_base, mirna_data) - 1
            lncrna_data = np.power(log_base, lncrna_data) - 1
            rnaseq_data = np.power(log_base, rnaseq_data) - 1

    print('Filter by variance if variance is not 0')
    mirna_data = mirna_data.loc[mirna_data.var(axis=1) >= args.variance_filter]
    lncrna_data = lncrna_data.loc[lncrna_data.std(axis=1) >= args.variance_filter]
    methyl_data = methyl_data.loc[methyl_data.std(axis=1) >= args.variance_filter]
    rnaseq_data = rnaseq_data.loc[rnaseq_data.std(axis=1) >= args.variance_filter]

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
    # define parameters for the name of the output file
    if not Path('Trained_models').exists():
        Path('Trained_models').mkdir(parents=True, exist_ok=True)
    suffix = f'_{args.outfile_suffix}' if args.outfile_suffix is not None else ''
    save_file = f'Trained_models/model_{args.cancer_type}_{num_omics}_omics_var_{args.variance_filter}{args.outfile_suffix}.hdf5'

    if args.groups is not None:
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
        dropR2=0.001,
        gpu_mode=False,
        seed=1
    )

    ent.build()

    ent.run()

    ent.save(save_file)

