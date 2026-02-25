import argparse
import os
from pathlib import Path

import pandas as pd
import polars as pl
import shap
from argparse import ArgumentParser, Namespace

from torch import nn, Tensor

from models import create_model
from params.load_params import LoadParams

import torch

from models.networks import FcVaeABCD, FcVaeA, FcVaeB, FcVaeC, FcVaeD
from models.vae_classifier_model import VaeClassifierModel

ID_COLS = {
    'a': 'Gene',
    'b': 'Probe',
    'c': 'miRNA_ID',
    'd': 'Gene'
}

class ModelWrapper(nn.Module):
    def __init__(self, classifier_model: VaeClassifierModel):
        super(ModelWrapper, self).__init__()
        self.classifier_model = classifier_model
        classifier_model.update()

    def forward(self, input: Tensor):
        tensor_df = input
        y_embed = self.classifier_model.netEmbed([tensor_df] * 4)[0]
        y_down = self.classifier_model.netDown(y_embed)
        return y_down


def set_omics_dims(omics_mode, samples):
    dims = []
    omics_dfs = {}
    for omic in omics_mode:
        print(f'Reading file {omic.upper()}.tsv')
        file_path = os.path.join('data', f'{omic.upper()}.tsv')
        raw_omics_data = pl.read_csv(file_path, separator='\t', has_header=True, columns=[ID_COLS[omic]] + samples).fill_nan(0).fill_null(0)
        dims.append(len(raw_omics_data))
        omics_dfs[omic] = raw_omics_data
    return dims, omics_dfs


def set_fc_embedding_networks(embedding_model: FcVaeABCD, omics_dims: list, gpu: bool):
    module = embedding_model if not gpu else embedding_model.module
    models = {
        'a': FcVaeA(omics_dims, dim_1A=module.dim_1A, dim_2A=module.dim_2A,
                    dim_3=module.dim_3, latent_dim=module.encode_fc_3.fc_block[0].out_features),
        'b': FcVaeB(omics_dims, dim_1B=module.dim_1B, dim_2B=module.dim_2B,
                    dim_3=module.dim_3, latent_dim=module.encode_fc_3.fc_block[0].out_features),
        'c': FcVaeC(omics_dims, dim_1C=module.dim_1C, dim_2C=module.dim_2C,
                    dim_3=module.dim_3, latent_dim=module.encode_fc_3.fc_block[0].out_features),
        'd': FcVaeD(omics_dims, dim_1D=module.dim_1D, dim_2D=module.dim_2D,
                    dim_3=module.dim_3, latent_dim=module.encode_fc_3.fc_block[0].out_features)
    }
    # split the concatenated layers, encode_3, mean, log_var and decode_2, into respective chunks to split them into the
    # datasets they correspond to
    encode_fc_3_weight_chunks = torch.chunk(module.encode_fc_3.fc_block[0].weight, 4, 1)
    encode_fc_3_bias_chunks = torch.chunk(module.encode_fc_3.fc_block[0].bias, 4)
    encode_fc_mean_weight_chunks = torch.chunk(module.encode_fc_mean.fc_block[0].weight, 4, 1)
    encode_fc_mean_bias_chunks = torch.chunk(module.encode_fc_mean.fc_block[0].bias, 4)
    encode_fc_log_var_weight_chunks = torch.chunk(module.encode_fc_log_var.fc_block[0].weight, 4, 1)
    encode_fc_log_var_bias_chunks = torch.chunk(module.encode_fc_log_var.fc_block[0].bias, 4)
    decode_fc_2_weight_chunks = torch.chunk(module.decode_fc_2.fc_block[0].weight, 4, 1)
    decode_fc_2_bias_chunks = torch.chunk(module.decode_fc_2.fc_block[0].bias, 4)

    # assign the different models for each omic layer
    # Omic layer A, gene expression
    # assign the unique encoding and decoding values unique to the omic layer
    models['a'].encode_fc_1A = module.encode_fc_1A
    models['a'].encode_fc_1A.fc_block[3].inplace = False
    models['a'].encode_fc_2A = module.encode_fc_2A
    models['a'].encode_fc_2A.fc_block[3].inplace = False
    models['a'].decode_fc_3A = module.decode_fc_3A
    models['a'].decode_fc_3A.fc_block[3].inplace = False
    models['a'].decode_fc_4A = module.decode_fc_4A
    # assign the values of the joint layers
    models['a'].encode_fc_3.weight = nn.parameter.Parameter(
        encode_fc_3_weight_chunks[1], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['a'].encode_fc_3.bias = nn.parameter.Parameter(
        encode_fc_3_bias_chunks[1], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['a'].encode_fc_mean.weight = nn.parameter.Parameter(
        encode_fc_mean_weight_chunks[1], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['a'].encode_fc_mean.bias = nn.parameter.Parameter(
        encode_fc_mean_bias_chunks[1], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['a'].encode_fc_log_var.weight = nn.parameter.Parameter(
        encode_fc_log_var_weight_chunks[1], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['a'].encode_fc_log_var.bias = nn.parameter.Parameter(
        encode_fc_log_var_bias_chunks[1], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['a'].decode_fc_2.weight = nn.parameter.Parameter(
        decode_fc_2_weight_chunks[1], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    models['a'].decode_fc_2.bias = nn.parameter.Parameter(
        decode_fc_2_bias_chunks[1], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    # Omic layer B, DNA Methylation
    # assign the unique encoding and decoding values unique to the omic layer
    models['b'].encode_fc_1B = module.encode_fc_1B
    models['b'].encode_fc_1B.fc_block[3].inplace = False
    models['b'].encode_fc_2B = module.encode_fc_2B
    models['b'].encode_fc_2B.fc_block[3].inplace = False
    models['b'].decode_fc_3B = module.decode_fc_3B
    models['b'].decode_fc_3B.fc_block[3].inplace = False
    models['b'].decode_fc_4B = module.decode_fc_4B
    # assign the values of the joint layers
    models['b'].encode_fc_3.weight = nn.parameter.Parameter(
        encode_fc_3_weight_chunks[0], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['b'].encode_fc_3.bias = nn.parameter.Parameter(
        encode_fc_3_bias_chunks[0], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['b'].encode_fc_mean.weight = nn.parameter.Parameter(
        encode_fc_mean_weight_chunks[0], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['b'].encode_fc_mean.bias = nn.parameter.Parameter(
        encode_fc_mean_bias_chunks[0], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['b'].encode_fc_log_var.weight = nn.parameter.Parameter(
        encode_fc_log_var_weight_chunks[0], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['b'].encode_fc_log_var.bias = nn.parameter.Parameter(
        encode_fc_log_var_bias_chunks[0], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['b'].decode_fc_2.weight = nn.parameter.Parameter(
        decode_fc_2_weight_chunks[0], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    models['b'].decode_fc_2.bias = nn.parameter.Parameter(
        decode_fc_2_bias_chunks[0], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    # Omic layer C, miRNA expression
    # assign the unique encoding and decoding values unique to the omic layer
    models['c'].encode_fc_1C = module.encode_fc_1C
    models['c'].encode_fc_1C.fc_block[3].inplace = False
    models['c'].encode_fc_2C = module.encode_fc_2C
    models['c'].encode_fc_2C.fc_block[3].inplace = False
    models['c'].decode_fc_3C = module.decode_fc_3C
    models['c'].decode_fc_3C.fc_block[3].inplace = False
    models['c'].decode_fc_4C = module.decode_fc_4C
    # assign the values of the joint layers
    models['c'].encode_fc_3.weight = nn.parameter.Parameter(
        encode_fc_3_weight_chunks[2], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['c'].encode_fc_3.bias = nn.parameter.Parameter(
        encode_fc_3_bias_chunks[2], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['c'].encode_fc_mean.weight = nn.parameter.Parameter(
        encode_fc_mean_weight_chunks[2], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['c'].encode_fc_mean.bias = nn.parameter.Parameter(
        encode_fc_mean_bias_chunks[2], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['c'].encode_fc_log_var.weight = nn.parameter.Parameter(
        encode_fc_log_var_weight_chunks[2], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['c'].encode_fc_log_var.bias = nn.parameter.Parameter(
        encode_fc_log_var_bias_chunks[2], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['c'].decode_fc_2.weight = nn.parameter.Parameter(
        decode_fc_2_weight_chunks[2], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    models['c'].decode_fc_2.bias = nn.parameter.Parameter(
        decode_fc_2_bias_chunks[2], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    # Omic layer D, lncRNA expression
    # assign the unique encoding and decoding values unique to the omic layer
    models['d'].encode_fc_1D = module.encode_fc_1D
    models['d'].encode_fc_1D.fc_block[3].inplace = False
    models['d'].encode_fc_2D = module.encode_fc_2D
    models['d'].encode_fc_2D.fc_block[3].inplace = False
    models['d'].decode_fc_3D = module.decode_fc_3D
    models['d'].decode_fc_3D.fc_block[3].inplace = False
    models['d'].decode_fc_4D = module.decode_fc_4D
    # assign the values of the joint layers
    models['d'].encode_fc_3.weight = nn.parameter.Parameter(
        encode_fc_3_weight_chunks[3], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['d'].encode_fc_3.bias = nn.parameter.Parameter(
        encode_fc_3_bias_chunks[3], requires_grad=module.encode_fc_3.fc_block[0].weight.requires_grad)
    models['d'].encode_fc_mean.weight = nn.parameter.Parameter(
        encode_fc_mean_weight_chunks[3], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['d'].encode_fc_mean.bias = nn.parameter.Parameter(
        encode_fc_mean_bias_chunks[3], requires_grad=module.encode_fc_mean.fc_block[0].weight.requires_grad)
    models['d'].encode_fc_log_var.weight = nn.parameter.Parameter(
        encode_fc_log_var_weight_chunks[3], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['d'].encode_fc_log_var.bias = nn.parameter.Parameter(
        encode_fc_log_var_bias_chunks[3], requires_grad=module.encode_fc_log_var.fc_block[0].weight.requires_grad)
    models['d'].decode_fc_2.weight = nn.parameter.Parameter(
        decode_fc_2_weight_chunks[3], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    models['d'].decode_fc_2.bias = nn.parameter.Parameter(
        decode_fc_2_bias_chunks[3], requires_grad=module.decode_fc_2.fc_block[0].weight.requires_grad)
    if gpu:
        models = {key: nn.DataParallel(models[key], [0]) for key in models}
    return models



def explain(classifier_model: VaeClassifierModel, raw_omics_df: pd.DataFrame, tumor_codes:pl.DataFrame, cancer_type, omic: str, device):
    sample_number = 75

    background_sample = raw_omics_df.join(tumor_codes, on='Barcode').filter(pl.col('Tumor_codes') != cancer_type).drop('Tumor_codes')
    tumor_sample = raw_omics_df.join(tumor_codes, on='Barcode').filter(pl.col('Tumor_codes') == cancer_type).drop('Tumor_codes')
    sample_number = min(sample_number, len(background_sample), len(tumor_sample))

    background_sample = background_sample.sample(n=sample_number, seed=42)
    tumor_sample = tumor_sample.sample(n=sample_number, seed=42)

    # get the patient ids for the VAE classifier models
    background_patients = background_sample[:, 0].to_list()
    # type cast to floats; polars read the dataframe as strings due to the ids being a column, not a column name, after transposing
    background_sample = background_sample[:, 1:].cast(pl.Float32)
    tumor_sample = tumor_sample[:, 1:].cast(pl.Float32)

    background = background_sample.to_torch().to(device)
    tumor_expr_tensor = tumor_sample.to_torch().to(device)

    # set the input for the VAE Classifier model
    classifier_model.set_input({'index': background_patients, 'input_omics': background, 'label': Tensor(tumor_codes.filter(pl.col('Barcode').is_in(background_patients))['Tumor_codes']).to(torch.long)})
    classifier_model.input_omics = [None, None, None, None]
    classifier_model.input_omics[ord(omic) - ord('a')] = background

    explainer = shap.DeepExplainer(ModelWrapper(classifier_model), background)
    shap_values = explainer.shap_values(tumor_expr_tensor, ranked_outputs=None, check_additivity=False)

    return shap_values[:, :, cancer_type]


def set_parser():
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs used to train the model to load; it is part of the filename and, thus, '
                             'can alter when the number of epochs is changed')
    parser.add_argument('-c', '--clinical-file', default='data/clinical.tsv',
                        help='clinical file which contains all patient information')
    parser.add_argument('-d', '--detect-na', action='store_true', default=False,
                        help='Detect the NAs in the raw data files if given')
    parser.add_argument('-g', '--gpu', default=False, action='store_true',
                        help='Indicates whether torch processes on GPU or CPU')
    parser.add_argument('omics_mode',
                        help='Number of omics applied with OmiEmbed, must be valid according to the input of OmiEmbed')
    parser.add_argument('cancer_type', help='Type of cancer which shall be explained')
    parser.add_argument('experiment_name',
                        help='Path to the trained model of the experiment which shall be explained')
    return parser


if __name__ == '__main__':
    p = set_parser()
    args = p.parse_args()

    # define constants
    OMICS_MODES = {'a', 'b', 'c', 'ab', 'abc', 'abcd'}
    SOURCES = {
        'a': 'Gene_Expression',
        'b': 'DNA_Methylation',
        'c': 'miRNA_Expression',
        'd': 'lncRNA_Expression'
    }
    DEVICES = ['cpu', 'cuda']
    # create directories for SOURCES if they do not exist
    for omic in SOURCES.values():
        if not os.path.exists(f'explanation/{omic}'):
            os.mkdir(f'explanation/{omic}')
    # read the clinical data
    clinical_df = pl.read_csv(args.clinical_file, separator='\t', has_header=True)
    cancer_codes = clinical_df.select('Tumor', 'Tumor_codes').unique()
    cancer_codes = {row[0]: row[1] for row in cancer_codes.rows() }
    param_loader = LoadParams(args.experiment_name, args.epochs, len(cancer_codes), use_gpus=args.gpu)
    param = param_loader.get_params()
    # handle correct input
    if args.experiment_name not in os.listdir('checkpoints'):
        raise FileNotFoundError(f'No experiment {args.experiment_name} in checkpoints')
    # assign the paths to the embedding and downstream model
    downstream_path = Path(os.path.join('checkpoints', args.experiment_name, f'{args.epochs}_net_Down.pth'))
    embedding_path = Path(os.path.join('checkpoints', args.experiment_name, f'{args.epochs}_net_Embed.pth'))
    if not (downstream_path.exists() and embedding_path.exists()):
        raise ValueError(f'Experiment {args.experiment_name} has no fully trained model with {args.epochs} epochs yet. '
                         'Choose another experiment or run OmiEmbed to access it')
    if args.omics_mode not in OMICS_MODES:
        raise ValueError('Omics mode must be one of ' + str(OMICS_MODES))
    if args.cancer_type not in cancer_codes.keys():
        raise ValueError('Cancer class not known, must be one of ' + str(cancer_codes.keys()))
    # read the omics files and the dimensions
    print('Reading the omics data')
    param.omics_dims, omics_dfs = set_omics_dims(args.omics_mode, clinical_df['Barcode'].to_list())
    param.ch_separate = False
    # read the models (downstream and embedding) from checkpoints directory
    torch.set_printoptions(precision=8)
    print('Loading the model')
    model = create_model(param)
    model.setup(param)
    downstream_model = model.netDown
    embedding_model = model.netEmbed

    # change the inplace variable of the FCBlocks of the downstream models to False; they are set True by OmiEmbed, but
    # must not be True for the shap value evaluation
    if args.gpu:
        downstream_model.module.input_fc.fc_block[3].inplace = False
        downstream_model.module.mul_fc[0].fc_block[3].inplace = False
    else:
        downstream_model.input_fc.fc_block[3].inplace = False
        downstream_model.mul_fc[0].fc_block[3].inplace = False

    # assign further needed variables
    device = DEVICES[args.gpu]
    # create embedding models per layer
    print('Create the embedding models for each omic')
    individual_embed_models = set_fc_embedding_networks(embedding_model, param.omics_dims, args.gpu)
    # read the input dataset files for OmiEmbed
    for omic in args.omics_mode:
        param.omics_mode = omic
        param.omics_num = 1
        classifier_model = VaeClassifierModel(param)
        classifier_model.phase = 'p3'
        classifier_model.netEmbed = individual_embed_models[omic]
        classifier_model.netDown = downstream_model
        print(f'Explaining the embedding and the downstream model for {omic} omics')
        df_to_explain = omics_dfs[omic].transpose(include_header=True, header_name='Barcode', column_names=omics_dfs[omic][ID_COLS[omic]])[1:]
        result = explain(classifier_model, df_to_explain, clinical_df.select('Barcode', 'Tumor_codes'), cancer_codes[args.cancer_type], omic, device)
        result_df = pl.DataFrame(result)
        result_df.columns = df_to_explain.columns[1:]
        result_df.write_csv(f'explanation/{SOURCES[omic]}/shap_values_{args.experiment_name}.tsv', separator='\t')
