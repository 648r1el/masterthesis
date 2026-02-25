import re
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import datetime as dt

DATA_PATH = 'GDCdata/TCGA-COAD'

DNA_MET_PATIENT_IDS = f'{DATA_PATH}/dna_methylation_patient_ids.tsv'
MIRNA_PATIENT_IDS = f'{DATA_PATH}/mirna_patient_ids.tsv'
LNCRNA_PATIENT_IDS = f'{DATA_PATH}/lncrna_patient_ids.tsv'

DNA_MET_DATA_PATH = f'{DATA_PATH}/DNA_Methylation/Methylation_Beta_Value'
MIRNA_DATA_PATH = f'{DATA_PATH}/Transcriptome_Profiling/miRNA_Expression_Quantification'
LNCRNA_DATA_PATH = f'{DATA_PATH}/Transcriptome_Profiling/Gene_Expression_Quantification'
CLINICAL_DATA_PATH = f'{DATA_PATH}/Clinical/Clinical_Supplement'

DNA_MET_MOFA_FILE = '../MOFA2/Input/mofa_coad_dna_methylation.tsv'
MIRNA_MOFA_FILE = '../MOFA2/Input/mofa_coad_mirna.tsv'
LNCRNA_MOFA_FILE = '../MOFA2/Input/mofa_coad_lncrna.tsv'
CLINICAL_MOFA_FILE = '../MOFA2/mofa_coad_groups.tsv'
RNA_SEQ_MOFA_FILE = '../MOFA2/Input/mofa_coad_rna_seq.tsv'
RNA_SEQ_DESEQ_FILE = '../MOFA2/deseq_coad_rna_seq.tsv'

def _get_cancer_stage(d: dict):
    # decide by metastasis level if there is cancer stage IV or not
    if d['M'] == 'MX' or d['N'] == 'NX' or d['T'] == 'TX' or d['M'] is None or d['N'] is None or d['T'] is None:
        return 5
    elif d['M'].startswith('M1'):
        return 4
    elif d['N'].startswith('N1'):
        return 3
    elif d['T'].startswith('T3') or d['T'].startswith('T4'):
        return 2
    elif d['T'].startswith('T2') or d['T'].startswith('T1'):
        return 1
    else:
        return 0

def read_clinical():
    clinical_data = dict()
    all_therapies = set()
    for f in Path(CLINICAL_DATA_PATH).glob('**/*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        root.iter()
        patient_id = root[1].find('{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_barcode').text
        cancer_stage = {'T': [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}pathologic_T')][0].text,
                        'N': [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}pathologic_N')][0].text,
                        'M': [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}pathologic_M')][0].text
                        }
        if re.search('omf', f.name):
            sex = None
            age = 0
            days_to_last_followup = 0
            days_to_death = 0
            vital_status = None
            colon_polyps_present = None
            history_of_colon_polyps = None
        else:
            sex = [*root.iter('{http://tcga.nci/bcr/xml/shared/2.7}gender')][0].text
            age = [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/2.7}age_at_initial_pathologic_diagnosis')][0].text
            days_to_last_followup = [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/2.7}days_to_last_followup')][0].text
            days_to_death = [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/2.7}days_to_death')][0].text
            vital_status = [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/2.7}vital_status')][0].text
            colon_polyps_present = [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/coad_read/2.7}colon_polyps_present')][0].text
            history_of_colon_polyps = [*root.iter('{http://tcga.nci/bcr/xml/clinical/shared/coad_read/2.7}history_of_colon_polyps')][0].text
            # all_therapies |= set([i.text for i in [*root.iter('{http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7}therapy_type')]])
        days_to_last_followup = 0 if days_to_last_followup is None else int(days_to_last_followup)
        days_to_death = 0 if days_to_death is None else int(days_to_death)
        metastasis = cancer_stage['M'].startswith('M1') if (cancer_stage['M'] is not None
                                                            and cancer_stage['M'] != 'MX') else None
        clinical_data[patient_id] = {
            'age': age,
            'sex': int(sex == 'FEMALE') if sex is not None else None,
            'survival_time': max(days_to_last_followup, days_to_death),
            'vital_status': int(vital_status.lower() == 'dead') if vital_status is not None else None,
            'stage': _get_cancer_stage(cancer_stage),
            'metastasis': int(metastasis) if metastasis is not None else 2,
            'colon_polyps_present': colon_polyps_present == 'YES' if colon_polyps_present is not None else None,
            'history_of_colon_polyps': history_of_colon_polyps == 'YES' if history_of_colon_polyps is not None else None
        }
    return pd.DataFrame.from_dict(clinical_data, orient='index')

def read_mirna():
    mirna_data = dict()
    patient_ids = pd.read_csv(MIRNA_PATIENT_IDS, sep='\t', index_col='id')
    for f in Path(MIRNA_DATA_PATH).glob('**/*.txt'):
        data = pd.read_csv(f, sep='\t', usecols=['miRNA_ID', 'reads_per_million_miRNA_mapped'],
                           index_col='miRNA_ID').to_dict()['reads_per_million_miRNA_mapped']
        mirna_data[patient_ids.at[f.parent.name, 'cases.submitter_id']] = data
    return pd.DataFrame.from_dict(mirna_data)

def read_lncrna():
    lncrna_data = dict()
    patient_ids = pd.read_csv(LNCRNA_PATIENT_IDS, sep='\t', index_col='id')
    for f in Path(LNCRNA_DATA_PATH).glob('**/*.tsv'):
        data = pd.read_csv(f, sep='\t', index_col='gene_id', header=1)
        data = data.loc[data['gene_type'] == 'lncRNA', :]
        lncrna_data[patient_ids.at[f.parent.name, 'cases.submitter_id']] = data.to_dict()['fpkm_unstranded']
    return pd.DataFrame.from_dict(lncrna_data)

def read_methylation():
    met_data = dict()
    patient_ids = pd.read_csv(DNA_MET_PATIENT_IDS, sep='\t', index_col='id')
    for f in Path(DNA_MET_DATA_PATH).glob('**/*.txt'):
        data = pd.read_csv(f, sep='\t', index_col=0, header=None).to_dict()[1]
        met_data[patient_ids.at[f.parent.name, 'cases.submitter_id']] = data
    return pd.DataFrame.from_dict(met_data)

def read_rna_seq():
    rna_seq_data = dict()
    patient_ids = pd.read_csv(LNCRNA_PATIENT_IDS, sep='\t', index_col='id')
    for f in Path(LNCRNA_DATA_PATH).glob('**/*.tsv'):
        data = pd.read_csv(f, sep='\t', index_col='gene_id', header=1)
        data = data.loc[(data['gene_type'] != 'lncRNA') & (data['gene_type'] != 'miRNA') & ~(data['gene_type'].isna()), :]
        rna_seq_data[patient_ids.at[f.parent.name, 'cases.submitter_id']] = data.to_dict()['fpkm_unstranded']
    return pd.DataFrame.from_dict(rna_seq_data)

def read_rna_seq_for_deseq2():
    rna_seq_data = dict()
    patient_ids = pd.read_csv(LNCRNA_PATIENT_IDS, sep='\t', index_col='id')
    for f in Path(LNCRNA_DATA_PATH).glob('**/*.tsv'):
        data = pd.read_csv(f, sep='\t', index_col='gene_id', header=1)
        data = data.loc[(data['gene_type'] != 'lncRNA') & (data['gene_type'] != 'miRNA') & ~(data['gene_type'].isna()), :]
        rna_seq_data[patient_ids.at[f.parent.name, 'cases.submitter_id']] = (
            (data['unstranded'] + data['stranded_first'] + data['stranded_second']).to_dict())
    return pd.DataFrame.from_dict(rna_seq_data, orient='index')

if __name__ == '__main__':
    # read_clinical().to_csv(CLINICAL_MOFA_FILE, sep='\t')
    read_rna_seq().to_csv(RNA_SEQ_MOFA_FILE, sep='\t')  # needs 5 minutes
    read_mirna().to_csv(MIRNA_MOFA_FILE, sep='\t')
    read_lncrna().to_csv(LNCRNA_MOFA_FILE, sep='\t')  # needs 5 minutes
    read_methylation().to_csv(DNA_MET_MOFA_FILE, sep='\t')  # needs approximately 20 minutes
    # read_rna_seq_for_deseq2().to_csv(RNA_SEQ_DESEQ_FILE, sep='\t')
