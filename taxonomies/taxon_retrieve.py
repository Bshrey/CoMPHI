import pandas as pd
import os
import requests 
import wget
import gzip

genbank_file_names = []
with open('../GCF_to_NC.csv', 'r') as f:
    for line in f:
        genbank_file_names.append(line.split(',')[-1].strip())
genbank_file_names = genbank_file_names

refseq_host_matches = pd.read_csv('../refseq.csv').set_index('Assembly')[['Host']].to_dict()
refseq_host_matches = refseq_host_matches['Host']

phage_host_name = {}
base_url = 'https://ftp.ncbi.nih.gov/genomes/all'
for genbank_fn in genbank_file_names:
    folder_url = f"{base_url}/{genbank_fn[0:3]}/{genbank_fn[4:7]}/{genbank_fn[7:10]}/{genbank_fn[10:13]}/{genbank_fn.replace('_genomic.fna', '')}"
    url = f"{folder_url}/{genbank_fn.rstrip('.fna')}.gbff.gz"
    wget.download(url)
    with gzip.open(f"{genbank_fn.rstrip('.fna')}.gbff.gz", 'r') as f:
        content = f.read().decode().strip()
    os.remove(f"{genbank_fn.rstrip('.fna')}.gbff.gz")
    host = ''
    for line in content.split('\n'):
        if '/host' in line:
            host = line.strip().lstrip('/host="').rstrip('"')
    if host == '':
        host = refseq_host_matches.get('_'.join(genbank_fn.split('_')[:2]))
    phage_host_name['_'.join(genbank_fn.split('_')[:2])] = host

phage_host_name = pd.read_csv('phage_host_name_mapping.csv').set_index('phage').to_dict()['host_name']
phage_host_id = {}
for phage, host_name in phage_host_name.items():
    result = requests.post(
        url='https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy',
        headers={
            "Accept": "application/json", 
            "api-key": "e994c8d33778342b377eba7777a317791407", 
            "Content-Type": "application/json"
        },
        json={'taxons': [host_name]}
    )
    try:
        phage_host_id[phage] = result.json()['taxonomy_nodes'][0]['taxonomy']['tax_id']
    except:
        print(result.json())
        phage_host_id[phage] = ''

phage_host_accession = {}
for phage, host_id in phage_host_id.items():
    result = requests.get(
        url=f'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/taxon/{host_id}/dataset_report',
        headers={
            "Accept": "application/json", 
            "api-key": "e994c8d33778342b377eba7777a317791407", 
            "Content-Type": "application/json"
        },
        params={
            'filters.reference_only': True,
            'filters.has_annotation': False
        }
    )
    try:
        phage_host_accession[phage] = result.json()['reports'][0]['current_accession']
    except:
        phage_host_accession[phage] = ''
df = pd.DataFrame.from_dict(phage_host_accession, orient='index', columns=['host_accession']).reset_index().rename(columns={'index': 'phage'})
df.to_csv('phage_host_accession.csv', index=False)

df = pd.DataFrame.from_dict(phage_host_name, orient='index', columns=['host_name']).reset_index().rename(columns={'index': 'phage'})
df.to_csv('phage_host_name_mapping.csv', index=False)

df = pd.DataFrame.from_dict(phage_host_id, orient='index', columns=['host_taxon_id']).reset_index().rename(columns={'index': 'phage'})
df.to_csv('phage_host_taxon_id.csv', index=False)

cmds = []
with open('../taxon_cmds.txt', 'r') as f:
    for line in f:
        taxon = line.split(' ')[4]
        cmds.append(f"{line.strip()} taxon_{taxon}.zip")
with open('../taxon_cmds.txt', 'w') as f:
    f.write('\n'.join(cmds))

taxon_lin = pd.read_csv('../taxon_lineages.csv')
host_tax_to_phage = pd.read_csv('phage_host_taxon_id.csv')
taxon_lin = taxon_lin.merge(host_tax_to_phage, left_on='taxon_id', right_on='host_taxon_id', how='left')

keep_phylum = ['Pseudomonadota', 'Actinomycetota', 'Bacillota', 'Cyanobacteriota', 'Bacteroidota']
phage_phylum = taxon_lin[['phage', 'PHYLUM']].rename(columns={'PHYLUM': 'phylum'})
