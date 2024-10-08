import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import subprocess
import concurrent.futures

def processPandH(p):
    folder_path = 'C:/Program Files/NCBI/blast-2.14.1+/bin'
    #outpath = 'C:/Users/18134/Desktop/Research/SimPhageHost/' + p + h + '.txt'
    outpath = 'C:/Users/18134/Desktop/Research/SimPhagePhagedecEskapee/' + p + '.txt'
    inpath = 'C:/Users/18134/Desktop/Research/all_phage_files_dec/' + p + '.fna'
    #subpath = 'C:/Users/18134/Desktop/Research/all_host_files/' + h + '.fna'
    # batch_script = "run_as_admin.bat"
    # blast = "blasttest.bat"
    # print(folder_path)
    # Specify the command you want to run
    '''print("in", inpath)
    print("subpath", subpath)
    print("out", outpath)
    print("folPath", folder_path)'''

    #command = 'blastn -query ' + inpath + ' -subject ' + subpath + ' -out ' + outpath + ' -outfmt 6 -max_target_seqs 1 -max_hsps 1 -word_size 11 -evalue 10'
    #command = 'blastn -query ' + inpath + ' -db ' + "allphages" + ' -out ' + outpath + ' -outfmt 6 -max_target_seqs 2 -max_hsps 1 -word_size 11 -evalue 0.000001 -reward 1 -penalty -2 -gapopen 0 -gapextend 0 -perc_identity 90 -num_threads 4'
    command = 'blastn -query ' + inpath + ' -db ' + "eskapeephage" + ' -out ' + outpath + ' -outfmt 6 -max_target_seqs 2 -max_hsps 1 -word_size 11 -evalue 10 -reward 1 -penalty -2 -gapopen 0 -gapextend 0 -perc_identity 90 -num_threads 4'
    # command = %BLASTN_PATH% -query query.txt -db ddna -out outpath -outfmt 6 -evalue 0.00001 -max_target_seqs 1
    #print(command)
    # Run the command in the specified folder

    # result = subprocess.run([batch_script, command], shell=True, cwd=folder_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = subprocess.run(command, shell=True, cwd=folder_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    print(result)

phages=pd.read_csv('eskapee.csv',header=None,sep=',').iloc[:,0]
hosts=pd.read_csv('eskapee.csv',header=None,sep=',').iloc[:,1]

if __name__ == '__main__':
    output = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for p in phages:
            #for h in hosts:
                executor.submit(processPandH,p)
