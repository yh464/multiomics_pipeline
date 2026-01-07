import os, sys
sys.path.insert(0,'/rds/project/rb643/rds-rb643-ukbiobank2/Data_Users/yh464/scripts')
from _utils.slurm import array_submitter
os.chdir(os.path.dirname(os.path.realpath(__file__)))

out_dir = '/rds/project/rds-Nl99R8pHODQ/multiomics/gsmap'

submitter = array_submitter(n_cpu = 40, timeout = 720, name = 'preprocess_sc_4gsmap',
    wd = os.path.dirname(os.path.realpath(__file__)))

for x in os.listdir():
    if not os.path.isdir(x): continue
    for y in os.listdir(x):
        if y[-5:] != '.h5ad': continue
        prefix = f'{x}_{y[:-5]}'
        out_file = f'{out_dir}/{prefix}/generate_ldscore/'
        if os.path.isdir(out_file):
            out_file += f'{prefix}_chunk{len(os.listdir(out_file))-4}/{prefix}.9.l2.ldscore.feather'
            if os.path.isfile(out_file): continue
        cmd = f'bash preprocess_sc_4gsmap.sh {prefix}'
        submitter.add(cmd)
submitter.submit()