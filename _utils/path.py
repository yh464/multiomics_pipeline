# -*- coding: utf-8 -*-
#!/usr/env/bin python3

'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
2025-01-20

This utility manages the nomenclature of file names / paths
'''

import os, re, json
from fnmatch import fnmatch
import numpy as np
import pandas as pd
from .logger import logger
log = logger()

class project():
    def __init__(self, root_dir = os.path.realpath('..')):
        self.project_root = os.path.realpath(root_dir)
        os.makedirs(f'{self.project_root}/.path', exist_ok = True) # hidden folder to store path config
        self.config_file = f'{self.project_root}/.path/path_config.json'
        if not os.path.isfile(self.config_file):
            self.config = {
                'raw': f'{self.project_root}/raw/%dataset%/%prefix%.h5ad', # raw data directory will be scanned
                'normalised': f'{self.project_root}/normalised/%dataset%/%prefix%.h5ad', # normalised data will also be scanned
            }
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent = 4)
        else:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)

        self.progress_file = f'{self.project_root}/.path/progress.txt'
        if not os.path.isfile(self.progress_file):
            self.progress = self.scan_h5ad()
            other_files = list(self.config.keys())
            other_files.remove('raw'); other_files.remove('normalised')
            self.progress[other_files] = False
            self.progress.to_csv(self.progress_file, sep = '\t', index = True, header = True)
        else:
            self.progress = pd.read_csv(self.progress_file, sep = '\t', header = 0, index_col = ['dataset','prefix'])

    def __del__(self): self.save() # ensure all updates to the progress and config files are saved

    def save(self): 
        self.progress.to_csv(self.progress_file, sep = '\t', index = True, header = True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent = 4)

    def scan_h5ad(self): # special function to scan for original and normalised h5ad files
        raw_list = []
        raw_pattern = self.config['raw'].replace('%dataset%','*').replace('%prefix%','*.h5ad')
        for root, _, files in os.walk(f'{self.project_root}/raw'):
            for file in files:
                full_path = os.path.join(root, file)
                if fnmatch(full_path, raw_pattern):
                    raw_list.append((os.path.basename(root), file)) # (dataset, prefix) tuple

        norm_list = []
        norm_pattern = self.config['normalised'].replace('%dataset%','*').replace('%prefix%','*.h5ad')
        for root, _, files in os.walk(f'{self.project_root}/normalised'):
            for file in files:
                full_path = os.path.join(root, file)
                if fnmatch(full_path, norm_pattern):
                    norm_list.append((os.path.basename(root), file)) # (dataset, prefix) tuple
        
        out_df = pd.DataFrame(index = pd.MultiIndex.from_tuples(list(set(raw_list + norm_list)), names = ['dataset', 'prefix']),
            columns = ['raw', 'normalised'], data = False)
        out_df.loc[raw_list, 'raw'] = True
        out_df.loc[norm_list, 'normalised'] = True
        return out_df
    
    def scan_all(self):
        # scan all files and update the progress table
        out_df = self.scan_h5ad()
        other_files = list(self.config.keys())
        other_files.remove('raw'); other_files.remove('normalised')
        for dataset, prefix in out_df.index:
            for ftype in other_files:
                pattern = self.config[ftype].replace('%dataset%', dataset).replace('%prefix%', prefix)
                if os.path.isfile(pattern):
                    out_df.loc[(dataset, prefix), ftype] = True
                elif any([fnmatch(file, pattern) for file in os.listdir(os.path.dirname(pattern))]): # in case path contains wildcards
                    out_df.loc[(dataset, prefix), ftype] = True
                else:
                    out_df.loc[(dataset, prefix), ftype] = False
        self.progress = out_df
        return out_df

    def find_h5ad(self, *datasets, normalised = False, long = True, exclude = []):
        out = []
        if len(datasets) == 0: return []
        if type(datasets[0]) in [list, tuple]:
            datasets = [y for x in datasets for y in x]
        
        dirname = f'{self.project_root}/{"normalised" if normalised else "raw"}'
        for p in sorted(datasets):
            xlist = []
            pdir = p.split('/')[0]
            if p.find('/') < 0: ppat = '*'
            elif p.find('^') >= 0: ppat = p.split('/')[1].replace('^','') # strict matching
            else: ppat = '*' + p.split('/')[1] + '*'
            for x in sorted(os.listdir(f'{dirname}/{pdir}')):
                if not fnmatch(x.replace('.gz',''), f'{ppat}.h5ad'): continue
                exc = False
                for y in exclude:
                    if fnmatch(pdir,'*'+y.split('/')[0]+'*') and fnmatch(x, '*'+y.split('/')[1]+'*'):
                        exc = True; break
                if exc: continue
                xlist.append(x.replace('.h5ad','').replace('.gz',''))
            if len(out) == 0 or pdir != out[-1][0]: out.append((pdir, xlist))
            else: out[-1] = (pdir, sorted(out[-1][1] + xlist))
        out_long = [(x,z) for x,y in out for z in y]
        log.log(f'Found {len(out_long)} h5ad files', calling_file = 'path/find_h5ad')
        for x, y in out_long:
            log.log(f'    {x}/{y}', calling_file = 'path/find_h5ad')

        if long: return out_long
        else: return out

    def _to_long_format(self, *datasets):
        if not isinstance(datasets[0], list) and not isinstance(datasets[0][0], tuple): 
            datasets = self.find_h5ad(datasets) # cmdline input into 'datasets'
        elif isinstance(datasets[0][0][1], list):
            datasets = [(x, z) for x,y in datasets[0] for z in y] # coerse into long format
        elif isinstance(datasets[0][0], tuple):
            datasets = datasets[0] # already in long format
        else: raise ValueError('Unrecognised input dataset names')
        return datasets
    
    def find(self, ftype, *datasets):
        # finds files from subsequent processing steps based on given dataset and prefix names
        datasets = self._to_long_format(*datasets)
        out = []
        for dataset, prefix in datasets:
            if ftype not in self.config:
                raise ValueError(f'File type {ftype} not found in path config')
            pattern = self.config[ftype].replace('%dataset%', dataset).replace('%prefix%', prefix)
            out.append(os.path.isfile(pattern))
            self.progress.loc[(dataset, prefix), ftype] = out[-1]
        log.log(f'Found {sum(out)} / {len(out)} files for processing step {ftype}', calling_file = 'path/find')
        return out, datasets
    
    def complete_step(self, ftype, dataset, prefix):
        self.progress.loc[(dataset, prefix), ftype] = True
        self.progress.to_csv(self.progress_file, sep = '\t', index = True, header = True)
    
    def to_pathname(self, ftype, *datasets):
        datasets = self._to_long_format(*datasets)
        out = []
        for dataset, prefix in datasets:
            if ftype not in self.config:
                raise ValueError(f'File type {ftype} not found in path config')
            pattern = self.config[ftype].replace('%dataset%', dataset).replace('%prefix%', prefix)
            out.append(pattern)
        return out

    def register(self, ftype, pattern, force = False):
        pattern = os.path.realpath(pattern)
        if ftype in self.config and pattern != self.config[ftype] and not force:
            raise ValueError(f'File type {ftype} already exists in path config. Use force = True to overwrite.')
        if '%dataset%' not in pattern or '%prefix%' not in pattern:
            raise ValueError('Pattern must contain %dataset% and %prefix% placeholders.')
        self.config[ftype] = pattern
        self.progress[ftype] = False
        self.save()