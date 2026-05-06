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
                'raw': 'raw/$dataset/$prefix.h5ad', # raw data directory will be scanned
                'normalised': 'normalised/$dataset/$prefix.h5ad', # normalised data will also be scanned
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

    def save(self): 
        self.progress.to_csv(self.progress_file, sep = '\t', index = True, header = True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent = 4)

    def scan_h5ad(self): # special function to scan for original and normalised h5ad files
        raw_list = []
        raw_pattern = f'{self.project_root}/' + self.config['raw'].replace('$dataset','*').replace('$prefix','*')
        for dataset in os.listdir(f'{self.project_root}/raw'):
            if not os.path.isdir(f'{self.project_root}/raw/{dataset}'): continue
            for prefix in os.listdir(f'{self.project_root}/raw/{dataset}'):
                full_path = os.path.join(self.project_root, 'raw', dataset, prefix)
                if fnmatch(full_path, raw_pattern) and full_path.endswith('.h5ad'):
                    raw_list.append((dataset, prefix.replace('.h5ad','').replace('.gz',''))) # (dataset, prefix) tuple

        norm_list = []
        norm_pattern = f'{self.project_root}/' + self.config['normalised'].replace('$dataset','*').replace('$prefix','*')
        for dataset in os.listdir(f'{self.project_root}/normalised'):
            if not os.path.isdir(f'{self.project_root}/normalised/{dataset}'): continue
            for prefix in os.listdir(f'{self.project_root}/normalised/{dataset}'):
                full_path = os.path.join(self.project_root, 'normalised', dataset, prefix)
                if fnmatch(full_path, norm_pattern) and full_path.endswith('.h5ad'):
                    norm_list.append((dataset, prefix.replace('.h5ad','').replace('.gz',''))) # (dataset, prefix) tuple
        
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
                pattern = f'{self.project_root}/' + self.config[ftype].replace('$dataset', dataset).replace('$prefix', prefix)
                # replace all placeholders up to the next '.' or '_' character with wildcards '*'
                while pattern.find('$') >= 0:
                    start = pattern.find('$')
                    end_dot = pattern.find('.', start)
                    end_underscore = pattern.find('_', start)
                    if end_dot < 0: end_dot = len(pattern)
                    if end_underscore < 0: end_underscore = len(pattern)
                    end = min(end_dot, end_underscore)
                    pattern = pattern[:start] + '*' + pattern[end:]
                if os.path.exists(pattern):
                    out_df.loc[(dataset, prefix), ftype] = True
                elif os.path.exists(os.path.dirname(pattern)) and \
                    any([fnmatch(file, os.path.basename(pattern)) for file in os.listdir(os.path.dirname(pattern))]): # in case path contains wildcards
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
        else: log.error('Unrecognised input dataset names')
        return datasets
    
    def find(self, ftype, *datasets):
        # finds files from subsequent processing steps based on given dataset and prefix names
        datasets = self._to_long_format(*datasets)
        out = []
        for dataset, prefix in datasets:
            if ftype not in self.config:
                log.error(f'File type {ftype} not found in path config')
            pattern = f'{self.project_root}/' + self.config[ftype].replace('$dataset', dataset).replace('$prefix', prefix)
            out.append(os.path.isfile(pattern))
            self.progress.loc[(dataset, prefix), ftype] = out[-1]
        log.log(f'Found {sum(out)} / {len(out)} files for processing step {ftype}', calling_file = 'path/find')
        return out, datasets
    
    def complete_step(self, ftype, dataset, prefix):
        self.progress.loc[(dataset, prefix), ftype] = True
        self.progress.to_csv(self.progress_file, sep = '\t', index = True, header = True)
    
    def to_pathname(self, ftype, dataset, prefix, **kwargs):
        if ftype not in self.config:
            log.error(f'File type {ftype} not found in path config')
        pattern = f'{self.project_root}/' + self.config[ftype].replace('$dataset', dataset).replace('$prefix', prefix)
        for key, value in kwargs.items():
            pattern = pattern.replace(f'${key}.', str(value)+'.').replace(f'${key}_', str(value)+'_')
            pattern = '/'.join([part if part != f'${key}' else str(value) for part in pattern.split('/')])
        os.makedirs(os.path.dirname(pattern), exist_ok = True) # automatically create directory when pathname is requested
        
        # if there are still placeholders, give a warning if file is not uniquely defined after replacing placeholders with wildcards
        missing_placeholders = []
        while pattern.find('$') >= 0:
            start = pattern.find('$')
            end_dot = pattern.find('.', start)
            end_underscore = pattern.find('_', start)
            end_slash = pattern.find('/', start)
            if end_dot < 0: end_dot = len(pattern)
            if end_underscore < 0: end_underscore = len(pattern)
            if end_slash < 0: end_slash = len(pattern)
            end = min(end_dot, end_underscore, end_slash)
            missing_placeholders.append(pattern[start:end])
            pattern = pattern[:start] + '*' + pattern[end:]
        if missing_placeholders:
            # search for files matching the pattern
            dir_pattern = os.path.dirname(pattern)
            files = [f for f in os.listdir(dir_pattern) if fnmatch(f, os.path.basename(pattern))]
            if len(files) > 1:
                log.warn(f'Variables are not fully defined for {pattern}: '+ ', '.join(missing_placeholders))
                log.warn(f'Found {len(files)} files matching the pattern: ')
                for f in files: log.warn(f'    {f}')
            elif len(files) == 1: pattern = os.path.join(dir_pattern, files[0])
        return pattern

    def to_pathname_multi(self, ftype, *datasets, **kwargs):
        datasets = self._to_long_format(*datasets)
        return [self.to_pathname(ftype, dataset, prefix, **kwargs) for dataset, prefix in datasets]

    def register(self, ftype, pattern, force = False):
        # pattern should be relative to the project root
        if pattern.startswith('/'):
            pattern = os.path.realpath(pattern)
            if not pattern.startswith(self.project_root):
                log.error('Provided absolute path is outside the project root directory.', calling_file = 'path/register')
            else: pattern = pattern.replace(f'{self.project_root}/', '')
        if ftype in self.config and pattern != self.config[ftype] and not force:
            log.error(f'File type {ftype} already exists in path config. Use force = True to overwrite.')
        elif ftype in self.config.keys() and pattern == self.config[ftype]: return
        if '$dataset' not in pattern or '$prefix' not in pattern:
            log.error('Pattern must contain $dataset and $prefix placeholders.')
        self.config[ftype] = pattern
        self.progress[ftype] = False
        self.save()