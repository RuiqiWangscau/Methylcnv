# -*- coding: utf-8 -*-
"""Discover include paths from an installed compiler; never install packages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--env',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
prefix=Path(args.env)
candidates=list((prefix/'bin').glob('*-conda-linux-gnu-c++'))
candidates+=list((prefix/'bin').glob('*-conda-linux-gnu-g++'))
system=shutil.which('g++')
if system:
    candidates.append(Path(system))
records=[]
for compiler in candidates:
    probe=subprocess.run([str(compiler),'-std=c++17','-E','-x','c++','-','-v'],
        input='#include <vector>\n#include <compare>\n#include <features.h>\n',
        text=True,capture_output=True,timeout=60)
    records.append({'compiler':str(compiler),'returncode':probe.returncode,'stderr':probe.stderr})
    if probe.returncode:
        continue
    capture=False
    paths=[]
    for line in probe.stderr.splitlines():
        if '#include <...> search starts here:' in line:
            capture=True
        elif 'End of search list.' in line:
            capture=False
        elif capture:
            directory=Path(line.strip())
            if directory.is_dir():
                paths.append(str(directory))
    if paths:
        Path(args.output).write_text(':'.join(paths)+'\n',encoding='utf-8')
        Path(args.output+'.diagnostics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('Compiler includes discovered:',compiler)
        break
else:
    Path(args.output+'.diagnostics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    raise SystemExit('No installed compiler provides usable C++/system headers. Environment repair is needed; see diagnostics JSON.')
