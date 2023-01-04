#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

def filltemplate(templatestring, notreeopt=None, perbcopt=None, 
                 kernelopt=None, ompopt=None):
    if notreeopt is None:
        raise ValueError('notreeopt must be specified')
    if perbcopt is None:
        raise ValueError('perbcopt must be specified')
    if kernelopt is None:
        raise ValueError('kernelopt must be specified')
    if ompopt is None:
        raise ValueError('ompopt must be specified')
    
    fillkw = {}
    _mfname = 'Makefile_v3'
    _namepart = ''

    # order of options matters for resulting filenames
    # getting that right is important for the executables
    if notreeopt:
        fillkw.update({'pyfill_addnotree': 'OPTIONS += -DNOTREE'})
        _mfname = _mfname + '_nt'
        _namepart = _namepart + '_notree'
    else:
        fillkw.update({'pyfill_addnotree': ''})
    if kernelopt == 'C2':
        fillkw.update({'pyfill_kernelopt': 'C2'})
        _mfname = _mfname + '_c2'
        _namepart = _namepart + '_C2'
    elif kernelopt == 'GADGET':
        fillkw.update({'pyfill_kernelopt': 'GADGET'})
        _mfname = _mfname + '_gd'
        _namepart = _namepart + '_gadget'
    if perbcopt:
        fillkw.update({'pyfill_addperiodic': 'OPTIONS += -DPERIODIC'})
        _mfname = _mfname + '_perbc'
        _namepart = _namepart + '_perbc'
    else:
        fillkw.update({'pyfill_addperiodic': ''})
    if ompopt:
        fillkw.update({'pyfill_addopenmp': 'ADDOPENMP += $(DOOPENMP)'})
        _mfname = _mfname + '_omp'
        _namepart = _namepart + '_omp'
    else:
        fillkw.update({'pyfill_addopenmp': ''})
    
    
    fillkw.update({'pyfill_namepart': _namepart})
    _mfcontent = templatestring.format(**fillkw)
    if not os.path.isdir('./makefiles_fromtemplate'):
        os.mkdir('./makefiles_fromtemplate')
    if not os.path.isdir('./makefiles_debug'):
        os.mkdir('./makefiles_debug')
    with open('./makefiles_fromtemplate/' + _mfname, 'w') as f:
        f.write(_mfcontent)

def runfilloop(templatestring):
    for notreeopt in [True, False]:
        for perbcopt in [True, False]:
            for kernelopt in ['GADGET', 'C2']:
                for ompopt in [True, False]:
                    filltemplate(templatestring,
                                 notreeopt=notreeopt,
                                 perbcopt=perbcopt,
                                 kernelopt=kernelopt,
                                 ompopt=ompopt)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filen_template = sys.argv[1]
    else:
        filen_template = './_template_makefiles'
    with open(filen_template, 'r') as f:
        templatestring = f.read()
    runfilloop(templatestring)
