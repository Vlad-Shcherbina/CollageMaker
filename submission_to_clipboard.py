import sys
import subprocess
import re


def load_module(name):
    return '##### {} #####\n{}\n\n'.format(name, open(name).read())

text = ''
text += load_module('img_lib.py')
text += load_module('target_partition.py')
text += load_module('sol.py')

text = re.sub(r'(from \w+ import \*)', r'#\1', text)
text = text.replace('__main__', '__mian__')

if sys.platform == 'win32':
    p = subprocess.Popen(
        'clip', shell=True, stdin=subprocess.PIPE)
else:
    p = subprocess.Popen(
        'xsel --clipboard --input', shell=True, stdin=subprocess.PIPE)
p.communicate(text)
ret = p.wait()
assert ret == 0
print('solution copied to clipboard')
