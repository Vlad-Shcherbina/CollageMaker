import subprocess
import re

p = subprocess.Popen(
    'xsel --clipboard --input', shell=True, stdin=subprocess.PIPE)


def load_module(name):
    return '##### {} #####\n{}\n\n'.format(name, open(name).read())

text = ''
text += load_module('img_lib.py')
text += load_module('sol.py')

text = re.sub(r'(from img_lib import \*)', r'#\1', text)
text = text.replace('__main__', '__mian__')

p.communicate(text)
ret = p.wait()
assert ret == 0
print('solution copied to clipboard')
