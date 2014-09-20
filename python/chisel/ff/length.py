"""
Length features: capture information about the length (in number of tokens) of a complete hypothesis.

Usage:

    * this module requires no configuration

    * the following features are defined

        1. FLength: len(source)
        2. ELength: len(target)
        3. LengthRatio: len(target) / len(source) (with no sanity checks)
        4. LengthDiff: len(target) - len(source)
        5. AbsLenDiff: abs(len(target) - len(source))

@author waziz
"""
import chisel

@chisel.ff.features('FLength', 'ELength', 'LengthRatio', 'LengthDiff')
def ELength(hypothesis): 
    f, e = hypothesis.source_.split(), hypothesis.translation_.split()
    return (len(f), len(e), float(len(e))/len(f), len(e) - len(f))

@chisel.ff.dense
def AbsLenDiff(hypothesis): 
    # we could use suffstats to avoid splitting text twice (in ELength and here)
    # however, split is pretty fast and I wanted to have an example using @ff.dense ;)
    f, e = hypothesis.source_.split(), hypothesis.translation_.split()
    return abs(len(e) - len(f))

if __name__ == '__main__':
    print __doc__
