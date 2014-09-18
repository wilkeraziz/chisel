import jieba
import sys

for line in sys.stdin:
    print ' '.join(token.encode('utf8') for token in jieba.cut(line.strip(), cut_all = False) if token.strip())
