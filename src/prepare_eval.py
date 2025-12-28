"""
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv -O heb-g2p-benchmark.tsv
uv run src/prepare_eval.py
"""


import pandas as pd
df = pd.read_csv('heb-g2p-benchmark.tsv', sep='\t', usecols=[0,1])
df.to_csv('eval.tsv', sep='\t', index=False, header=False)
