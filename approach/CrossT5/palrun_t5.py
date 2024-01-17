import subprocess
import time
import sys
lst = []
for i in range(2):
    p = subprocess.Popen(['python3 approach/RQ2_T5Encoder_Transformer_Attention-based_CpyNet_No_Emb.py %d'%i], shell=True)
    time.sleep(1)
    lst.append(p)
for x in lst:
    x.wait()