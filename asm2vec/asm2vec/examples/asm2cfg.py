from asm2vec.parse import parse_fp
from asm2vec.model import Asm2Vec
import pandas as pd

asmCodePath = '/home/cy/ccd/dataset/ASM/CISC/1/14.s'
with open(asmCodePath, 'r') as fp:
    print(fp)
    funcs = parse_fp(fp)
    print(funcs)
    print(len(funcs))

model = Asm2Vec(d=20)
trainingRepo = model.make_function_repo(funcs)
model.train(trainingRepo)

for tf in trainingRepo.funcs():
    
    # print('Norm of trained function "{}" = {}'.format(tf.sequential().name(), np.linalg.norm(tf.v)))
    # break
    print(tf.v)
    # df = pd.DataFrame(tf.v).T