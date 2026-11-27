import numpy as np
import random


def BGBank(x,y,templates,bg_bank,n_aug):
    n_batch,nch,nsamp=x.shape
    gen_x=[]
    gen_y=[]
    for label in y:
        for n in range(n_aug):
            temp=templates[:,:,label]
            bg=bg_bank[np.random.randint(0,bg_bank.shape[0]),:,:]
            gen_x.append(temp+bg)
            gen_y.append(label)
    return np.array(gen_x),np.array(gen_y)

def BGBank_P3(x,y,templates,bg_bank,n_aug):
    n_batch,nch,nsamp=x.shape
    gen_x=[]
    gen_y=[]
    gen_temp=[]
    for i,label in enumerate(y):
        for n in range(n_aug):
            temp=templates[i]
            bg=bg_bank[np.random.randint(0,bg_bank.shape[0]),:,:]
            gen_x.append(temp+bg)
            gen_y.append(label)
            gen_temp.append(temp)
    return np.array(gen_x),np.array(gen_y),np.array(gen_temp)


    
