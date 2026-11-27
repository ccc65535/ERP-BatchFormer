from json import load
from re import split
import time
import warnings

from algorithm.ERP_BatchFormer import ERP_BatchFormer
from algorithm.deepnet import Deep4Net
from augmentation.base import augment
from lib.SDTW import SoftDTW
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold

from algorithm.util import *



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader

import random
from lib.loss_function import *
from lib.LoadSet import *
import pandas as pd
import copy
from augmentation import BGBank

def inba_k_fold(k,kfold,meta):

    meta_new=copy.deepcopy(meta)
    meta_new.insert(2,'dataset','None')

    for sub in range(1,26):

        non_tar_ind=meta[(meta['subject']==sub) &( meta['class']==0)].index
        tar_ind=meta[(meta['subject']==sub) &( meta['class']==1)].index

        ##
        seg0=int(len(non_tar_ind)/kfold)
        st,ed=int(k*seg0),int((k+1)*seg0)

        test_ind=non_tar_ind[st:ed]
        rest_ind=[i for i in non_tar_ind if i not in test_ind]
        train_ind=rest_ind[:-seg0]
        valid_ind=rest_ind[-seg0:]
        
        meta_new.loc[train_ind,'dataset']='train'
        meta_new.loc[valid_ind,'dataset']='valid'
        meta_new.loc[test_ind,'dataset']='test'

        ##
        seg1=int(len(tar_ind)/kfold)
        st,ed=int(k*seg1),int((k+1)*seg1)

        test_ind=tar_ind[st:ed]
        rest_ind=[i for i in tar_ind if i not in test_ind]
        train_ind=rest_ind[:-seg1]
        valid_ind=rest_ind[-seg1:]

        meta_new.loc[train_ind,'dataset']='train'
        meta_new.loc[valid_ind,'dataset']='valid'
        meta_new.loc[test_ind,'dataset']='test'

    return meta_new




    

x_dtype, y_dtype = torch.float, torch.long


durations = [.4,.5,.6,.7]
device_id=0
device = torch.device("cuda:{:d}".format(device_id) if torch.cuda.is_available() else "cpu")
seed=42

subjects=range(25)
srate = 100


# eeg_all,meta=load_BrainInv()
# np.save('./P3_data.npy',eeg_all)
# meta.to_csv('P3_meta.csv',index=False)

eeg_all=np.load('./P3_data.npy')
meta_all=pd.read_csv('P3_meta.csv')
y=meta_all['class'].values
subs=meta_all['subject'].values-1


res_file_name='./record/P3-multifan-deep-bgbank-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
res_file=pd.ExcelWriter(res_file_name)

res_file_name2='./record/P3-multifan-deep-bgbank-f1-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
res_file2=pd.ExcelWriter(res_file_name2)

nclass=2

for duration in durations:
    
    t=int(srate*duration)


    eeg=eeg_all[:,:,:t]

    ntrial,nchannel,nsample=eeg.shape
   



    loo_global_accs = []
    loo_global_model_states = []
    loo_fine_tuning_accs = []
    loo_fine_tuning_f1=[]

    kfold=5

    for k in range(kfold):
        meta=inba_k_fold(k,kfold,meta_all)

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
            torch.backends.cudnn.benchmark = False
            # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
            torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        

        
        train_ind=meta.loc[meta['dataset']=='train'].index
        valid_ind=meta.loc[meta['dataset']=='valid'].index
        test_ind=meta.loc[meta['dataset']=='test'].index

        
        trainX, validateX, testX=eeg[train_ind],eeg[valid_ind],eeg[test_ind]
        trainY, validateY, testY=y[train_ind],y[valid_ind],y[test_ind]
        trainSub, validateSub, testSub =subs[train_ind],subs[valid_ind],subs[test_ind]
            

        templates=np.zeros((nchannel,nsample,nclass,len(subjects)))
        # templates2=np.zeros((nchannel,nsample,nclass,len(subjects)))

        for sub in subjects:
            inds=np.where(trainSub==sub)[0]
            tar_trials=[]
            non_tar_trials=[]
            for ind in inds:
                if trainY[ind]==0:
                    non_tar_trials.append(trainX[ind])
                else:
                    tar_trials.append(trainX[ind])

            n_tar=len(tar_trials)
            n_nontar=len(non_tar_trials)

            templates[:,:,1,sub]=np.mean(tar_trials,axis=0)
            templates[:,:,0,sub]=np.mean(non_tar_trials,axis=0)

            


        trainX, validateX, testX = generate_tensors(
            trainX, validateX, testX, dtype=x_dtype)
        trainY, validateY, testY = generate_tensors(
            trainY, validateY, testY, dtype=y_dtype)
        trainSub, validateSub, testSub = generate_tensors(
            trainSub, validateSub, testSub, dtype=y_dtype)

        ######
        batch_size,max_epochs,lr = 256,600,1e-3
        alpha=1e-3


        train_dataset=TensorDataset(
                trainX,
                trainY,
                trainSub
            )
        train_data_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        
        )

        valid_dataset=TensorDataset(
                validateX,
                validateY,
                validateSub,
        )
        valid_data_loader=DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True
        
        )  



        hid_dim=nchannel*4
        


        all_save_path='./model_save/multifan-deepnet-mod-fold-'+str(k)+'-'+str(duration)+'s-P3.pth'

        loss_fun1=nn.CrossEntropyLoss()
        loss_fun2=SoftDTW(gamma=0.5)
        
       



        last_save=0
        ## fine-tuning
        sub_accs = []
        sub_f1=[]
        for sub_id in subjects:

            # sub_id=sub_id+1
            print(f'fold{k},sub{sub_id}')

            train_sub_ind=meta.loc[(meta['subject']==sub_id+1) &(meta['dataset']=='train')].index
            validate_sub_ind=meta.loc[(meta['subject']==sub_id+1) &(meta['dataset']=='valid')].index
            test_sub_ind=meta.loc[(meta['subject']==sub_id+1) &(meta['dataset']=='test')].index
            
            
            sub_trainX, sub_trainY = eeg[train_sub_ind], y[train_sub_ind]
            sub_validateX, sub_validateY = eeg[validate_sub_ind], y[validate_sub_ind]
            sub_testX, sub_testY = eeg[test_sub_ind], y[test_sub_ind]

            temp_sub=templates[:,:,:,sub_id]
            n_aug=4
            bg=[]

            for i,label in enumerate(sub_trainY):
                temp=temp_sub[:,:,label]
                bg.append(sub_trainX[i]-temp)
            bg=np.array(bg)
            # bg_bank=np.zeros(bg.shape)
            bg_bank=bg

            for i,func in enumerate(['TimeReverse','FTSurrogate']):
                bg_t,_=augment(func,bg,np.ones(bg.shape[0]),p=1,srate=srate)
                # if i>0:
                bg_bank=np.concatenate((bg_bank,bg_t),axis=0)
                # else:
                #     bg_bank=bg_t


            gen_x,gen_y=BGBank.BGBank(sub_trainX,sub_trainY,templates[:,:,:,sub_id],bg_bank,n_aug)
            sub_trainX=torch.cat((torch.tensor(sub_trainX),torch.tensor(gen_x)),dim=0)
            sub_trainY=np.concatenate((sub_trainY,gen_y),axis=0)

            # gen_x,gen_y=BGBank.BGBank(sub_validateX,sub_validateY,templates[:,:,:,sub_id],bg_bank,8)
            # sub_validateX=torch.cat((sub_validateX,torch.tensor(gen_x)),dim=0)
            # sub_validateY=np.concatenate((sub_validateY,gen_y),axis=0)
            
            sub_testX=torch.tensor(sub_testX,dtype=x_dtype).to(device)
            # sub_testY=torch.tensor(sub_testY,dtype=torch.long).to(device)



            batch_size,max_epochs,lr = 64,600,1e-3
            # lamda=0.8
            patience=20

            sub_train_dataset=TensorDataset(
                torch.tensor(sub_trainX,dtype=x_dtype),
                torch.tensor(sub_trainY,dtype=torch.long),
            )
            sub_train_data_loader=DataLoader(
                sub_train_dataset,
                batch_size=batch_size,
                shuffle=True
            
            )  

                        
            sub_valid_dataset=TensorDataset(
                torch.tensor(sub_validateX,dtype=x_dtype),
                torch.tensor(sub_validateY,dtype=torch.long),
            )
            sub_valid_data_loader=DataLoader(
                sub_valid_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            

            base_model=Deep4Net(
                n_chans=nchannel, 
                n_times=nsample, 
                n_outputs=nclass,
                filter_time_length=10,
                filter_length_2=5,
                filter_length_3=5,
                filter_length_4=5,
                pool_time_length=5,
                pool_time_stride=1,
                drop_prob=0.2).to(device)

            sub_model=ERP_BatchFormer(
                nchannel, int(srate*duration), nclass,
                hid_dim=256,
                base_model=base_model,
                dropout_rate=0.1
            ).to(device)
            
            sub_model.load_state_dict(torch.load(all_save_path))
            

   
            optimizer=optim.Adam(sub_model.parameters(), lr=lr,weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

            max_acc=0
            min_loss=np.inf
            save_path='./model_save/p3-deep-multifan-mod-bgbank-'+str(duration)+'s-fold-'+str(k)+'-sub-'+str(sub_id)+'.pth'

            for epoch in range(max_epochs):

                sub_model.train()
                total_loss=0
                new_templates=torch.zeros((nchannel,nsample,nclass)).to(device)
                
                for i, data in enumerate(sub_train_data_loader):

                    # lamda = np.random.beta(8,2)

                    x,label=data
                    x,label=x.to(device),label.to(device)
                    # sub_model.train()
                    
                    # temp=torch.tensor([templates[:,:,:,sub_id] for l in label]).to(device)
                    # temp=temp.transpose(1,3)
                    # temp=temp.transpose(2,3)
                    temp=torch.tensor([templates[:,:,l,sub_id] for l in label]).to(device)
                    # templates_batch=np.zeros((nchannel,nsample,nclass))
                    
                    # for yy in range(nclass):
                    #     ind=torch.where(label==yy)[0]
                    #     if len(ind)==0:
                    #         templates_batch[:,:,yy]=templates[:,:,yy,sub_id]
                    #     templates_batch[:,:,yy]=torch.mean(x[ind,:,:],dim=0).detach().cpu().numpy()
                    
                    # temp2=torch.tensor([templates_batch[:,:,l] for l in label]).to(device)
                  
                    out,template_feature=sub_model(x)
                    # out=sub_model(x)

                
                    loss1=loss_fun1(out,label)
                    # loss2=loss_fun2(template_feature,temp,label)
                    loss2=loss_fun2(template_feature,temp)
                    # loss3= sub_model.soft_parameter_regularization_loss()
                    # inner_varient,inter_varient = sub_model.center_similarity(template_feature,label)
                    
                    loss=loss1+alpha*loss2
                    
                    total_loss+=loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(sub_model.parameters(), 0.5)
                    optimizer.step()

                    if epoch==10 :
                        unique_labels, inverse_indices = torch.unique(label, return_inverse=True)
                        for l in unique_labels:
                            ind=torch.where(label==l)
                            new_templates[:,:,l]+=torch.sum(template_feature[ind],dim=0)

                if epoch==10:

                    non_template=new_templates.detach().cpu()[:,:,0]
                    tar_template=new_templates.detach().cpu()[:,:,1]

                    non_template=non_template/np.std(non_template.numpy())*np.std(templates[:,:,0,sub_id])
                    tar_template=tar_template/np.std(tar_template.numpy())*np.std(templates[:,:,1,sub_id])

                    templates[:,:,0,sub_id]=non_template
                    templates[:,:,1,sub_id]=tar_template

                sub_model.eval()
                with torch.no_grad():
                    valid_loss=0
                    valid_pred_labels=[]
                    valid_true_labels=[]
                    for i, data in enumerate(sub_valid_data_loader):
                        torch.cuda.empty_cache()
                        # sub_model.eval()

                        x,label=data
                        x,label=x.to(device),label.to(device)

                        # temp=torch.tensor([templates[:,:,:,sub_id] for l in label]).to(device)
                        # temp=temp.transpose(1,3)
                        # temp=temp.transpose(2,3)
                        temp=torch.tensor([templates[:,:,l,sub_id] for l in label]).to(device)
                  
                        out,template_feature=sub_model(x)
                        # out=sub_model(x)
                
                        loss1=loss_fun1(out,label)
                        loss2=loss_fun2(template_feature,temp)
                        # loss2=loss_fun2(template_feature,temp,label)
                        
                        loss=loss1+alpha*loss2

                        valid_loss+=loss
                        valid_pred_labels+=list(out.argmax(axis=1).detach().cpu().numpy())

                        valid_true_labels+=list(label.detach().cpu().numpy())

                    del x,label
    
                    valid_acc = balanced_accuracy_score(valid_true_labels, valid_pred_labels)

                    sub_model.eval()

                    output,_= sub_model(sub_testX)
                    
                    pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                    true_labels = sub_testY
                    sub_acc = balanced_accuracy_score(true_labels,pred_labels )

                    valid_loss/=len(sub_valid_data_loader)

                    print(f'epoch:{epoch},train loss:{total_loss/(len(sub_train_data_loader)):.3f},valid loss:{valid_loss:.4f},valid acc:{valid_acc:.3f},test acc:{sub_acc:.3f}')

                    if (valid_loss<min_loss)or (valid_acc>max_acc):

                        torch.save(sub_model.state_dict(),save_path)
                        print('save model.')
                        if (valid_loss<min_loss):
                            min_loss=valid_loss
                        if (valid_acc>max_acc):
                            max_acc=valid_acc
                        last_save=epoch

                        
                        

                    if last_save+patience<epoch:
                        print('early stop')
                        break

                    

            state_dict = torch.load(save_path, map_location=device)
            sub_model.load_state_dict(state_dict)
            sub_model.eval()

            
            # test_temp=torch.tensor([templates[:,:,:,sub_id] for l in range(sub_testX.size(0))]).to(device)
            output,_ = sub_model(sub_testX)
            pred_labels=output.argmax(axis=1).detach().cpu().numpy()

            true_labels = sub_testY
            sub_acc = balanced_accuracy_score(true_labels,pred_labels)
            f1=f1_score(true_labels,pred_labels)
            
            sub_accs.append(sub_acc)
            sub_f1.append(f1)

        loo_fine_tuning_accs.append(sub_accs)
        loo_fine_tuning_f1.append(sub_f1)

    global_sub_accs = np.array(loo_global_accs).T
    ft_sub_accs = np.array(loo_fine_tuning_accs).T

    pd.DataFrame(ft_sub_accs).to_excel(excel_writer=res_file,sheet_name=str(duration), index=False, header=False)
    res_file._save()

    pd.DataFrame(np.array(loo_fine_tuning_f1).T).to_excel(excel_writer=res_file2,sheet_name=str(duration), index=False, header=False)
    res_file2._save()

      