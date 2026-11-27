
import time
import warnings

from matplotlib import axis


from algorithm.ERP_BatchFormer import ERP_BatchFormer

from algorithm.deepnet import Deep4Net
from lib.SDTW import SoftDTW
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


from sklearn.metrics import balanced_accuracy_score, f1_score

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

def gen_ind_temps(X,Y,subs,subjects,n_avg=10):

    templates=np.zeros(X.shape)

    for sub in subjects:
        inds=np.where(subs==sub)[0]
        sub_X=X[inds]
        sub_Y=Y[inds]
        sub_template=templates[inds]

        tar_inds=np.where(sub_Y==1)[0]
        non_tar_inds=np.where(sub_Y==0)[0]

        # tar_trials,non_tar_trials=[],[]
        # tar_labels,non_tar_labels=[],[]
        # tar_temps,non_tar_temps=[],[]

        n_tar=len(tar_inds)
        n_nontar=len(non_tar_inds)

        for ii in range(0,n_tar,n_avg):
            if ii+n_avg<n_tar:
                seg_ind=tar_inds[ii:ii+n_avg]
            else:
                seg_ind=tar_inds[ii:]
            sub_template[seg_ind,:,:]=np.mean(sub_X[seg_ind],axis=0)
        
        for ii in range(0,n_nontar,n_avg):
            if ii+n_avg<n_nontar:
                seg_ind=non_tar_inds[ii:ii+n_avg]
            else:
                seg_ind=non_tar_inds[ii:]
            sub_template[seg_ind,:,:]=np.mean(sub_X[seg_ind],axis=0)

        templates[inds]=sub_template
    
    return templates


    

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


res_file_name='./record/P3-multifan-deep-mod-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
res_file=pd.ExcelWriter(res_file_name)

res_file_name2='./record/P3-multifan-deep-mod-f1-'+time.strftime('%m-%d-%H-%M')+'.xlsx'
res_file2=pd.ExcelWriter(res_file_name2)

nclass=2

for duration in durations:
    
    t=int(srate*duration)


    eeg=eeg_all[:,:,:t]
    eeg-=np.mean(eeg,axis=2,keepdims=True)

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
        
        train_templates=gen_ind_temps(trainX,trainY,trainSub,subjects)
        validate_templates=gen_ind_temps(validateX,validateY,validateSub,subjects)
        

        trainX, validateX, testX = generate_tensors(
            trainX, validateX, testX, dtype=x_dtype)
        trainY, validateY, testY = generate_tensors(
            trainY, validateY, testY, dtype=y_dtype)
        trainSub, validateSub, testSub = generate_tensors(
            trainSub, validateSub, testSub, dtype=y_dtype)
        
        train_templates, validate_templates = generate_tensors(
            train_templates, validate_templates, dtype=x_dtype)

        ######
        batch_size,max_epochs,lr = 256,600,1e-3
        alpha=1e-2
        beta=1e-2


        train_dataset=TensorDataset(
                trainX,
                trainY,
                trainSub,
                train_templates,
                
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
                validate_templates,
        )
        valid_data_loader=DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True
        
        )  



        hid_dim=nchannel*4
        
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
                drop_prob=0.5)
        
        all_model=ERP_BatchFormer(
            nchannel, int(srate*duration), nclass,
            hid_dim=256,
            base_model=base_model,
            dropout_rate=0.1
        ).to(device)




        all_save_path='./model_save/multifan-deepnet-mod-fold-'+str(k)+'-'+str(duration)+'s-P3.pth'

        loss_fun1=nn.CrossEntropyLoss()
        loss_fun2=SoftDTW(gamma=0.5)
        loss_fun3=nn.MSELoss()
        
        optimizer=optim.Adam(all_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

        patience=20
        min_loss=np.inf
        last_epoch=0
        a=.2

        for epoch in range(max_epochs):
            all_model.train()
            total_loss=0
            monitor_loss=0
            for i, data in enumerate(train_data_loader):

  

                x,label,subject,temp=data
                x,label,subject,temp=x.to(device),label.to(device),subject.to(device),temp.to(device)

               
                  
                out,template_feature=all_model(x)
                
                loss1=loss_fun1(out,label)
                loss2=loss_fun2(template_feature,temp)
                # loss3 = torch.abs(torch.sum(torch.std(temp,axis=2))-torch.sum(torch.std(template_feature,axis=2)))
                
                # threshold_=torch.mean(torch.std(temp,axis=2))
                # diff = torch.abs(template_feature[:, :, 1:] - template_feature[:, :, :-1]).mean(dim=2)
                # mask = (diff < threshold_).float()
                # safe_d = torch.clamp(diff, min=1e-4)
                # penalty = mask *  torch.sqrt(threshold_ / safe_d - 1)
                # loss4=penalty.mean()
                        
                loss=loss1+alpha*(loss2)
                
                total_loss+=loss
                # monitor_loss+=torch.sum(loss3)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_model.parameters(), 0.5)
                optimizer.step()

            all_model.eval()
            with torch.no_grad():
                valid_loss=0
                valid_pred_labels=[]
                valid_true_labels=[]
                for i, data in enumerate(valid_data_loader):
                    torch.cuda.empty_cache()

                    x,label,subject,temp=data
                    x,label,subject,temp=x.to(device),label.to(device),subject.to(device),temp.to(device)
                   
                  
                    out1,_=all_model(x)
                    

                    loss=loss_fun1(out1,label)
                
                

                    valid_loss+=loss
                    valid_pred_labels+=list(out1.argmax(axis=1).detach().cpu().numpy())

                    valid_true_labels+=list(label.detach().cpu().numpy())
                    # valid_true_labels+=list(l[:,0].detach().cpu().numpy())
                    

                valid_acc = balanced_accuracy_score(valid_true_labels, valid_pred_labels)

                all_model.eval()
                output,_= all_model(testX.to(device))
                pred_labels=output.argmax(axis=1).detach().cpu().numpy()

                true_labels = testY.numpy()
                acc = balanced_accuracy_score(true_labels,pred_labels )

                valid_loss/=len(valid_data_loader)

                pts=f'epoch:{epoch},train loss:{total_loss/(len(train_data_loader)):.3f},valid loss:{valid_loss:.4f},valid acc:{valid_acc:.3f},test acc:{acc:.3f}'
                print(pts)
                # print(f'\t std loss:{monitor_loss}')

                if valid_loss<min_loss:
                    last_epoch=epoch
                    min_loss=valid_loss
                
                if epoch-last_epoch>patience:
                    torch.save(all_model.state_dict(),all_save_path)
                    break



        last_save=0
        ## fine-tuning
        sub_accs = []
        sub_f1=[]
        for sub_id in subjects:

            # sub_id=sub_id+1
            print(f'fold{k},sub{sub_id}')

            train_sub_ind=np.where(trainSub==sub_id)[0]
            validate_sub_ind=np.where(validateSub==sub_id)[0]
            test_sub_ind=np.where(testSub==sub_id)[0]
            
            
            sub_trainX, sub_trainY,sub_train_template = trainX[train_sub_ind], trainY[train_sub_ind],train_templates[train_sub_ind]
            sub_validateX, sub_validateY,sub_validate_template = validateX[validate_sub_ind], validateY[validate_sub_ind],validate_templates[validate_sub_ind]
            sub_testX, sub_testY = testX[test_sub_ind], testY[test_sub_ind]

        
            
            sub_testX=torch.tensor(sub_testX,dtype=x_dtype).to(device)
            # sub_testY=torch.tensor(sub_testY,dtype=torch.long).to(device)



            batch_size,max_epochs,lr = 64,600,1e-3
            # lamda=0.8
            patience=20

            sub_train_dataset=TensorDataset(
                torch.tensor(sub_trainX,dtype=x_dtype),
                torch.tensor(sub_trainY,dtype=torch.long),
                torch.tensor(sub_train_template,dtype=x_dtype)
            )
            sub_train_data_loader=DataLoader(
                sub_train_dataset,
                batch_size=batch_size,
                shuffle=True
            
            )  

                        
            sub_valid_dataset=TensorDataset(
                torch.tensor(sub_validateX,dtype=x_dtype),
                torch.tensor(sub_validateY,dtype=torch.long),
                torch.tensor(sub_validate_template,dtype=x_dtype)
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
                dropout_rate=0.4
            ).to(device)
            sub_model.load_state_dict(torch.load(all_save_path))
            

   
            optimizer=optim.Adam(sub_model.parameters(), lr=lr,weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

            max_acc=0
            min_loss=np.inf
            save_path='./model_save/p3-multifan-deep-mod2-'+str(duration)+'s-fold-'+str(k)+'-sub-'+str(sub_id)+'.pth'

            for epoch in range(max_epochs):

                sub_model.train()
                total_loss=0
                monitor_loss=0
                new_templates=torch.zeros((nchannel,nsample,nclass)).to(device)
                
                for i, data in enumerate(sub_train_data_loader):

                    # lamda = np.random.beta(8,2)

                    x,label,temp=data
                    x,label,temp=x.to(device),label.to(device),temp.to(device)

                   
                  
                    out,template_feature=sub_model(x)
                    # out=sub_model(x)

                    loss1=loss_fun1(out,label)
     
                    loss2=loss_fun2(template_feature,temp)   
                    # loss3 = torch.abs(torch.sum(torch.std(temp,axis=2))-torch.sum(torch.std(template_feature,axis=2)))
                    # threshold_=torch.mean(torch.std(temp,axis=2))
                    # # loss4 = torch.abs(template_feature[:, :, 1:] - template_feature[:, :, :-1]).mean(dim=1)

                    # threshold_=torch.mean(torch.std(temp,axis=2))
                    # diff = torch.abs(template_feature[:, :, 1:] - template_feature[:, :, :-1]).mean(dim=2)
                    # mask = (diff < threshold_).float()
                    # safe_d = torch.clamp(diff, min=1e-8)
                    # penalty = mask *  torch.sqrt(threshold_ / safe_d - 1)
                    # loss4=penalty.sum()
                    
                    if epoch<10:
                        loss=loss1+alpha*(loss2)
                    else:
                        loss=loss1+0.1*alpha*(loss2)

                    # monitor_loss+=torch.sum(loss3)
                    total_loss+=loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(sub_model.parameters(), 0.5)
                    optimizer.step()

                    

                sub_model.eval()
                with torch.no_grad():
                    valid_loss=0
                    valid_pred_labels=[]
                    valid_true_labels=[]
                    for i, data in enumerate(sub_valid_data_loader):
                        torch.cuda.empty_cache()
                        # sub_model.eval()

                        x,label,temp=data
                        x,label,temp=x.to(device),label.to(device),temp.to(device)
                  
                        out,template_feature=sub_model(x)
                        # out=sub_model(x)
                
                        loss1=loss_fun1(out,label)
                                                           
                        loss2=loss_fun2(template_feature,temp)   
                        # loss3=loss_fun3(template_feature,temp) 
                        # loss3 = torch.abs(torch.sum(torch.std(temp,axis=2))-torch.sum(torch.std(template_feature,axis=2)))
                        # threshold_=torch.mean(torch.std(temp,axis=2))
                        # # loss4 = torch.abs(template_feature[:, :, 1:] - template_feature[:, :, :-1]).mean(dim=1)

                        # threshold_=torch.mean(torch.std(temp,axis=2))
                        # diff = torch.abs(template_feature[:, :, 1:] - template_feature[:, :, :-1]).mean(dim=2)
                        # mask = (diff < threshold_).float()
                        # safe_d = torch.clamp(diff, min=1e-8)
                        # penalty = mask *  torch.sqrt(threshold_ / safe_d - 1)
                        # loss4=penalty.sum()
                                
                        if epoch<10:
                            loss=loss1+alpha*(loss2)
                        else:
                            loss=loss1+0.1*alpha*(loss2)

            

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
                    # print(f'\t std loss:{monitor_loss}')
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

      