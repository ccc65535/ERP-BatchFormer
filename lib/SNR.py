import numpy as np
def SNR(sig):
    n_trial = sig.shape[1]
    mean = np.mean(sig, axis=0, keepdims=True)
    # noise = sig - mean
    noise = np.std(sig, axis=0, keepdims=True)
    # noise = sig - mean

    # 信号能量=叠加平均的幅值平方
    # p_sig=np.square(mean)

    # 信号能量=erp信号的幅值平方
    p_sig = np.square(mean)

    # 噪声能量=噪声平方
    p_noise = np.square(noise)

    snr = np.mean((p_sig) / (p_noise))
    # snr = (p_sig) / (p_noise)
    snr=10*np.log10(snr)

    return snr

def SNR_trial(sig,template,N_t):

    mean=(template*N_t+sig)/(N_t+1)
    # noise = sig - mean
    noise = mean-sig
    # noise = sig - mean

    # 信号能量=叠加平均的幅值平方
    # p_sig=np.square(mean)

    # 信号能量=erp信号的幅值平方
    p_sig = np.square(mean)

    # 噪声能量=噪声平方
    p_noise = np.square(noise)

    ind_0=np.transpose(np.where(p_noise==0))
    # if len(ind_0)>0:
    #     ind_0
    #     for ind in ind_0:
    #         mask=np.ones(p_sig.shape)
    #         mask[ind[0],ind[1]]=False
    #         p_sig[mask]

    ratio=(p_sig) / (p_noise)
    ratio=ratio[ratio!=np.inf]
    snr = np.mean(ratio)
    # snr = (p_sig) / (p_noise)
    snr=10*np.log10(snr)

    return snr


def SNR_sub(sig):
    n_trial = sig.shape[0]
    n_label = sig.shape[1]
    n_ch = sig.shape[3]
    snr = []

    for label in range(n_label):
        for ch in range(n_ch):
            snr.append(SNR(sig[:, label, :, ch]))

    mean_snr = np.mean(snr)
    return mean_snr