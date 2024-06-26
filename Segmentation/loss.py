import torch
from torch.autograd import Function
import torch.nn.functional as F
import math
import torch.nn as nn
'''
def weigthed_l1(x,y,b):
    a = math.exp(-81/b)
    #a = b/140;
    mask = (y[:,1:2,:,:]>a).double()/(1e-12+y[:,1:2,:,:])+(y[:,1:2,:,:]<=a).double()/a
    msk = torch.cat([torch.ones_like(mask),mask],dim = 1)
    return torch.mean(torch.abs((x - y)*msk))
'''
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def weigthed_l1_attention(x,y,b):    
    #return torch.mean((x[:,0,:,:]-y[:,0,:,:])**2/(1e-12+x[:,2,:,:])+torch.log(1e-12+torch.abs(x[:,2,:,:])))+torch.mean(x[:,2,:,:]*(x[:,1,:,:]-y[:,1,:,:])**2)
    return torch.sqrt(torch.mean((x[:,0,:,:]-y[:,0,:,:])**2/(1e-12+x[:,2,:,:])+torch.log(1e-12+torch.abs(x[:,2,:,:]))))
def weigthed_l1_b1(x,y,b):    
    return torch.mean(torch.abs(x[:,0,:,:]-y[:,0,:,:]))
def weigthed_l1_t2(x,y,b):    
    return torch.mean(torch.abs(x[:,1,:,:]-y[:,1,:,:]))

def grad_loss(x,y,device='cuda'):
    mean = 0
    cx = [[[[1, -1]]]];
    cy = [[[[1],[-1]]]];
    cx = torch.FloatTensor(cx).to(device=device, dtype=torch.float32)
    cy = torch.FloatTensor(cy).to(device=device, dtype=torch.float32)
    for i in range(0,x.shape[1]):
        x1 = x[:,i:i+1,:,:]
        y1 = y[:,i:i+1,:,:]
        xx = F.conv2d(x1,cx,padding=1)
        xy = F.conv2d(x1,cy,padding=1)
        yx = F.conv2d(y1,cx,padding=1)
        yy = F.conv2d(y1,cy,padding=1)
        mean += 0.5*(torch.mean(torch.abs(xx - yx))+torch.mean(torch.abs(xy - yy)))
    return mean

def relaxmat(M,num_states,E2,device):
    E1 = 0.9970
    M_recon = torch.zeros_like(M).to(device=device, dtype=torch.float32)
    E2_ov = torch.unsqueeze(E2,2)
    M_recon[:,:,0,:] = E2_ov*M[:,:,1,:]
    for x in range(1,num_states):
        M_recon[:,:,3*x-1,:] = E1*M[:,:,3*x-1,:]
        M_recon[:,:,3*x-2,:] = E2_ov*M[:,:,3*x+1,:]
        M_recon[:,:,3*x,:] = E2_ov*M[:,:,3*x-3,:]
    M_recon[:,:,3*num_states-1,:] = E1*M[:,:,3*num_states-1,:]
    return M_recon

def complex_matmul(A_r,A_i,b_r,b_i):
    r=torch.matmul(A_r,b_r)-torch.matmul(A_i,b_i)
    i=torch.matmul(A_i,b_r)+torch.matmul(A_r,b_i)
    return r,i

def flipmat(alpha,num_pulses,refcon,device):
    alpha = alpha*refcon/180
    T_1_r=torch.stack((torch.stack((torch.cos(alpha/2)**2,torch.sin(alpha/2)**2,torch.zeros_like(alpha)),2),torch.stack((torch.sin(alpha/2)**2,torch.cos(alpha/2)**2,torch.zeros_like(alpha)),2),torch.stack((torch.zeros_like(alpha),torch.zeros_like(alpha),torch.cos(alpha)),2)),3)
    T_1_i=torch.stack((torch.stack((torch.zeros_like(alpha),torch.zeros_like(alpha),-0.5*torch.sin(alpha)),2),torch.stack((torch.zeros_like(alpha),torch.zeros_like(alpha),0.5*torch.sin(alpha)),2),torch.stack((-torch.sin(alpha),torch.sin(alpha),torch.zeros_like(alpha)),2)),3)
        
    return T_1_r.to(device=device, dtype=torch.float32),T_1_i.to(device=device, dtype=torch.float32)

def flipmat_ext(M,num_pulses,angle,refcon,device):
    alpha = angle*refcon/180
    T_1_r=torch.stack((torch.stack((torch.cos(alpha/2)**2,torch.sin(alpha/2)**2,torch.zeros_like(alpha)),2),torch.stack((torch.sin(alpha/2)**2,torch.cos(alpha/2)**2,torch.zeros_like(alpha)),2),torch.stack((torch.zeros_like(alpha),torch.zeros_like(alpha),torch.cos(alpha)),2)),3)
    T_1_i=torch.stack((torch.stack((torch.zeros_like(alpha),torch.zeros_like(alpha),-0.5*torch.sin(alpha)),2),torch.stack((torch.zeros_like(alpha),torch.zeros_like(alpha),0.5*torch.sin(alpha)),2),torch.stack((-torch.sin(alpha),torch.sin(alpha),torch.zeros_like(alpha)),2)),3)
    M_recon = torch.zeros_like(M).to(device=device, dtype=torch.float32)
    for x in range(1,num_pulses+1):
        M_recon[:,:,3*x-3:3*x,0:1],M_recon[:,:,3*x-3:3*x,1:2] = complex_matmul(T_1_r,T_1_i,M[:,:,3*x-3:3*x,0:1],M[:,:,3*x-3:3*x,1:2])
    return M_recon

def EPGdecaycurve(ETL,excite_angle,flip_angle,E2,refcon,device):
    M=torch.zeros(E2.shape[0],E2.shape[1],3*ETL,2).to(device=device, dtype=torch.float32)
    M[:,:,0,0]=E2*torch.sin(excite_angle)
    echo_amp=torch.zeros(E2.shape[0],E2.shape[1],ETL).to(device=device, dtype=torch.float32)
    T_1_r, T_1_i = flipmat(flip_angle*(math.pi/180),ETL,refcon,device)
    M[:,:,0:3,0:1],M[:,:,0:3,1:2] = complex_matmul(T_1_r,T_1_i,M[:,:,0:3,0:1],M[:,:,0:3,1:2])
    echo_amp[:,:,0]=torch.sqrt(M[:,:,1,0]**2+M[:,:,1,1]**2+1e-12)*E2;
    for x in range(1,ETL):
        M=relaxmat(M,ETL,torch.pow(E2,2),device)
        M =flipmat_ext(M,ETL,flip_angle*(math.pi/180),refcon,device)
        echo_amp[:,:,x]=torch.sqrt(M[:,:,1,0]**2+M[:,:,1,1]**2+1e-12)*E2
    return echo_amp

def EPG_SLR(ETL,B1,E2,refcon,mxy_180,mxy_90,device):
    [mm, nn]=mxy_180.shape
    epg_i = EPGdecaycurve(ETL,math.asin(mxy_90[0,0])*B1,180*mxy_180[0,0]*B1,E2,refcon,device)
    for ii in range(1,nn):
        epg_i = epg_i + EPGdecaycurve(ETL,mxy_90[0, ii]*B1,180*mxy_180[0, ii]*B1,E2,refcon,device)
    return epg_i[:,:,ETL-1]/(epg_i[:,:,0]+1e-12);
    #return epg_i[:,:,:]

def TSEreadout(echo_data,ETL,device):
    k_space_PD_B1 = torch.zeros_like(echo_data[:,:,:,0:2]).to(device=device, dtype=torch.float32)
    k_space_T2_B1 = torch.zeros_like(echo_data[:,:,:,0:2]).to(device=device, dtype=torch.float32)
    NPE = int(echo_data.shape[2])
    sta = 0
    for tf in range(0,2*ETL):
        k_space = torch.rfft(echo_data[:,:,:,tf],2,onesided=False)
        if tf < ETL:
            if tf == 0:
                shots = 11
            elif tf == 1:
                shots = 19
            else:
                shots = 22
            k_space_PD_B1[:,sta:sta+shots,:,:]=k_space[:,sta:sta+shots,:,:]
            k_space_PD_B1[:,NPE-sta-shots:NPE-sta,:,:]=k_space[:,NPE-sta-shots:NPE-sta,:,:]
            sta = sta + shots
        else:
            if 2*ETL - tf-1 == 0:
                shots = 11
            elif 2*ETL - tf-1 == 1:
                shots = 19
            else:
                shots = 22
            sta = sta - shots;
            k_space_T2_B1[:,sta:sta+shots,:,:]=k_space[:,sta:sta+shots,:,:]
            k_space_T2_B1[:,NPE-sta-shots:NPE-sta,:,:]=k_space[:,NPE-sta-shots:NPE-sta,:,:]
    return k_space_PD_B1, k_space_T2_B1


def model_loss(out,b1_t2,mxy_180,mxy_90,ETL=5,te=9,device='cuda',refcon = 180):
    b1 = torch.matmul(b1_t2[:,18:19],torch.ones_like(b1_t2[0:1,0:18]))
    e2 = torch.exp(-0.5*te/b1_t2[:,0:18])
    sig = EPG_SLR(2*ETL,b1+1e-12,e2+1e-12,refcon,mxy_180,mxy_90,device)
    return torch.mean(torch.abs(sig-out))
'''
    PDw_k,T2w_k = TSEreadout(torch.unsqueeze(out[:,2,:,:]+1e-16,3)*EPG_SLR(2*ETL,out[:,0,:,:]+1e-16,torch.pow(out[:,1,:,:]+1e-16,te/100),refcon,mxy_180,mxy_90,device),ETL,device)
    PDw = torch.ifft(PDw_k,2)
    T2w = torch.ifft(T2w_k,2)
    return 0.5*(torch.mean(torch.abs(mask*(torch.sqrt(PDw[:,:,:,0]**2+PDw[:,:,:,1]**2+1e-12)-tse[:,0,:,:])))+torch.mean(torch.abs(mask*(torch.sqrt(T2w[:,:,:,0]**2+T2w[:,:,:,1]**2+1e-12)-tse[:,1,:,:]))))'''
    
def model_loss_woProfile(out,tse,mask,mxy_180,mxy_90,ETL=5,te=9,device='cuda',refcon = 180):
    PDw_k,T2w_k = TSEreadout(torch.unsqueeze(out[:,2,:,:]+1e-16,3)*EPGdecaycurve(2*ETL,out[:,0,:,:]*math.pi/2+1e-16,180*out[:,0,:,:]+1e-16,torch.pow(out[:,1,:,:]+1e-16,te/100),refcon,device),ETL,device)
    return 0.5*(torch.mean(torch.abs(mask*(torch.irfft(PDw_k,2,onesided=False)-tse[:,0,:,:])))+torch.mean(torch.abs(mask*(torch.irfft(T2w_k,2,onesided=False)-tse[:,1,:,:]))))


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        #data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        #data_range = data_range[:, None, None, None]
        data_range = 1
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        self.w = self.w.cuda()
        #print(torch.sum(self.w))
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S