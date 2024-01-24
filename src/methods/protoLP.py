# Adaptation of the publicly available code of the paper entitled "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning":
# https://github.com/yhu01/PT-MAP
from tqdm import tqdm
import torch
from src.utils import get_one_hot
import time
import math
from src.utils import Logger, extract_features
import numpy as np
import torch.nn.functional as F


def centerDatas(datas):
    datas = datas - datas.mean(1, keepdim=True)
    datas = datas / torch.norm(datas, dim=2, keepdim=True)

    return datas


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1), 'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas


def SVDreduction(ndatas, K):
    # ndatas = torch.linear.qr(datas.permute(0, 2, 1),'reduced').R
    # ndatas = ndatas.permute(0, 2, 1)
    _, s, v = torch.svd(ndatas)
    ndatas = ndatas.matmul(v[:, :, :K])

    return ndatas


# def predict(gamma, Z, labels):
#     # #Certainty_scores = 1 + (Z*torch.log(Z)).sum(dim=2) / math.log(5)
#     # Z[:,:n_lsamples].fill_(0)
#     # Z[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
#     Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
#     Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
#     #tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
#     delta = torch.sum(Z, 1)
#     #L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
#     iden = torch.eye(5,device='cuda')
#     iden = iden.reshape((1, 5, 5))
#     iden = iden.repeat(10000, 1, 1)
#     W = torch.bmm(torch.transpose(Z,1,2), Z/delta.unsqueeze(1))
#     #W = W/W.sum(1).unsqueeze(1)
#     #isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W, dim=-1,keepdim=True))
#     # checknan(laplacian=isqrt_diag)
#     #W = W * isqrt_diag[:, None, :] * isqrt_diag[:, :, None]
#     #W = W * isqrt_diag * torch.transpose(isqrt_diag,dim0=2,dim1=1)
#     L = iden - W#(W + W.bmm(W))/2
#     Z_l = Z[:,:n_lsamples]
#
#     #A = np.dot(np.linalg.inv(torch.matmul(torch.transpose(Z_l,1,2), Z_l) + gamma * L), torch.bmm(torch.transpose(Z_l,1,2), Y))
#     u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*iden)
#     A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
#     Pred = Z.bmm(A)
#     normalizer = torch.sum(Pred,dim=1,keepdim=True)
#     # #normalizer = Pred[:,:n_lsamples].max(dim=1)[0].unsqueeze(1)
#     Pred = (n_shot+n_queries)*Pred/normalizer
#     # normalizer = torch.sum(Pred, dim=2, keepdim=True)
#     # Pred = Pred/normalizer
#     # Pred[:, :n_lsamples].fill_(0)
#     # Pred[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
#     # N = PredZ.shape[0]
#     # K = PredZ.shape[1]
#     # pred = np.zeros((N, K))
#     #
#     # for k in range(K):
#     #     current_pred = np.dot(Z, A[:, k])
#
#     return Pred#.clamp(0,1)

def predictW(gamma, Z, labels, n_runs, n_lsamples, n_usamples, n_ways, n_shot, n_queries):
    # print(n_lsamples, n_usamples, n_queries)
    # print(labels.shape)
    Y = get_one_hot(labels).cuda()
    # Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    # Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
    delta = torch.sum(Z, 1)
    L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
    Z_l = Z[:,:n_lsamples]

    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    P = Z.bmm(A)
    # P = F.normalize(P, dim=2, p=1)
    _, n, m = P.shape
    r = torch.ones(n_runs, n_lsamples + n_usamples,device='cuda')
    c = torch.ones(n_runs, n_ways,device='cuda') * (n_shot + n_queries)
    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    while torch.max(torch.abs(u - P.sum(2))) > 0.01:
        u = P.sum(2)
        #normalizing rows (probability)
        P *= (r / u).view((n_runs, -1, 1))
        # print(P[0,0])
        # print(P[0,0].sum())
        # #normalizing class samples per class
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        # print(P[0,0].sum())
        P[:,:n_lsamples].fill_(0)
        P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        if iters == maxiters:
            break
        iters = iters + 1
    return P


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways


# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, device, n_ways, lam, balancing):
        self.device = device
        self.mus = None  # shape [n_runs][n_ways][n_nfeat]
        self.n_ways = n_ways
        self.lam = lam
        self.balancing = balancing

    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()

    def initFromLabelledDatas(self, data, n_tasks, shot, n_ways, n_queries, n_nfeat):
        self.mus_ori = data.reshape(n_tasks, shot+n_queries,n_ways, n_nfeat)[:,:shot,].mean(1)
        self.mus = self.mus_ori.clone()
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)

    def initFromCenter(self, mus):
        # self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:1,].mean(1)
        self.mus = mus
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)
        # self.mus_ori = torch.randn(n_runs, n_ways,n_nfeat,device='cuda')
        # self.mus_ori = self.mus_ori/self.mus_ori.norm(dim=2,keepdim=True)
        # self.mus = self.mus_ori.clone()

    def updateFromEstimate(self, estimate, alpha, l2=False):

        diff = self.mus_ori - self.mus
        Dmus = estimate - self.mus
        if l2 == True:
            self.mus = self.mus + alpha * (Dmus) + 0.01 * diff
        else:
            self.mus = self.mus + alpha * (Dmus)
        # self.mus/=self.mus.norm(dim=2, keepdim=True)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        if self.balancing=='dirichlet':
            P = F.normalize(P, dim=2, p=1)
        elif self.balancing=='balanced':
            P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
        # P = F.normalize(P, dim=2, p=1)
            u = torch.zeros(n_runs, n).cuda()
            maxiters = 1000
            iters = 1
        # normalize this matrix
            while torch.max(torch.abs(u - P.sum(2))) > epsilon:
                u = P.sum(2)
                P *= (r / u).view((n_runs, -1, 1))
                P *= (c / P.sum(1)).view((n_runs, 1, -1))
                if iters == maxiters:
                    break
                iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self, data, y_s, n_tasks, n_sum_query, n_queries, shot):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (data.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_tasks, n_sum_query)
        c = torch.ones(n_tasks, self.n_ways) * n_queries

        n_lsamples = self.n_ways * shot

        # Query probabilities
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test

        # Support probabilities
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, y_s.unsqueeze(2), 1)
        return p_xj, p_xj_test



    def estimateFromMask(self, mask, ndatas):
        emus = mask.permute(0, 2, 1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))
        return emus


class PROTOLP(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.n_ways = args.n_ways
        self.number_tasks = args.batch_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.PLC = args.PLC
        self.lam = args.lam
        self.n_queries = args.n_query
        self.n_sum_query = args.n_query * args.n_ways
        self.n_epochs = args.n_epochs
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.balancing = args.balanced

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []

    def getAccuracy(self, preds_q, y_q):
        preds_q = preds_q.argmax(dim=2)

        acc_test = (preds_q == y_q).float().mean(1, keepdim=True)
        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(self.number_tasks)
        return m, pm

    def get_GaussianModel(self):
        method_info = {'device': self.device, 'lam': self.lam, 'n_ways': self.n_ways, 'balancing': self.balancing}
        return GaussianModel(**method_info)

    def power_transform(self, support, query):
        """
            inputs:
                support : torch.Tensor of shape [n_task, s_shot, feature_dim]
                query : torch.Tensor of shape [n_task, q_shot, feature_dim]

            outputs:
                support : torch.Tensor of shape [n_task, s_shot, feature_dim]
                query : torch.Tensor of shape [n_task, q_shot, feature_dim]
        """
        support[:,] = torch.pow(support[:,] + 1e-6, self.beta)
        query[:,] = torch.pow(query[:,] + 1e-6, self.beta)
        return support, query

    def centerData(self, datas):
        # PT code
        #    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
        #   datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
        # centre of mass of all data support + querries
        datas[:, :] -= datas[:, :].mean(1, keepdim=True)  # datas[:, :, :] -
        norma = torch.norm(datas[:, :, :], 2, 2)[:, :, None].detach()
        datas[:, :, :] /= norma

        return datas

    def scaleEachUnitaryDatas(self, datas):

        norms = datas.norm(dim=2, keepdim=True)
        return datas / norms

    def QRreduction(self, data):

        ndatas = torch.qr(data.permute(0, 2, 1)).R
        ndatas = ndatas.permute(0, 2, 1)
        return ndatas

    def SVDreduction(self, datas, K):
        # ndatas = torch.linear.qr(datas.permute(0, 2, 1),'reduced').R
        # ndatas = ndatas.permute(0, 2, 1)
        _, s, v = torch.svd(datas)
        ndatas = datas.matmul(v[:, :, :K])

        return ndatas
    def PT(self, datas, beta=0.5):
        # ------------------------------------PT-MAP-----------------------------------------------
        nve_idx = np.where(datas.cpu().detach().numpy() < 0)
        datas[nve_idx] *= -1
        datas[:, ] = torch.pow(datas[:, ] + 1e-6, beta)
        datas[nve_idx] *= -1  # return the sign
        return datas

    def centerDatas(self, datas):
        # PT code
        #    datas[:, :] = datas[:, :, :] - datas[:, :].mean(1, keepdim=True)
        #   datas[:, :] = datas[:, :, :] / torch.norm(datas[:, :, :], 2, 2)[:, :, None]
        # centre of mass of all data support + querries
        datas[:, :] -= datas[:, :].mean(1, keepdim=True)  # datas[:, :, :] -
        norma = torch.norm(datas[:, :, :], 2, 2)[:, :, None].detach()
        datas[:, :, :] /= norma

        return datas

    def performEpoch(self, model, data, y_s, y_q, shot, epochInfo=None):

        p_xj, _ = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
                                        n_queries=self.n_queries, shot=shot)
        self.probas = p_xj
        m_estimates = model.estimateFromMask(self.probas, data)
        model.updateFromEstimate(m_estimates, self.alpha)

        m_estimates = model.estimateFromMask(self.probas, data)
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

    def run_adaptation(self, model, data, y_s, y_q, shot):

        # _, preds_q = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
        #                                   n_queries=self.n_queries, shot=shot)

        # print("Initialisation model accuracy", self.getAccuracy(preds_q=preds_q, y_q=y_q))
        self.logger.info(' ==> Executing PT-MAP adaptation on {} shot tasks ...'.format(shot))
        t0 = time.time()
        for epoch in tqdm(range(self.n_epochs)):
            self.performEpoch(model=model, data=data, y_s=y_s, y_q=y_q, shot=shot, epochInfo=(epoch, self.n_epochs))

            p_xj, _ = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
                                             n_queries=self.n_queries, shot=shot)
            self.probas = p_xj
            pesudo_L = predictW(0.05, self.probas, y_s, self.number_tasks, shot*self.n_ways, self.n_ways*self.n_queries, self.n_ways, shot, self.n_queries)
            beta = 0.7
            m_estimates = model.estimateFromMask((beta*pesudo_L + (1-beta)*p_xj).clamp(0,1), data)
            model.updateFromEstimate(m_estimates, self.alpha)
            _, preds_q = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
                                             n_queries=self.n_queries, shot=shot)            # acc = self.getAccuracy(op_xj)
            self.record_info(y_q=y_q, pred_q=preds_q, new_time=time.time()-t0)
            t0 = time.time()

    def record_info(self, y_q, pred_q, new_time):
        """
        inputs:
            y_q : torch.Tensor of shape [n_tasks, q_shot]
            q_pred : torch.Tensor of shape [n_tasks, q_shot]:
        """
        pred_q = pred_q.argmax(dim=2)
        self.test_acc.append((pred_q == y_q).float().mean(1, keepdim=True))
        self.timestamps.append(new_time)

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'acc': self.test_acc}

    def run_task(self, task_dic, shot, train_loader=None, reductor=None):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']
        train_mean = task_dic['train_mean'].unsqueeze(0).unsqueeze(0)
        # print(task_dic['train_mean'].shape)
        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = x_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        # Extract features
        support, query = extract_features(self.model, support, query)
        # print(support.shape, query.shape)
        if self.PLC:
            batch, n_sup, dim = support.shape
            datas = torch.cat([support, query], dim=1)
            datas = self.PT(datas)
            datas = self.scaleEachUnitaryDatas(datas)
            datas = self.SVDreduction(datas, 40)
            datas = self.centerData(datas)
            support, query = datas[:,:n_sup], datas[:,n_sup:]
        else:
            support = F.normalize(support, dim=2)  #- train_mean.cuda()#
            query = F.normalize(query, dim=2)  #- train_mean.cuda()#
            datas = torch.cat([support, query], dim=1)
            datas = self.SVDreduction(datas, 40)
            # datas = self.centerData(datas)
            support, query = datas[:,:shot*self.n_ways], datas[:,shot*self.n_ways:]


        # Run adaptation
        data = torch.cat([support, query], dim=1)
        gaus_model = self.get_GaussianModel()
        gaus_model.initFromLabelledDatas(data=data, n_tasks=self.number_tasks,shot=shot, n_ways=self.n_ways, n_queries=self.n_queries, n_nfeat=data.size(2))
        self.run_adaptation(model=gaus_model, data=data, y_s=y_s, y_q=y_q, shot=shot)
        # Extract adaptation logs
        logs = self.get_logs()

        return logs