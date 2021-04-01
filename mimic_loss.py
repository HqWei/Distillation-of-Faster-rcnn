import torch.nn.functional as F
import torch


def compute_loss_classification(Z_t, Z_s, mu, L_hard, T=1, weighted = True):
    #vettori di pesi
    # if weighted:
    #     if torch.cuda.is_available():
    #         wc = torch.where((y == 0), 1.5 * torch.ones(Z_t.shape[0]).cuda(), torch.ones(Z_s.shape[0]).cuda()).double()
    #     else:
    #         wc = torch.where((y == 0), 1.5 * torch.ones(Z_t.shape[0]), torch.ones(Z_s.shape[0])).double()
    # else:
    wc = torch.ones(Z_s.shape[0]).cuda().float()
    Z_s = Z_s.float()
    Z_t = Z_t.float()

    P_t = F.softmax(Z_t /T, dim=1)
    P_s = F.softmax(Z_s /T, dim=1)

    try:
        #print('P_t & P_s:')
        #print(P_t.shape)
        #print(P_s.shape)
        P = torch.sum(P_t * torch.log(P_s), dim=1)# era P_s è e-10
    except:
        #print('Err:')
        #print(P_t.shape)
        #print(P_s.shape)
        mu=1
        L_soft = torch.zeros([1]).cuda().float()
        L_cls = L_hard.float()
        return L_cls, L_soft
        # raise RuntimeError('Stop!')
    L_soft = -torch.mean(P*wc)

    L_cls = mu * L_hard.float() + (1 - mu) * L_soft


    return L_cls, L_soft

# def compute_loss_classification(Z_t, Z_s, mu, L_hard, y, T=1, weighted = True):
#     #vettori di pesi
#     if weighted:
#         if torch.cuda.is_available():
#             wc = torch.where((y == 0), 1.5 * torch.ones(Z_t.shape[0]).cuda(), torch.ones(Z_s.shape[0]).cuda()).double()
#         else:
#             wc = torch.where((y == 0), 1.5 * torch.ones(Z_t.shape[0]), torch.ones(Z_s.shape[0])).double()
#     else:
#         wc = torch.ones(Z_s.shape[0]).cuda().double()
#     Z_s = Z_s.double()
#     Z_t = Z_t.double()
#
#     P_t = F.softmax(Z_t /T, dim=1)
#     P_s = F.softmax(Z_s /T, dim=1)
#
#
#     P = torch.sum(P_t * torch.log(P_s), dim=1)# era P_s è e-10
#
#     L_soft = -torch.mean(P*wc)
#
#     L_cls = mu * L_hard.double() + (1 - mu) * L_soft
#
#     return L_cls, L_soft

def compute_loss_regression(smooth_l1_loss, Rs, Rt, y_reg_s, y_reg_t , m,ni):

  s_box_diff = Rs - y_reg_s
  t_box_diff = Rt - y_reg_t
  # in_s_box_diff = bbox_inside_weights_s * s_box_diff
  # in_t_box_diff = bbox_inside_weights_t * t_box_diff
  # in_s_box_diff = in_s_box_diff * bbox_outside_weights_s
  # in_t_box_diff = in_t_box_diff * bbox_outside_weights_t
  in_s_box_diff =  s_box_diff
  in_t_box_diff =  t_box_diff

  in_s_bd_quad = in_s_box_diff.pow(2)
  in_t_bd_quad = in_t_box_diff.pow(2)
  norm_s = in_s_bd_quad
  norm_t = in_t_bd_quad
  dim= range(1,len(in_s_box_diff.shape))
  for i in sorted(dim, reverse=True):
      norm_s = norm_s.sum(i)
      norm_t = norm_t.sum(i)
  if torch.cuda.is_available():
      zeros = torch.zeros(norm_s.shape).cuda()
  else:
      zeros = torch.zeros(norm_s.shape)
  try:
      l_b = torch.where((norm_s + m <= norm_t), zeros, norm_s)
      l_reg =  smooth_l1_loss + ni * l_b.mean()
  except:
      l_reg = smooth_l1_loss
      l_b=torch.zeros([1]).cuda().float()


  return l_reg, l_b.mean(), norm_s.mean(), norm_t.mean()




