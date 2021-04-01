        output = model(to_device(input)) #student输出
        output_teacher = model_teacher(to_device(input)) #teacher输出


        # rois_label_t=output['cls_target']

        '''
        Classification and regression distillation in RPN:
        '''
        if cfg_distillation.get('cls_distillation_rpn',None):
            cfg_cls_distillation = cfg_distillation.get('cls_distillation_rpn')
            rpn_cls_score_t = output_teacher['RoINet.cls_pred']
            rpn_cls_score_s = output['RoINet.cls_pred']
            RPN_loss_cls_s = output['RoINet.cls_loss']
            start_mu=cfg_cls_distillation.get('start_mu')
            end_mu=cfg_cls_distillation.get('end_mu')
            mu=start_mu+(end_mu-start_mu)*(float(epoch)/max_epoch)
            loss_rpn_cls, loss_rpn_cls_soft = compute_loss_classification(rpn_cls_score_t, rpn_cls_score_s, mu,
                                                                          RPN_loss_cls_s, T=1, weighted=True)
            # loss_rcn_cls, loss_rcn_cls_soft = compute_loss_classification(rcn_cls_score_t, rcn_cls_score_s, mu,
            #
            output['RoINet.cls_loss']=loss_rpn_cls

        if cfg_distillation.get('loc_distillation_rpn', None):
            cfg_loc_distillation = cfg_distillation.get('loc_distillation_rpn')
            RCNN_loss_bbox_s = output['RoINet.loc_loss']
            bbox_pred_s = output['RoINet.loc_pred']
            bbox_pred_t = output_teacher['RoINet.loc_pred']
            rpn_rois_target_s = output['RoINet.loc_target']
            rpn_rois_target_t = output_teacher['RoINet.loc_target']

            start_ni = cfg_loc_distillation.get('start_ni')
            end_ni = cfg_loc_distillation.get('end_ni')
            ni = start_ni + (end_ni - start_ni) * (float(epoch) / max_epoch)
            loss_rpn_reg, loss_rpn_reg_soft, _, _ = \
                compute_loss_regression(RCNN_loss_bbox_s, bbox_pred_s, bbox_pred_t, rpn_rois_target_s, rpn_rois_target_t,
                                        m=0.01, ni=ni)
            # loss_rcn_cls, loss_rcn_cls_soft = compute_loss_classification(rcn_cls_score_t, rcn_cls_score_s, mu,
            #                                                               RCNN_loss_cls_s, T=1, weighted=True)
            output['RoINet.loc_loss'] = loss_rpn_reg


        '''
        Classification and regression distillation in rcnn:
        '''
        if cfg_distillation.get('cls_distillation',None):
            cfg_cls_distillation = cfg_distillation.get('cls_distillation')
            rcn_cls_score_t = output_teacher['cls_pred']
            rcn_cls_score_s = output['cls_pred']
            RCNN_loss_cls_s = output['BboxNet.cls_loss']
            start_mu=cfg_cls_distillation.get('start_mu')
            end_mu=cfg_cls_distillation.get('end_mu')
            mu=start_mu+(end_mu-start_mu)*(float(epoch)/max_epoch)
            loss_rcn_cls, loss_rcn_cls_soft = compute_loss_classification(rcn_cls_score_t, rcn_cls_score_s, mu,
                                                                          RCNN_loss_cls_s, T=1, weighted=True)
            # loss_rcn_cls, loss_rcn_cls_soft = compute_loss_classification(rcn_cls_score_t, rcn_cls_score_s, mu,
            #
            output['BboxNet.cls_loss']=loss_rcn_cls

        if cfg_distillation.get('loc_distillation',None):
            cfg_loc_distillation=cfg_distillation.get('loc_distillation')
            RCNN_loss_bbox_s=output['BboxNet.loc_loss']
            bbox_pred_s=output['loc_pred']
            bbox_pred_t=output_teacher['loc_pred']
            rois_target_s=output['loc_target']
            rois_target_t=output_teacher['loc_target']

            start_ni=cfg_loc_distillation.get('start_ni')
            end_ni=cfg_loc_distillation.get('end_ni')
            ni=start_ni+(end_ni-start_ni)*(float(epoch)/max_epoch)
            loss_rcn_reg, loss_rcn_reg_soft,_,_ = \
                compute_loss_regression(RCNN_loss_bbox_s, bbox_pred_s, bbox_pred_t,rois_target_s, rois_target_t, m=0.01, ni=ni)
            # loss_rcn_cls, loss_rcn_cls_soft = compute_loss_classification(rcn_cls_score_t, rcn_cls_score_s, mu,
            #                                                               RCNN_loss_cls_s, T=1, weighted=True)
            output['BboxNet.loc_loss'] = loss_rcn_reg

        '''
        Feature level distillation:
        '''
        # sup_loss = (torch.pow(sup_feature - stu_feature_adap, 2) * mask_batch).sum() / norms
        # sup_loss = sup_loss * args.imitation_loss_weigth
        if cfg_distillation.get('feature_distillation', None):
            cfg_feature_distillation=cfg_distillation.get('feature_distillation')
            sup_feature=output_teacher['features'][0]
            stu_feature=output['features'][0]
            stu_feature_adap=model_adap(stu_feature)


            start_weigth=cfg_feature_distillation.get('start_weigth')
            end_weigth=cfg_feature_distillation.get('end_weigth')
            imitation_loss_weigth = start_weigth + (end_weigth - start_weigth) * (float(epoch) / max_epoch)
            if cfg_feature_distillation.get('need_mask', None):
                mask_batch = output_teacher['RoINet.mask_batch']
                mask_list = []
                for mask in mask_batch:
                    mask = (mask > 0).float().unsqueeze(0)
                    mask_list.append(mask)
                mask_batch = torch.stack(mask_list, dim=0)
                # print('sum:%d' %  mask_batch.sum(), flush=True)
                norms = mask_batch.sum() ** 2
                # print('norms:%d'%norms,flush=True)
                sup_loss = (torch.pow(sup_feature - stu_feature_adap, 2) * mask_batch).sum() / norms

                # print('sup_loss:%f' % sup_loss,flush=True)
                if sup_loss>100 or sup_loss<-100:
                    print('sup_loss:%f' % sup_loss, flush=True)
                    sup_loss=sup_loss*0.000001

                # raise RuntimeError('Stop')
            else:
                sup_loss = (torch.pow(sup_feature - stu_feature_adap, 2)).sum()

            # imitation_loss_weigth=0.0001

            sup_loss = sup_loss * imitation_loss_weigth
            output['sup.loss']=sup_loss
