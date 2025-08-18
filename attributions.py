import torch
import timm
import traceback

from utils.get_logger import logger
from utils.imagenet_misc import prediction_imagenet_a, prediction_imagenet_r

def get_attribution(xai, model, args):
        """
        Get attribution method for the model.
        Args:
            method (str): Name of the attribution method.
            model (torch.nn.Module): The model to be used for attribution.
        """
    
        from utils.iba_utils import iba_layers, save_estimator, load_estimator, postprocess_heatmap
        
        is_imagenet_a = 'imagenet-a' in args.data_path
        is_imagenet_r = 'imagenet-r' in args.data_path
        
        logger.info('[ImageNet] Is ImageNet-A: {}'.format(is_imagenet_a))
        logger.info('[ImageNet] Is ImageNet-R: {}'.format(is_imagenet_r))
        
        # Get config and input image size from timm config
        config = timm.data.resolve_data_config({}, model=model)
        args.input_size = config['input_size'][-1]
        
        logger.info(xai)
        
        if xai == 'iba':
            from IBA.iba import IBA
            
            #### Default settings for IBA
            # args.start_target = 6
            # args.end_target = 7
            # args.op_name = ['blocks']
            
            iba = IBA(args.network,
                    device=args.rank,
                    beta=args.beta,
                    optimization_steps=args.optim_steps,
                    lr=args.iba_lr,
                    st=args.start_index,
                    )
            
        elif xai == 'coiba':
            from IBA.coiba import CoIBA
            iba = CoIBA(model_name=args.network,
                    device=args.rank,
                    beta=args.beta,
                    optimization_steps=args.optim_steps,
                    lr=args.iba_lr,
                    st=args.start_index,
                    )
            
        # Setting adaptations for IBA & Insert IBA to the target layers
        args = iba_layers(args)
        iba.iba_preset(model, args.target_layer)
        
        if xai == 'iba':
            assert len(args.target_layer) == 1, 'IBA only supports single target layer'
        
        iba.reset_estimate()
        
        if args.load_estim:
            try:
                iba = load_estimator(args, args.target_layer, iba)
            except Exception as e:
                logger.error(traceback.format_exc())
                args.load_estim = False
                
        if not args.load_estim:
            if args.rank == 0:
                iba = save_estimator(args, args.target_layer, iba, model)
            try:
                torch.distributed.barrier()
            except:
                pass
            if args.rank != 0:
                iba = load_estimator(args, args.target_layer, iba)
                
        def model_loss_closure(x, target, is_imagenet_a=False, is_imagenet_r=False):
            
            pred = model(x)
            if is_imagenet_a:
                pred = prediction_imagenet_a(pred)
            elif is_imagenet_r:
                pred = prediction_imagenet_r(pred)
                
            loss = torch.nn.functional.cross_entropy(pred, target)
            return loss
        
        def attribute(input, class_idx):
            heatmap = iba.analyze(input, class_idx, model_loss_closure, is_imagenet_a=is_imagenet_a, is_imagenet_r=is_imagenet_r,)
            heatmap = postprocess_heatmap(heatmap)
            return heatmap
        
        return attribute
    