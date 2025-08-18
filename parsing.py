import argparse



def parsing():
    parser = argparse.ArgumentParser()
    # parser.add_argument('method', type=str)
    
    
    parser.add_argument('xai', type=str, help='Network architecture to use')
    parser.add_argument('--target-layer', '--target_layer', type=str, nargs='+', default=['layer4'], help='Target layer for attribution')
    
    parser.add_argument('--data-path', '--data_path', type=str, default='/path/to/imagenet', help='Path to the dataset')
    parser.add_argument('--num-classes', '--num_classes', type=int, default=1000, help='Number of classes in the dataset')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the process in distributed training')
    parser.add_argument('--network', type=str, default='resnet50', help='Network architecture to use')
    parser.add_argument('--load_weight', '--load_weight', default=None, help='Path to the weight file')

    # Configs for IBA
    parser.add_argument('--start-target', '--start_target', type=int, default=4, help='Start layer for attribution')
    parser.add_argument('--end-target', '--end_target', type=int, default=5, help='End target layer for attribution')
    parser.add_argument('--op-name', '--op_name', type=str, nargs='+', default=['blocks'], help='Operation names for attribution')
    parser.add_argument('--optim-steps', '--optim_steps', type=int, default=10, help='Number of optimization steps for IBA')
    parser.add_argument('--iba-lr', '--iba_lr', type=float, default=1, help='Learning rate for IBA')
    parser.add_argument('--beta', type=float, default=10, help='Beta value for IBA')
    parser.add_argument('--load-estim', '--load_estim', action='store_true', help='Load precomputed estimators')
    parser.add_argument('--iba-path', '--iba_path', type=str, default='iba_misc', help='Path to save/load IBA estimators')
    
    parser.add_argument('--start-index', '--start_index', type=int, default=0, help='Start index of token (if st=1, skip the cls token for perturbation)')
    

    return parser