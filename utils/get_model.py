import timm

def get_model(name, args):
    
    model = timm.models.create_model(name, pretrained=(True if args.load_weight is None else args.load_weight), num_classes=args.num_classes)

    config = timm.data.resolve_data_config({}, model=model)
    input_size = config['input_size'][-1]
    return model, input_size

