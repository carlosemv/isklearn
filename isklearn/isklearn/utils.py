import argparse

def _str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_args(args):
    parser = argparse.ArgumentParser()
    for argname, params in args.items():
    	parser.add_argument(f'--{argname}', **params)
    return parser.parse_known_args()[0]

class ArgumentException(Exception):
    def __init__(self, param="", argvalue="", *args, **kwargs):
    	msg = f"Invalid value {argvalue} for parameter {param}"
    	super().__init__(msg, *args, **kwargs)