import os, sys, time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from src.argparse import parse_cl_args
from src.distprocs import DistProcs

class TeeOutput:
    """Write output to both terminal and a log file."""
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        pass  # handled externally by log_file.close()

def main():
    start_t = time.time()
    args = parse_cl_args()

    log_dir = 'Outputs/output_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{args.tsc}_{args.mode}_output.txt")
    log_file = open(log_filename, 'w', encoding='utf-8')
    sys.stdout = TeeOutput(log_file, sys.__stdout__)
    sys.stderr = TeeOutput(log_file, sys.__stderr__)

    try:
        print('='*60)
        print('RUN CONFIGURATION:')
        print('='*60)
        for arg, value in sorted(vars(args).items()):
            print(f'  {arg}: {value}')
        print('='*60)
        print()
        print('start running main...')
        distprocs = DistProcs(args, args.tsc, args.mode)
        distprocs.run()
        print(args)
        print('...finish running main')
        print('run time '+str((time.time()-start_t)/60))
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Terminal output saved to {log_filename}")

if __name__ == '__main__':
    main()

