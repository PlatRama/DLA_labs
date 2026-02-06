import argparse

from experiments.experiment_cnn import compare_cnn_depths, compare_cnn_with_residual, train_teacher_model
from experiments.experiment_distillation import run_distillation_experiments
from experiments.experiment_mlp import compare_depths, compare_with_residual


def main():
    parser = argparse.ArgumentParser(description='Lab1 experiments:')
    
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['mlp', 'cnn', 'distillation', 'all'],
                       help='Which experiment to run')
    
    parser.add_argument('--mlp-type', type=str, default='all',
                       choices=['depths', 'residual', 'all'],
                       help='For MLP experiments: which comparison to run')
    
    parser.add_argument('--cnn-type', type=str, default='all',
                       choices=['depths', 'residual', 'teacher', 'all'],
                       help='For CNN experiments: which comparison to run')
    
    args = parser.parse_args()
    
    if args.experiment == 'mlp' or args.experiment == 'all':
        print("\n" + "="*80)
        print("RUNNING MLP EXPERIMENTS (Exercise 1.1 & 1.2)")
        print("="*80 + "\n")
        
        if args.mlp_type in ['depths', 'all']:
            compare_depths()
        
        if args.mlp_type in ['residual', 'all']:
            compare_with_residual()
    
    if args.experiment == 'cnn' or args.experiment == 'all':
        print("\n" + "="*80)
        print("RUNNING CNN EXPERIMENTS (Exercise 1.3)")
        print("="*80 + "\n")
        
        if args.cnn_type in ['depths', 'all']:
            compare_cnn_depths()
        
        if args.cnn_type in ['residual', 'all']:
            compare_cnn_with_residual()
        
        if args.cnn_type == 'teacher':
            train_teacher_model()
    
    if args.experiment == 'distillation':
        print("\n" + "="*80)
        print("RUNNING DISTILLATION EXPERIMENTS (Exercise 2.2)")
        print("="*80 + "\n")
        
        run_distillation_experiments()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
