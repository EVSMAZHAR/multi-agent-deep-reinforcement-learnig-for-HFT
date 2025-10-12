import argparse, yaml
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    print("Evaluation skeleton. Compute metrics & confidence intervals here.")
if __name__ == '__main__':
    main()
