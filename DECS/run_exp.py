import os, csv
import numpy as np
from time import time
import argparse
from main import train
from seed import set_seed


if __name__ == "__main__":
    set_seed()
    # Global experiment settings
    method = 'DECS'
    expdir = 'results/temp'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='usps', type=str, help="Dataset for Clustering")
    parser.add_argument('--trials', default=1, type=int)
    parser.add_argument('--pretrain-epochs', default=500, type=int, help="Number of epochs for pretraining")
    parser.add_argument('--maxiter', default=2e4, type=int, help="Maximum number of iterations")
    args = parser.parse_args()
    args.aug_pretrain = True
    args.pretrained_weights = None
    args.verbose = 1
    args.testing = False
    args.weights = None
    args.aug_cluster = True
    args.batch_size = 256
    args.tol = 0.001
    trials = args.trials

    # Log files
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(expdir + '/results.csv', 'a')
    logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'ari', 'time'])
    logwriter.writeheader()

    args.method = method
    args.optimizer = 'adam'
    args.lr = 0.001

    dataset = args.dataset
    logwriter.writerow(dict(trials=method + '_' + dataset, acc='', nmi='', ari='', time=''))
    result = np.zeros(shape=[trials, 4], dtype=float)
    args.update_interval = 30
    for trial in range(trials):
        args.save_dir = os.path.join(expdir, method, dataset, 'trial%d' % trial)
        t0 = time()
        train(args)
        t1 = time()

        # saving log results
        log = open(os.path.join(args.save_dir, 'log.csv'), 'r')
        reader = csv.DictReader(log)
        metrics = []
        for row in reader:
            metrics.append([row['acc'], row['nmi'], row['ari']])
        result[trial, :] = np.asarray(metrics[-1] + [int(t1 - t0)])
        log.close()

    # save log to global log file
    for t, line in enumerate(result):
        logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2], time=line[3]))
    mean = np.mean(result, 0)
    logwriter.writerow(dict(trials=' ', acc=mean[0], nmi=mean[1], ari=mean[2], time=mean[3]))
    logfile.flush()

    logfile.close()
