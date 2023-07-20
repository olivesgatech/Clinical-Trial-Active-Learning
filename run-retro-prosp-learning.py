import os

os.system('')
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import torch
import pandas as pd
import numpy as np
from myUtils.trainer_updated import Trainer_New
from query_strategies import RandomSampling, LeastConfidenceSampler, EntropySampler, \
    MarginSampler, BadgeSampler, CoresetSampler


def parse_everything():
    parser = argparse.ArgumentParser(description="PyTorch Prospective/Retrospective AL")
    parser.add_argument('--architecture', type=str, default='resnet_18',
                        choices=['resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152',
                                 'densenet_121', 'densenet_161', 'densenet_169', 'densenet_201',
                                 'vgg_11', 'vgg_13', 'vgg_16', 'vgg_19', 'mlp'],
                        help='architecture name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='RCT',
                        choices=['OASIS', 'RCT'],
                        help='dataset name (default: RCT)')
    parser.add_argument('--data_path', type=str, #default='./Data', changed this (zmf)
                        default='/media/zoe/HD/Datasets',
                        help='dataset path')
    parser.add_argument('--train_spreadsheet', type=str, default='/home/zoe/activelearning/Spreadsheets/prime_trex_compressed.csv')
    parser.add_argument('--test_spreadsheet', type=str,
                        default='/home/zoe/activelearning/Spreadsheets/prime_trex_test_new.csv')
    parser.add_argument('--train_type', type=str, default='traditional',
                        choices=['traditional', 'positive_congruent'])
    parser.add_argument('--mode', type=str, default='acc', choices=['acc', 'val'])
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--run_status', type=str, default='train',
                        choices=['train', 'test'], help='')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=32,  ######## 128;
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=16,  #default is 1
                        metavar='N', help='input batch size for \
                                testing (default: auto)')

    parser.add_argument('--forgetting_mode', default='None', choices=['dynamic', 'fixed', 'None'],
                        help='examine forgetting stats on dynamic or fixed test set (or none)')

    parser.add_argument('--forgetting_strategy', type=str, default='rand', choices=['rand', 'least_conf', 'entropy', 'margin', 'badge', 'coreset'])
    parser.add_argument('--skip_seq', default=False, type=bool,
                        help='skip weeks in the natural order')
    parser.add_argument('--skip_rand', default=False, type=bool,
                        help='skip weeks in randomized order')

    # optimizer params
    parser.add_argument('--optimizer', default="adam",
                        help='optimizer to use, default is sgd. Can also use adam')  #### sgd
    parser.add_argument('--lr', type=float, default=0.00015,  ####### 0.00015
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training') ## change to false later
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pretrained', type=bool, default=False)

    # active learning parameters
    parser.add_argument('--strategy', type=str, default='rand',
                        choices=['rand','least_conf', 'entropy', 'margin','badge','coreset'],
                        help='strategy used for sample query in active earning experiment')
    parser.add_argument('--start_strategy', type=str, default='rand_init',
                        choices=['rand_init']) # can customize original selection of points
    parser.add_argument('--nstart', type=int, default=128,
                        help='number of samples in the initial data pool')
    parser.add_argument('--nend', type=int, default=10000,  ######## this is the budget
                        help='maximum amount of points to be queried')
    parser.add_argument('--nquery', type=int, default=800,
                        help='number of new samples to be queried in each round')
    parser.add_argument('--min_acc', type=float, default=97.0,
                        help='number of samples to be queried in each round')
    parser.add_argument('--visit_mode', type=str, default='None', choices=['yes', 'None'],
                        help='train off of visit #')
    parser.add_argument('--dynamic_test_size', type=int, default=0,
                        help='if customizing dynamic test size, this needs to be changed')
    parser.add_argument('--skip', type=int, default=0,
                        help='number of weeks to skip between')
    parser.add_argument('--sample_past', type=bool, default=False,
                        help='select past samples (randomly) for CT')
    parser.add_argument('--current_only', type=bool, default=False,
                        help='only query new visit data')

    # for continual learning
    parser.add_argument('--continual', type=bool, default=False)

    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--date', type=str, default='06-08-23')
    args = parser.parse_args()

    return args


def main():
    # parse all arguments
    args = parse_everything()
    placeholder = np.array([None])
    # make sure gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # make gpu ids inputable
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            print('gpu ids: ', args.gpu_ids)
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default batch size
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    # default test batch size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    # print all arguments
    print(args)

    # init seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up train and test spreadsheets
    tr_path = args.train_spreadsheet
    te_path = args.test_spreadsheet # this will be different for fixed vs dynamic settings

    # Read in train, test spreadsheet data
    train_data = pd.read_csv(tr_path)
    test_data = pd.read_csv(te_path)
    Patient_Dir_tr = train_data['File_Path'] # Total train patients
    Patient_Dir_te = test_data['File_Path'] # total test patients

    max_train = max(train_data['Visit'])
    print('Train data max visit: ', max_train)
    max_test = max(test_data['Visit'])
    print('Test data max visit: ', max_test)

    # Smallest visit # is deciding factor
    deciding = min(max_train, max_test)

    # initialize starting week if performing random week selection
    if args.skip_seq == True:
        init_visit = 1  # start with week 1
    elif args.skip_rand == True:
        init_visit = np.random.choice(visit_arr, size=1, replace=False)
        visit_arr = np.delete(visit_arr, np.where(visit_arr==init_visit))
        init_visit=init_visit[0]
    else:
        init_visit = 1

    # get active learning parameters
    NUM_QUERY = args.nquery
    init_samples = args.nstart
    train_pool = len(Patient_Dir_tr)
    test_pool = len(Patient_Dir_te)
    total = np.arange(train_pool)
    total_test = np.arange(test_pool)

    # Get START IDXS for CONTINUAL LEARNING and ACTIVE LEARNING

    # RetroActive learning
    if args.visit_mode == 'None':
        start_idxs = total[np.random.permutation(len(total))][:init_samples]

    # PAL learning
    else:
        week1_tr = train_data[train_data['Visit'] == init_visit]

        # Get index list of all init visit images
        week_tr_pool_ind = week1_tr['Ind'].to_numpy()
        # Create week 1 pool (indexed from 0 to total len in order)
        start_idxs = ((np.random.permutation(week_tr_pool_ind)))[:init_samples]

    # This ensures the test set is the same in both PAL AND RAL
    if args.forgetting_mode == 'fixed':
        start_idxs_test = total_test[np.random.permutation(len(total_test))]
        week_te_pool_ind = test_data['Ind'].to_numpy()
    elif args.dynamic_test_size > 0:
        week1_te = test_data[test_data['Visit'] == init_visit]
        # For test set; index list of all week 1 test images
        week_te_pool_ind = week1_te['Ind'].to_numpy()
        # # This is optional; if making the dynamic test set smaller you would need something like this
        start_idxs_test = ((np.random.permutation(week_te_pool_ind)))[:args.dynamic_test_size]
    else:
        # Dynamic test set that adds in all images from visit at each round
        week1_te = test_data[test_data['Visit'] == init_visit]
        # For test set; index list of all week 1 test images
        week_te_pool_ind = week1_te['Ind'].to_numpy()
        # Random permutation of the total indices corresponding to week 1 images
        start_idxs_test = ((np.random.permutation(week_te_pool_ind)))


    # sampling
    if args.visit_mode == 'yes':
        inds = week_tr_pool_ind
    else:
        inds = placeholder

    # Init sampler classes 
    if args.strategy == 'rand':
        print('Using a random sampler')
        sampler = RandomSampling(train_pool, start_idxs, inds)


    elif args.strategy == 'least_conf':
        print('Using least confidence sampler')
        sampler = LeastConfidenceSampler(train_pool, start_idxs, inds)

    elif args.strategy == 'entropy':
        print('Using least entropy sampler')
        sampler = EntropySampler(train_pool, start_idxs, inds)

    elif args.strategy == 'margin':
        print('Using least margin sampler')
        sampler = MarginSampler(train_pool, start_idxs, inds)

    elif args.strategy == 'coreset':
        print('Using coreset sampler')
        sampler = CoresetSampler(train_pool, start_idxs, inds)

    elif args.strategy == 'badge':
        print('Using badge sampler')
        sampler = BadgeSampler(train_pool, start_idxs)

    else:
        print('Sampler not implemented!!!!')
        raise NotImplementedError

    # Default: if adding in entire visit test data at each round (easier to use sampler class for dynamic test set)
    if args.dynamic_test_size == 0:
        # Adding full visit data in at each round
        sampler_test = RandomSampling(test_pool, start_idxs_test, week_te_pool_ind)

    NUM_ROUND = args.rounds
    print('Rounds: %d' % NUM_ROUND)


    # init dataframe
    DATE = args.date

    # Results folder
    df_path = os.path.join('excel', args.dataset, args.architecture, args.start_strategy, args.strategy + '_' + str(args.nquery), DATE)
    print(df_path)

    if args.continual:
        ext = '_continual'
    else:
        ext = ''
    if args.visit_mode == 'yes':
        df_path = os.path.join(df_path, 'prospective_learning' + ext)
    else:
        df_path = os.path.join(df_path, 'retrospective_learning' + ext)
    if args.skip_seq == True:
        df_path = os.path.join(df_path, 'sequential_week_skip')
    elif args.skip_rand == True:
        df_path = os.path.join(df_path, 'random_week_skip')
    if args.forgetting_mode == 'dynamic':
        df_name = os.path.join(df_path, 'test_accuracy_dynamic' + '_' + str(args.dynamic_test_size) + '_' + str(args.seed) + '.xlsx')
        files_sampled_path_TEST = os.path.join(df_path, 'patients_sampled_TEST' + str(args.seed) + '.xlsx')
    elif args.forgetting_mode == 'fixed':
        df_name = os.path.join(df_path, 'test_accuracy_fixed' + str(args.seed) + '.xlsx')
    else:
        df_name = os.path.join(df_path, 'test_accuracy' + str(args.seed) + '.xlsx')

    files_sampled_path = os.path.join(df_path, 'patients_sampled' + str(args.seed) + '.xlsx')

    # Create results folder
    if not os.path.exists(df_path):
        os.makedirs(df_path)

    # Init dataframe (df)
    df = pd.DataFrame(columns=['Train Samples', 'Test Acc', 'Balanced Acc', 'Test Precision', 'Test Recall', '# forgetting events', 'Test samples'])

    visit_tab = init_visit # keep track of visit
    cur_visit = init_visit # current visit

    # Create Trainer
    trainer = Trainer_New(args)

    # Pick out data corresponding to current visit
    week_tr = train_data[train_data['Visit'] == visit_tab]
    week_te = test_data[test_data['Visit'] == visit_tab]
    # Get index list of all current visit images
    week_tr_pool_ind_total = week_tr['Ind'].to_numpy()
    print('Initial count of images in FIRST week: ', week_tr_pool_ind_total.shape)
    week_te_ind_tot = week_te['Ind'].to_numpy()
    complete = False

    if args.run_status == 'train':
        # train over number of epochs
        for rounds in range(NUM_ROUND):
            print('Round: %d' % rounds)
            # Next visit
            # Need to do this first in the case of BADGE, Coreset, etc which uses unlabeled pool to determine next samples to query (need to expand that pool to include next visit)
            if args.skip_seq == True:
                visit_tab += 2
                if visit_tab > deciding:
                    break
            elif args.skip_rand == True:
                if len(visit_arr) >= 1:
                    visit_tab = np.random.choice(visit_arr, size=1, replace=False)
                    visit_arr = np.delete(visit_arr, np.where(visit_arr == visit_tab))
                    visit_tab = visit_tab[0]
                else:
                    break
            elif args.visit_mode=='yes' or args.forgetting_mode=='dynamic':
                visit_tab += 1
                if visit_tab > deciding:
                    complete = True # for the last round
                    #break
            # Data from next visit; necessary when defining the unlabeled loaders (for strategies that compute embeddings, etc to query new data) in Trainer class
            if not complete:
                print('The next visit is: ', visit_tab)
                week_tr = train_data[train_data['Visit'] == visit_tab]
                week_te = test_data[test_data['Visit'] == visit_tab]
                # Get index list of all current visit images
                week_tr_pool_ind_current = week_tr['Ind'].to_numpy()
                week_te_cur = week_te['Ind'].to_numpy()
                # Concatenate to get total list of indices corresponding to current + past visits
                if args.visit_mode == 'yes':
                    # if true: only look at current visit data to query from
                    if args.current_only:
                        week_tr_pool_ind_total = week_tr_pool_ind_current
                        week_te_ind_tot = week_te_cur
                    else:
                        # allow past visit data to be queried from
                        week_tr_pool_ind_total = np.concatenate((week_tr_pool_ind_total, week_tr_pool_ind_current))
                        week_te_ind_tot = np.concatenate((week_te_ind_tot, week_te_cur))

            if args.visit_mode == 'None':
                print('Regular active learning')
                week_tr_pool_ind_total = train_data['Ind'].to_numpy()
                week_te_ind_tot = test_data['Ind'].to_numpy()

            # init epoch and accuracy parameters
            epoch = 0
            acc = 0.0

            # start training
            # First, initialize trainer_loader with current indexes
            # get current training indices
            current_idxs = sampler.idx_current
            # get test idxs (for dynamic test set)
            if args.forgetting_mode == 'dynamic':
                current_idxs_test = sampler_test.idx_current
                # This accounts for BOTH dynamic and fixed test set
            else:
                current_idxs_test = placeholder


            # For recorded statistics of patients sampled
            if rounds == 0:
                selected_patients = train_data.iloc[current_idxs]
                selected_patients['Round'] = int(rounds)
                if args.forgetting_mode == 'dynamic':
                    patients_test = test_data.iloc[current_idxs_test]
                    patients_test['Round'] = int(rounds)
            else:
                new = train_data.iloc[new_idxs]
                new['Round'] = int(rounds)
                selected_patients = pd.concat([selected_patients, new], ignore_index=False)
                if args.forgetting_mode == 'dynamic':
                    patients_test_new = test_data.iloc[new_idxs_test]
                    patients_test_new['Round'] = int(rounds)
                    patients_test = pd.concat([patients_test, patients_test_new], ignore_index=False)
                    
            # UPDATE TRAIN, TEST LOADERS WITH CURRENT INDEXES
            trainer.update_loaders(current_idxs=current_idxs, batch_size=args.batch_size, data=train_data,
                                   total_week_indices=week_tr_pool_ind_total,
                                   current_idxs_test=current_idxs_test, data_test=test_data,
                                   total_te_week_ind=week_te_ind_tot, train_pool=train_pool)

            # training routine for datasets that don't need validation dataset vs those that do
            # Accuracy is used to prevent overfitting
            while acc < args.min_acc:
                # train for this epoch

                acc, model_trained, opt = trainer.training(epoch)

                # increment epoch counter
                epoch += 1

            # test statistics
            print('---Testing normal data....---')
            if args.dynamic_test_size >= 0:
                mode = 'dynamic'
            if args.forgetting_mode == 'fixed':
                mode = 'fixed'
            elif args.visit_mode == 'None' and args.forgetting_mode=='None': # regular active learning; no forgetting events [unused in paper]
                mode = 'normal'
            print('MODE: ', mode)

            if args.visit_mode=='yes':
                title_new = 'prospective'
            else:
                title_new = 'retrospective'
            accs, preds_dict, num_forgets = trainer.testing(epoch, mode=mode, vis_mode=title_new, q=args.strategy, round=rounds)


            test_acc = accs['acc']
            test_precision = accs['precision']
            test_recall = accs['recall']

            df.loc[rounds, 'Test Acc'] = test_acc
            df.loc[rounds, 'Balanced Acc'] = accs['balanced acc']
            df['Test Precision'] = df['Test Precision'].astype('object')
            df['Test Recall'] = df['Test Recall'].astype('object')
            df.at[rounds, 'Test Precision'] = list(test_precision)
            df.at[rounds, 'Test Recall'] = list(test_recall)
            df.loc[rounds, 'Train Samples'] = len(current_idxs)
            if args.visit_mode == 'yes':
                df.loc[rounds, 'Visit #'] = cur_visit
            if args.forgetting_mode != 'None':
                df.loc[rounds, '# forgetting events'] = num_forgets
            if args.forgetting_mode == 'dynamic':
                df.loc[rounds, 'Test samples'] = len(current_idxs_test)
            else:
                df.loc[rounds, 'Test samples'] = test_pool # Fixed test set
            df.loc[rounds, 'AUC'] = accs['auc']

            if args.visit_mode == 'yes' or args.dynamic_test_size==0:
                cur_visit = visit_tab

                # query new samples
                week_tr = train_data[train_data['Visit'] == visit_tab]
                week_te = test_data[test_data['Visit'] == visit_tab]

                # Get index list of all visit images
                week_tr_pool_ind = week_tr['Ind'].to_numpy()
                week_te_pool_ind = week_te['Ind'].to_numpy()

            # Updating when dynamic test set size is full visit images at each round
            if args.forgetting_mode == 'dynamic' and args.dynamic_test_size == 0:
                new_idxs_test = sampler_test.query(len(week_te_pool_ind), week_te_pool_ind)

            # Update sampler for: regular ACTIVE LEARNING
            if args.visit_mode == 'None':
                if args.strategy == 'least_conf' or args.strategy == 'entropy' or args.strategy == 'margin':
                    print('calculating probabilities')
                    probs = trainer.get_probs(mode='tr')
                    new_idxs = sampler.query(NUM_QUERY, probs)
                elif args.strategy == 'badge':
                    print('calculating gradient embeddings')
                    embeddings = trainer.get_badge_embeddings(mode='tr')
                    new_idxs = sampler.query(NUM_QUERY, embeddings)
                elif args.strategy == 'coreset':
                    print('calculating embeddings')
                    new_idxs = sampler.query(NUM_QUERY, trainer)
                else:
                    # Random sampling
                    # using random sampling class
                    new_idxs = sampler.query(NUM_QUERY, placeholder, opt=args.current_only)

            # Update sampler for PROSPECTIVE
            else:
                if args.strategy == 'rand':
                    new_idxs = sampler.query(NUM_QUERY, week_tr_pool_ind, opt=args.current_only)

                elif args.strategy == 'least_conf' or args.strategy == 'entropy' or args.strategy == 'margin':
                    print('calculating probabilities')
                    probs = trainer.get_probs(mode='tr')
                    new_idxs = sampler.query(NUM_QUERY, probs)

                elif args.strategy == 'badge':
                    print('calculating gradient embeddings')
                    embeddings = trainer.get_badge_embeddings(mode='tr')
                    new_idxs = sampler.query(NUM_QUERY, embeddings)

                elif args.strategy == 'coreset':
                    print('calculating embeddings')
                    new_idxs = sampler.query(NUM_QUERY, trainer)

            if args.sample_past:
                previous = trainer.randomly_sample(number=128)
            else:
                previous = np.array([])
            # update sampler
            sampler.update(new_idx=new_idxs, cont=args.continual, add=previous)
            if args.forgetting_mode == 'dynamic':
                sampler_test.update(new_idx=new_idxs_test, cont=args.continual, add=previous)

            # RESET model if Active Learning
            if args.continual == False:
                print('resetting model')
                trainer.clear_statistics()

        # save to dataframe
        df.to_excel(df_name)
        selected_patients.to_excel(files_sampled_path)
        if args.forgetting_mode=='dynamic':
            patients_test.to_excel(files_sampled_path_TEST)

    elif args.run_status == 'test':
        # test over all epochs
        # if you want to test separately.
        trainer.testing()
    else:
        raise (Exception('please set args.run_status=train or test'))


# start main
if __name__ == "__main__":
    main()
