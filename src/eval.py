import numpy as np
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint, Logger, extract_mean_features, train_dim_reduction
from src.methods.tim import ALPHA_TIM, TIM_GD#, ALPHA_ALPA, ALPA
from src.methods.AM import ALPHA_AM, AM
from src.methods.protoLP import PROTOLP
from src.methods.LPMAP import LPMAP
from src.methods.ALPA_cleaning import ALPA_CLEANING
from src.methods.iLPC import iLPC
from src.methods.laplacianshot import LaplacianShot
from src.methods.bdcspn import BDCSPN
from src.methods.simpleshot import SimpleShot
from src.methods.baseline import Baseline, Baseline_PlusPlus
from src.methods.pt_map import PT_MAP
from src.methods.protonet import ProtoNet
from src.methods.entropy_min import Entropy_min
from src.datasets import Tasks_Generator, CategoriesSampler, get_dataset, get_dataloader
import time

class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

    def run_full_evaluation(self, model):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.method))
        load_checkpoint(model=model, model_path=self.args.ckpt_path, type=self.args.model_tag)
        dataset = {}
        loader_info = {'aug': False, 'out_name': False}

        if self.args.target_data_path is not None:  # This mean we are in the cross-domain scenario
            loader_info.update({'path': self.args.target_data_path,
                                'split_dir': self.args.target_split_dir})

        train_set = get_dataset('train', args=self.args, **loader_info)
        dataset['train_loader'] = train_set

        test_set = get_dataset(self.args.used_set, args=self.args, **loader_info)
        # print(len(test_set))
        dataset.update({'test': test_set})

        train_loader = get_dataloader(sets=train_set, args=self.args)
        train_mean, _ = extract_mean_features(model=model,  train_loader=train_loader, args=self.args, logger=self.logger, device=self.device)
        # reductor = train_dim_reduction(model=model,  train_loader=train_loader, args=self.args, logger=self.logger, device=self.device)
        # from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, Isomap
        # from sklearn.decomposition import PCA, KernelPCA, FastICA, LatentDirichletAllocation
        # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # import plda.classifier
        # reductor = plda.classifier.Classifier()
        # reductor = LinearDiscriminantAnalysis()
        # reductor = LatentDirichletAllocation(n_components=64, random_state=0)
        # reductor = SpectralEmbedding(n_components=64, gamma=1)
        # train_mean = 0
        # reductor = None

        results = []
        acc_all = []
        conf_all = []
        start = time.time()
        for shot in self.args.shots:
            sampler = CategoriesSampler(dataset['test'].labels, self.args.batch_size,
                                        self.args.n_ways, shot, self.args.n_query,
                                        self.args.balanced, self.args.alpha_dirichlet)

            test_loader = get_dataloader(sets=dataset['test'], args=self.args,
                                         sampler=sampler, pin_memory=True)
            task_generator = Tasks_Generator(n_ways=self.args.n_ways, shot=shot, loader=test_loader,
                                             train_mean=train_mean, log_file=self.log_file)
            results_task = []
            acc_all_task = []
            for i in range(int(self.args.number_tasks/self.args.batch_size)):

                method = self.get_method_builder(model=model)

                tasks = task_generator.generate_tasks()

                # Run task
                logs = method.run_task(task_dic=tasks, shot=shot)#, reductor=reductor)#, train_loader=train_loader)

                acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
                acc_all_task.append(logs['acc'][:, -1])
                # if i%50==0:
                #     print(i, ' ', compute_confidence_interval(acc_all_task))
                results_task.append(acc_mean)
                # del method
            results.append(results_task)
            acc_all_batches = np.concatenate(acc_all_task, axis = 0)
            mean, conf = compute_confidence_interval(acc_all_batches)
            acc_all.append(mean)
            conf_all.append(conf)
        # print(acc_all_batches.shape)
        mean_accuracies = np.asarray(results).mean(1)
        end = time.time()
        seconds = (end - start)/500
        self.logger.info('----- Final test results -----')
        for shot in self.args.shots:
            if self.args.method=='ALPHA-ALPA' or self.args.method=='ALPA-BALANCED':
                print('{}-shot mean test accuracy over {} tasks: {:0.2f}, +- {:0.2f}, PLC: {}, dataset: {}, backbone: {}, method: {}, phi: {}, alpha: {}, K : {}, alpha_dirichlet: {}, plc: {}, alpha_values: {}, time: {}'.format(shot, self.args.number_tasks, acc_all[self.args.shots.index(shot)] * 100,
                        conf_all[self.args.shots.index(shot)] * 100, self.args.PLC, self.args.dataset, self.args.arch,self.args.method, self.args.phi, self.args.alpha, self.args.K, self.args.alpha_dirichlet, self.args.PLC, self.args.alpha_values, seconds))  # , self.args.alpha_values))
            else:
                print('{}-shot mean test accuracy over {} tasks: {:0.2f}, +- {:0.2f}, PLC: {},  alpha_dirichlet: {}, dataset: {}, backbone: {}, method: {}, phi: {}, time: {}'.format(shot, self.args.number_tasks, acc_all[self.args.shots.index(shot)]*100, conf_all[self.args.shots.index(shot)]*100, self.args.PLC, self.args.alpha_dirichlet, self.args.dataset, self.args.arch, self.args.method, self.args.phi, seconds))#, self.args.alpha_values))
            # self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
            #                                                                        mean_accuracies[self.args.shots.index(shot)]))
            self.logger.info('{}-shot mean test accuracy over {} tasks: {:0.2f}, +- {:0.2f}, PLC: {}, dataset: {}, backbone: {}, method: {}, phi: {}'.format(shot, self.args.number_tasks, acc_all[self.args.shots.index(shot)]*100, conf_all[self.args.shots.index(shot)]*100, self.args.PLC, self.args.dataset, self.args.arch, self.args.method, self.args.phi))#, self.args.alpha_values))
        return mean_accuracies

    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.method == 'ALPHA-TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.method == 'ALPHA-AM':
            method_builder = ALPHA_AM(**method_info)
        elif self.args.method == 'PROTOLP':
            method_builder = PROTOLP(**method_info)
        elif self.args.method == 'AM':
            method_builder = AM(**method_info)
        elif self.args.method == 'iLPC':
            method_builder = iLPC(**method_info)
        elif self.args.method == 'ALPA-CLEANING':
            method_builder = ALPA_CLEANING(**method_info)
        elif self.args.method == 'LP-MAP':
            method_builder = LPMAP(**method_info)
        elif self.args.method == 'TIM-GD':
            method_builder = TIM_GD(**method_info)
        elif self.args.method == 'LaplacianShot':
            method_builder = LaplacianShot(**method_info)
        elif self.args.method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.method == 'SimpleShot':
            method_builder = SimpleShot(**method_info)
        elif self.args.method == 'Baseline':
            method_builder = Baseline(**method_info)
        elif self.args.method == 'Baseline++':
            method_builder = Baseline_PlusPlus(**method_info)
        elif self.args.method == 'PT-MAP':
            method_builder = PT_MAP(**method_info)
        elif self.args.method == 'ProtoNet':
            method_builder = ProtoNet(**method_info)
        elif self.args.method == 'Entropy-min':
            method_builder = Entropy_min(**method_info)
        else:
            self.logger.exception("Method must be in ['TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
            raise ValueError("Method must be in ['TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
        return method_builder