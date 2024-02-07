import torch

from lightning_callbacks.scan_stack_initialization import StackInitializer
from lightning_callbacks.binary_classification_subgroups_estimations_and_evaluation import \
    BCEClustering
from lightning_callbacks.pcl_prototypes_estimation import PCLPrototypesEstimationAndEvaluation
from lightning_callbacks.run_kmeans_on_representations import KMeansOnRep
from lightning_callbacks.deep_cluster_prototypes_estimations import DeepClusterPrototypesManager
from lightning_callbacks.deep_ucsl_subgroups_pseudo_labels_estimation_and_evaluation import DeepUCSLPseudoLabeller
from lightning_callbacks.sup_con_subgroups_estimation_and_evaluation import SupConClusteringAndLinearProbing
import pytorch_lightning as pl

def get_callbacks_getter(model_type, saving_folder, fold):
    # define callbacks
    checkpoint_callback = SaveModelWeights(saving_folder=saving_folder, fold=fold)

    # define callbacks
    if model_type in ["Deep UCSL"]:
        pseudo_labels_manager = DeepUCSLPseudoLabeller()
        callbacks = [pseudo_labels_manager, checkpoint_callback]
    elif model_type in ["SimCLR", "BYOL"]:
        run_kmeans_on_rep = KMeansOnRep()
        callbacks = [run_kmeans_on_rep, checkpoint_callback]
    elif model_type in ["SCAN"]:
        stack_initializer = StackInitializer()
        callbacks = [stack_initializer, checkpoint_callback]
    elif model_type in ['SupCon']:
        supcon_clustering_and_probing = SupConClusteringAndLinearProbing()
        callbacks = [supcon_clustering_and_probing, checkpoint_callback]
    elif model_type in ["PCL"]:
        prototypes_manager = PCLPrototypesEstimationAndEvaluation()
        callbacks = [prototypes_manager, checkpoint_callback]
    elif model_type in ["Classification"]:
        bce_clustering = BCEClustering()
        callbacks = [bce_clustering, checkpoint_callback]
    elif model_type in ["DeepCluster-v2"]:
        deep_cluster_prototypes_manager = DeepClusterPrototypesManager()
        callbacks = [deep_cluster_prototypes_manager, checkpoint_callback]
    else:
        print("No callback !")
        callbacks = [checkpoint_callback]

    return callbacks

class SaveModelWeights(pl.callbacks.Callback):
    def __init__(self, saving_folder, fold):
        self.saving_folder = saving_folder
        self.fold = fold

    def on_fit_end(self, trainer, pl_module):
        torch.save(pl_module.model, self.saving_folder + str(self.fold) + '/model_epoch-' + str(pl_module.current_epoch) + '.pth')

