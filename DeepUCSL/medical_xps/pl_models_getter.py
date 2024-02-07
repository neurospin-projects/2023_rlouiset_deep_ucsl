from lightning_models.byol import LitBYOL
from lightning_models.swav import LitSwAV
from lightning_models.classifier import LitClassifier
from lightning_models.deep_cluster import LitDeepCluster
from lightning_models.deep_ucsl import LitDeepUCSL
from lightning_models.pcl import LitPCL
from lightning_models.sim_clr import LitSimCLR
from lightning_models.vae import LitVAE
from lightning_models.sup_con import LitSupCon
from lightning_models.scan import LitSCAN


def get_pl_model(model_type, n_classes, n_clusters, loss, loss_params, lr, fold):
    if model_type in ["Classification", "ClassificationOfSubgroups"]:
        return LitClassifier(model_type, n_clusters, n_classes, loss, loss_params, lr, fold)
    elif model_type == "Deep UCSL":
        return LitDeepUCSL(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "DeepCluster-v2":
        return LitDeepCluster(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "SimCLR":
        return LitSimCLR(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "BYOL":
        return LitBYOL(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "SwAV":
        return LitSwAV(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "SupCon":
        return LitSupCon(model_type, n_classes, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "SCAN":
        return LitSCAN(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "PCL":
        return LitPCL(model_type, n_clusters, loss, loss_params, lr, fold)
    elif model_type == "VAE":
        return LitVAE(model_type, loss, loss_params, lr, fold)
    else:
        return NotImplementedError
