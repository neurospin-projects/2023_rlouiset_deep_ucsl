# Deep UCSL
GitHub repository of the code and examples of Deep UCSL: "Automatic Discovery of Disease Subgroups by Contrasting with Healthy Controls".
A paper that we submitted to IEEE TMI.

## Abstract:
In Subgroup Discovery, practitioners are interested in discovering interpretable and homogeneous
subgroups within a group of patients. In this paper, assuming that healthy subjects (i.e., controls) share common but
irrelevant factors of variation with the patients, we motivate and develop a Contrastive Subgroup Discovery method,
entitled Deep UCSL. By contrasting controls with patients, we identify subgroups that stem only from the pathological
variability specific to the disease, while disregarding the common variability shared with the controls. To this end,
we propose a Deep Learning framework that learns a discriminative and expressive representation space through a
deep features extractor. From a mathematical standpoint, we derive a novel loss from the conditional joint likeli-
hood between latent clusters (disease subgroups) and patient/control labels. The optimization procedure is based on
an Expectation-Maximization strategy alternating between a) subgroups inference and b) features encoder parameters
update. Furthermore, we introduce a novel regularization term that promotes the representation space to emphasize
disease-specific variability while disregarding the common variability shared with the controls. Compared to previous
related works, our approach quantitatively improves the quality of the estimated subgroups, as demonstrated on
a MNIST-based toy example and four distinct real medical imaging datasets.

## Datasets
MNIST and Medical datasets can be downloaded with the links in the paper.
Datasets creation scripts are in medical_xps/datasets.
NeuroImaging datasets generally require authorizations and need to be pre-processed with CAT 12 (SPM) Software.

## To run a medical scripts please do use:

python setup.py install

python DeepUCSL/medical_xps/main.py --cfg_file DeepUCSL/medical_xps/configs_odir/DeepUCSL_config.yaml






