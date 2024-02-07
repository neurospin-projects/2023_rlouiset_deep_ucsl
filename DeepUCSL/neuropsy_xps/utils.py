def create_diagnosis_and_subtype_dict(batch_of_labels):
    # TODO: THESE LINES ARE SPECIFIC TO OUR EXPERIMENTAL SETUP: HC:0, SZ:1 and BD:2.
    batch_of_labels["subtype"] = batch_of_labels["diagnosis"] - 1.0
    batch_of_labels["diagnosis"] = (batch_of_labels["diagnosis"] > 0.5).float()
    return batch_of_labels