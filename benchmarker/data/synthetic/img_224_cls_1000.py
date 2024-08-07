"""Generate synthetic data for 224x224 images"""


from .helpers import gen_classification_data


def get_data(params, name_key="x"):
    """Generate synthetic 224x224 images. Set `params["size"]`
    appropriately. Import this function in the `data.py` of the
    problem, so it can be called by `INeuralNet`.

    """
    return gen_classification_data(params, 1000, name_key=name_key)
