from models.CQC import get_initial_cqc_array
from flwr.app import ArrayRecord, MetricRecord
from evaluation.eval_engine import global_cqc_evaluate
from loaders.load_cifar10 import load_cifar10_iid

def get_initial_model_array(context):
    model_name = context.run_config.get("model")
    if model_name == "CQC":
        return get_initial_cqc_array(context)
    # TODO: Add more models here as needed
    raise ValueError(f"Unsupported model: {model_name}")

# Create evaluation function with quantum parameters
def make_global_evaluate(context):

    # Load centralized test data (using partition 0 as test set)
    dataset_name = context.run_config.get("dataset")
    test_batch_size = 128
    # TODO: Add more datasets here as needed
    if dataset_name == "cifar10":
        _, testloader = load_cifar10_iid(partition_id=0, num_partitions=1, batch_size=test_batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model_name = context.run_config.get("model")
    if model_name == "CQC":
        def global_evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            return global_cqc_evaluate(server_round, arrays, context, testloader)
        return global_evaluate_fn
    # TODO: Add more models here as needed
    raise ValueError(f"Unsupported model: {model_name}")
    
