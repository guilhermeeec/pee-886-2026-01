from trainer.train_engine import handle_cqc_train_call
from evaluation.eval_engine import handle_cqc_evaluate_call
from loaders.load_cifar10 import load_cifar10_iid

def handle_train_call(msg, context):

    # Load the data
    dataset_name = context.run_config.get("dataset")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    # TODO: Add more datasets here as needed
    if dataset_name == "cifar10":
        trainloader, valloader = load_cifar10_iid(partition_id, num_partitions, batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    print(f"Client {partition_id}/{num_partitions} starting training...")

    model_name = context.run_config.get("model")
    if model_name == "CQC":
        return handle_cqc_train_call(msg, context, trainloader, valloader)
    # TODO: add more models here as needed
    raise ValueError(f"Unsupported model: {context.run_config.get('model')}")

def handle_evaluate_call(msg, context):
    # Load the data
    dataset_name = context.run_config.get("dataset")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    # TODO: Add more datasets here as needed
    if dataset_name == "cifar10":
        _, valloader = load_cifar10_iid(partition_id, num_partitions, batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    model_name = context.run_config.get("model")
    if model_name == "CQC":
        return handle_cqc_evaluate_call(msg, context, valloader)
    # TODO: add more models here as needed
    raise ValueError(f"Unsupported model: {context.run_config.get('model')}")