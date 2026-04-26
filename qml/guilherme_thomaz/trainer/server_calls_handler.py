from models.CQC import get_initial_cqc_array
from flwr.app import ArrayRecord, MetricRecord
from evaluation.eval_engine import global_cqc_evaluate

def get_initial_model_array(context):
    if context.run_config.get("model") == "CQC":
        return get_initial_cqc_array(context)
    # TODO: Add more models here as needed
    raise ValueError(f"Unsupported model: {context.run_config.get('model')}")

# Create evaluation function with quantum parameters
def make_global_evaluate(context):
    if context.run_config.get("model") == "CQC":
        def global_evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            return global_cqc_evaluate(server_round, arrays, context)
        return global_evaluate_fn
    # TODO: Add more models here as needed
    raise ValueError(f"Unsupported model: {context.run_config.get('model')}")
    
