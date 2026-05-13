from trainer.train_engine import handle_cqc_train_call
from evaluation.eval_engine import handle_cqc_evaluate_call

def handle_train_call(msg, context):
    model_name = context.run_config.get("model")
    if model_name == "CQC":
        return handle_cqc_train_call(msg, context)
    # TODO: add more models here as needed
    raise ValueError(f"Unsupported model: {context.run_config.get('model')}")

def handle_evaluate_call(msg, context):
    model_name = context.run_config.get("model")
    if model_name == "CQC":
        return handle_cqc_evaluate_call(msg, context)
    # TODO: add more models here as needed
    raise ValueError(f"Unsupported model: {context.run_config.get('model')}")