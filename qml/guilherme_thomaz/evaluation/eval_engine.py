from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CQC import CQC
from flwr.app import ArrayRecord, MetricRecord

def global_cqc_evaluate(
    server_round: int, arrays: ArrayRecord, context, testloader
) -> MetricRecord:
    n_qubits = context.run_config.get("n-qubits", 4)
    n_layers = context.run_config.get("n-layers", 3)

    # Load the model and initialize it with the received weights
    model = CQC(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the global model on the test set
    test_loss, test_accuracy = test_cqc(model, testloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_accuracy, "loss": test_loss})


def handle_cqc_evaluate_call(msg, context, valloader):

    # Read quantum parameters from configuration
    n_qubits = context.run_config.get("n-qubits", 4)
    n_layers = context.run_config.get("n-layers", 3)

    # Load the model and initialize it with the received weights
    model = CQC(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call the evaluation function
    eval_loss, eval_accuracy = test_cqc(model, valloader, device)

    return {
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "custom_metric_1_value": 0,
        "custom_metric_1_name": 0,
        "custom_metric_2_value": 0,
        "custom_metric_2_name": 0,
        "num_examples": len(valloader.dataset),
    }
    

def test_cqc(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            data = batch["img"].to(device)
            target = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            output = net(data)

            test_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(testloader)
    accuracy = correct / total

    return test_loss, accuracy
