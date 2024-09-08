import torch
from ..util.tqdm import TQDM as tqdm

def heatmap_regression_inference(model, dataloader, device, verbose=True):
    model.eval()
    model.to(device)
    all_outputs = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, disable=not verbose):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
    return torch.cat(all_outputs)

def heatmap_regression_inference_generator(model, dataloader, device, verbose=True):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs in tqdm(dataloader, disable=not verbose):
            inputs = inputs.to(device)
            outputs = model(inputs)
            yield outputs.cpu()

def classification_inference(model, dataloader, device, verbose=True):
    model.eval()
    model.to(device)
    all_outputs = []
    with torch.no_grad():
        for inputs in tqdm(dataloader, disable=not verbose):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
    return torch.flatten(torch.cat(all_outputs))