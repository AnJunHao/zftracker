import torch

def get_deconv_output(input, stride, padding, dilation, kernel_size, out_padding):
    return (
        (input - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + out_padding
        + 1
    )


def get_final_output(input, padding, out_padding):
    for i in range(3):
        input = get_deconv_output(
            input,
            2,
            padding=padding[i],
            dilation=1,
            kernel_size=4,
            out_padding=out_padding[i],
        )
    return input

class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

def is_deterministic(model, image=None):
    # Create random image for testing if none is provided
    if image is None:
        image = torch.rand(1, 3, 224, 224)

    # Initialize hooks for each layer in your model
    hooks = [Hook(layer) for layer in model.modules()]
    
    # Run your model
    y_1 = model(image)
    
    # Save the outputs
    outputs_1 = [hook.output for hook in hooks]
    
    # Run your model again
    y_2 = model(image)
    
    # Save the outputs
    outputs_2 = [hook.output for hook in hooks]

    inconsistent_layers = {}
    
    # Compare the outputs
    for i, (o1, o2) in enumerate(zip(outputs_1, outputs_2)):
        if not torch.equal(o1, o2):
            print(f"Layer {i} is inconsistent")
            inconsistent_layers[i] = list(model.modules())[i]
    
    # Remove the hooks when done
    for hook in hooks:
        hook.close()

    return inconsistent_layers