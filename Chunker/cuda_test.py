import torch
if torch.cuda.is_available(): 
    print("available")
else:
    print("not available")