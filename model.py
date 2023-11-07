import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    def __init__(self,
                in_channels:int=3,
                patch_size:int=16,
                embedding_dim:int=768):
        super().__init__()
        pass
# 1. Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        #3. make the image size is divisible by the patch size
        assert img_size%patch_size==0, f"Image size must be divisible by the patch size, image size:{img_size}, patch size:{patch_size}"

        self.num_patches=(img_size//patch_size)**2

        #5. create learnable class embeddings 
        self.class_embedding=nn.Parameter(data=torch.randn(1,1,embedding_dim),requires_grad=True)

        #6. creating learnable positional embeddings
        self.position_embedding=nn.Parameter(data=torch.rand(1,self.num_patches+1,embedding_dim),requires_grad=True)

        #7. create an embedding dropout value
        self.embedding_dropout=nn.Dropout(p=embedding_dropout)

        #8. create patch embedding layer
        self.patch_embedding=PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        

    

