import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitarray

#Load CLIP ViT-B/16 model
# For faster performance, you can use 'clip-ViT-B-32' instead


class BinaryFeatureExtractor(nn.Module):
    """
    This class defines a binary feature extractor using a pre-trained SentenceTransformer model.
    It includes several fully connected layers followed by Tanh activation functions,
    and a final Sign activation to produce binary outputs.
    
    The forward method takes input data (an image), processes it through the model and layers,
    and returns the binary feature representation.
    """
    def __init__(self, model_name='clip-ViT-B-32', num_of_hidden_layers=1, device='cpu', last_layer_dimension = 256):
        super(BinaryFeatureExtractor, self).__init__()
        
        # Seleccionar dispositivo: preferir MPS/CUDA si están disponibles
        self.device = device
        self.model = SentenceTransformer(model_name)
        # Mover el encoder al dispositivo elegido
        try:
            self.model.to(self.device)
        except Exception:
            # Algunos backends pueden no soportar .to() de forma directa; en tal caso, encode usará el device explícito
            pass

        emb_dim = None
        get_dim = getattr(self.model, 'get_sentence_embedding_dimension', None)
        if callable(get_dim):
            try:
                emb_dim = get_dim()
            except Exception:
                emb_dim = None
        if not emb_dim:
            try:
                # Inferir dimensión de embedding codificando una imagen dummy
                from PIL import Image as _PILImage
                dummy = _PILImage.new('RGB', (32, 32), color=(0, 0, 0))
                vec = self.model.encode(dummy, convert_to_tensor=True, device=self.device)
                emb_dim = int(vec.shape[-1])
            except Exception as e:
                raise ValueError("The model does not have a defined sentence embedding dimension.") from e

        self.num_of_hidden_layers = num_of_hidden_layers
        fully_connected_layers = []
        for _ in range(num_of_hidden_layers - 1):
            fully_connected_layers.append(nn.Linear(emb_dim, emb_dim))
        
        fully_connected_layers.append(nn.Linear(emb_dim, last_layer_dimension))
        self.hidden_layers = nn.ModuleList(fully_connected_layers)
        # Mover capas al dispositivo
        self.hidden_layers.to(self.device)
        # Este extractor se usa solo para inferencia; congelamos parámetros
        for p in self.hidden_layers.parameters():
            p.requires_grad_(False)
        self.eval()
        self.tanh = nn.Tanh()
        self.sign = torch.sign
    
    """
    Forward pass through the model and layers to obtain binary features.
    Starts by encoding an input image, then processes it through the defined fully connected layers
    with Tanh activations, and finally applies the Sign activation to produce binary outputs.
    """    
    def forward(self, x):
        # SentenceTransformer.encode usa inference_mode internamente y devuelve un "inference tensor".
        # Pasamos explícitamente el dispositivo y clonamos para obtener un tensor normal.
        try:
            x = self.model.encode(x, convert_to_tensor=True, device=self.device)
        except TypeError:
            # Versiones antiguas no aceptan parámetro device
            x = self.model.encode(x, convert_to_tensor=True)
        x = x.clone()
        if x.device.type != self.device:
            x = x.to(self.device)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.tanh(x)
        x = self.sign(x)
        return x

device = 'cpu'     
if torch.cuda.is_available():
  device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
  device = 'mps'

model = BinaryFeatureExtractor(model_name='clip-ViT-B-32', num_of_hidden_layers=2, device=device)
binaryVector1 = model.forward(Image.open('./src/twoDogsInSnow.jpeg'))
vec1 = binaryVector1.detach().cpu().numpy()
string1 = ''.join(['1' if i > 0 else '0' for i in vec1])

binaryVector2 = model.forward(Image.open('./src/dogsInSnow.jpeg'))
vec2 = binaryVector2.detach().cpu().numpy()
string2 = ''.join(['1' if i > 0 else '0' for i in vec2])


bitarray1 = bitarray.bitarray(string1)
bitarray2 = bitarray.bitarray(string2)
hamming_distance = (bitarray1 ^ bitarray2).count()
print(f'Hamming Distance between binary features: {hamming_distance}\n')


print(string1 + '\n' + '\n')   
print(string2)