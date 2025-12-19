import json
import torch
import torch.nn.functional as F
from binaryFeatureExtractor import BinaryFeatureExtractor

def main():
    # --- Configuration ---
    weights_path = "./artifacts/thirdOutput/binary_extractor.pt"
    json_path = "./clip_labels.json"
    output_path = "./binaryEmbeddings/thirdOutput/binary_results.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the checkpoint to get config and weights
    checkpoint = torch.load(weights_path, map_location=device)
    config = checkpoint["config"]

    # 2. Initialize the model with saved parameters
    model = BinaryFeatureExtractor(
        model_name=config["model_name"],
        num_of_hidden_layers=config["num_hidden_layers"],
        last_layer_dimension=config["last_dim"],
        device=device
    )

    # 3. Load the trained weights into the hidden layers
    model.hidden_layers.load_state_dict(checkpoint["hidden_layers_state"])
    model.eval()
    print(f"Model loaded successfully from {weights_path}")

    # 4. Load your dataset JSON
    with open(json_path, "r") as f:
        dataset = json.load(f)

    results = []

    # 5. Process each entry
    with torch.no_grad():
        for entry in dataset:
            img_name = entry["image"]
            # Convert list to tensor and add batch dimension
            clip_emb = torch.tensor(entry["clip_embedding"], dtype=torch.float32).to(device)
            
            # Normalize if your training used normalized embeddings
            clip_emb = F.normalize(clip_emb, dim=0).unsqueeze(0)

            # Pass through the trained hidden layers manually 
            x = model.forward_clip_embedding(clip_emb)
             
            # Convert to list for JSON storage
            binary_list = x.squeeze(0).cpu().numpy().tolist()

            results.append({
                "image": img_name,
                "category_id": entry.get("category_id"),
                "binary_embedding": binary_list
            })

    # 6. Save results
    with open(output_path, "w") as f:
        json.dump(results, f)
    
    print(f"Binarization complete. Saved to {output_path}")

if __name__ == "__main__":
    main()