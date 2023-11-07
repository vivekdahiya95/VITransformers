# VITransformers

## George Hotz 
### 1. Download a paper
### 2. Implement it
### 3. Keep doing it until you have skills
-  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## Getting setup
- datasets.ImageFolder
- torch.inference_mode()
- pathlib->Path
- os.walk
- read about *plot_decision_boundary* in the utils which comes from madewith ml
- torch.eq
- The subplot() function takes three arguments that describes the layout of the figure.The layout is organized in rows and columns, which are represented by the first and second argument.The third argument represents the index of the current plot.
- read deaply about how *mkdir* works
-  with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

## get the data
- read about pin memory 
- rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]

## replicating the architecture
- Patch + Position embeddings
    - for an image size of 224 and patch size of 16:
      Input (2D image): (224, 224, 3) -> (height, width, color channels)
      Output (flattened 2D patches): (196, 768) -> (number of patches, embedding dimension)
- Linear projection of flattened pathches
- Norm
- Multi head attention
- MLP
- Transformer Encoder
- MLP head







