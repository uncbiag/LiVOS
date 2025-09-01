import os
import gdown

# download 480p models
save_folder ='weights'
os.makedirs(save_folder, exist_ok=True)

# weights for livos-nomose-480p
weights_url = 'https://drive.google.com/uc?id=1ZjhVMUppZ5OZipCALU9ZYwiVF5DEPBxX'
output = f'{save_folder}/livos-nomose-480p.pth'
gdown.download(weights_url, output, quiet=False)

# weights for livos-nomose-ft-480p
weights_url = 'https://drive.google.com/uc?id=1rUxSjKreh5Q_X7FHvFRyoKBR7GqJr04P'
output = f'{save_folder}/livos-nomose-ft-480p.pth'
gdown.download(weights_url, output, quiet=False)

# weights for livos-wmose-480p
weights_url = 'https://drive.google.com/uc?id=1lhfnRCe1yPfB9dcWSA6hv7ABPzHFuNCY'
output = f'{save_folder}/livos-wmose-480p.pth'
gdown.download(weights_url, output, quiet=False)


# uncomment below to download 1024p/2048p trained models.
# # weights for livos-nomose-1024p
# weights_url = 'https://drive.google.com/uc?id=1_NjVaumXO8IPsFocXdRAvs6hqd-TOSGH'
# output = f'{save_folder}/livos-nomose-1024p.pth'
# gdown.download(weights_url, output, quiet=False)

# # weights for livos-nomose-2048p
# weights_url = 'https://drive.google.com/uc?id=1Od-djH3AUDEXJmXmqygCUSrFWUmVyNBG'
# output = f'{save_folder}/livos-nomose-2048p.pth'
# gdown.download(weights_url, output, quiet=False)
