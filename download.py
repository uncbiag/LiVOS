import os
import gdown

# download 480p models
save_folder ='weights'
os.makedirs(save_folder, exist_ok=True)

# weights for livos-nomose-480p
weights_url = 'https://drive.google.com/uc?id=1tG_BxCTWp_o9YH0vBqZqLC9KBsEGSsaH'
output = f'{save_folder}/livos-nomose-480p.pth'
gdown.download(weights_url, output, quiet=False)

# weights for livos-nomose-ft-480p
weights_url = 'https://drive.google.com/uc?id=1ToIDo6PIYF7lQGfO4F7HuHneyatKGWnx'
output = f'{save_folder}/livos-nomose-ft-480p.pth'
gdown.download(weights_url, output, quiet=False)

# weights for livos-wmose-480p
weights_url = 'https://drive.google.com/uc?id=13FVuxcEwNRfY70PA3O9pOyPO7Gx7Zl5N'
output = f'{save_folder}/livos-wmose-480p.pth'
gdown.download(weights_url, output, quiet=False)


# uncomment below to download 1024p/2048p trained models.
# # weights for livos-nomose-1024p
# weights_url = 'https://drive.google.com/uc?id=1ZBY2IOOZOy8bBDUdblRiHvN6x7zWY4JU'
# output = f'{save_folder}/livos-nomose-1024p.pth'
# gdown.download(weights_url, output, quiet=False)

# # weights for livos-nomose-2048p
# weights_url = 'https://drive.google.com/uc?id=1ebAf0d7T9yRfm03pt96JqdxBadb27zV5'
# output = f'{save_folder}/livos-nomose-2048p.pth'
# gdown.download(weights_url, output, quiet=False)