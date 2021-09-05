# Text2Img with S-IntroVAE

## Data
- image download from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- preprocessed char-CNN-RNN text embeddings: https://drive.google.com/file/d/0B3y_msrWZaXLaUc0UXpmcnhaVmM/view?resourcekey=0-jxKV2PqKHFdfLN7Q93SuxQ

### Data details
- Training
  - 7034 images
  - high quality image shape (304, 304, 3) # (high, weight, channel)
- Testing
  - 1155 images
  - image shape (304, 304, 3) # (high, weight, channel)

### pickle
- Data/flowers/train
  - 304images.pickle: high quality images with shape (304, 304, 3)
  - 76images.pickle: low quality images with shape (76, 76, 3)
  - char-CNN-RNN-embeddings.pickle: vector which is the output of char-CNN-RNN-embeddings with shape (10, 1024) with 7034 images
  - class_info.pickle: show that the images are belonged to which classes
  - filenames.pickle: show that the images filename with total 7034 images

## Model
![Alt text](https://github.com/James24029775/DLP_Project_S-IntroVAE/blob/master/models/Network.png)
