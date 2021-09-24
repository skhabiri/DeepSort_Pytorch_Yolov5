Our training and test data are under
data/train
data/test

torchvision dataset requires a file structure such as:

train/class1
train/class2
...

test/class1
test/class2
...

under each class directory  there are cropped images of that class.

A multilabel torchvision dataset class can be defined as a json file,
`train_labels.json` containing the image file name and bounding box coordinate
of the object and class id.
