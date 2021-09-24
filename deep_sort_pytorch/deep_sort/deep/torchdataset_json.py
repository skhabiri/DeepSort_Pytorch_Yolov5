import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image

from torchvision.io.image import read_image
from torch.utils.data import Dataset, DataLoader, random_split



# def msg(name=None):
#     return """
#         python ./yolo_To_torch_multilbl.py -y ./yolo_format -t ./torch_dataset_ml -r 0.2
#         """
#
# parser = argparse.ArgumentParser(usage=msg())
# parser.add_argument('-y', '--yolo', help='path to yolo format directory',
#                     const="./yolo_format", default="./yolo_format", dest="ypath")
# parser.add_argument('-t', '--torch', nargs='?', help='torch dataset output directory', type=str,
#                     const="./torch_dataset_ml", default="./torch_dataset_ml", dest="tpath")
# parser.add_argument('-r', '--ratio', help='ratio of test to train split', dest="ratio", type=float)
# parser.add_argument('-wh', '--size', nargs='+', help='desired cropped size, w, h', dest="size", type=int)
#
#
# args = vars(parser.parse_args())

# ypath = args["ypath"]
# tpath = args["tpath"]
# ratio = args["ratio"]
# dim = tuple(args["size"])

# img_path = ypath + '/images'
# label_path = ypath + '/labels'
# label_path = os.path.join(ypath, 'labels')

# label_dict = {"0": "bird", "1": "flatwing", "2": "quadcopter"}


def labelfile_to_df(ypath):
    """
    structures the content of label files into a single dataframe
    label_path: path to label files in yolo format; ex: './yolo_format/labels'

    return:
    lf_df: A dataframe. Each row contains contents of one label file.
    dataframe columns are ['Image', 'Class', 'xc', 'yc', 'width', 'height']
    """

    # list of dataframes
    df_lst = []
    label_path = os.path.join(ypath, 'labels')
    label_files = os.listdir(label_path)
    label_files = [filename for filename in label_files if not filename.startswith('.')]

    for lf in label_files:
        file = pd.read_csv(os.path.join(label_path, lf), header=None, sep=' ')
        file['Image'] = lf.replace('.txt', '.jpg')
        df_lst.append(file)

    # convert the list to a vertical stacked dataframe
    lf_df = pd.concat(df_lst, axis=0, ignore_index=True)
    lf_df.columns = ['Class', 'xc', 'yc', 'width', 'height', 'Image']
    lf_df = lf_df[['Image', 'Class', 'xc', 'yc', 'width', 'height']]

    return lf_df


def df_to_json(df, jasonfile, ratio):
    """
    dumps the dataframe to a json file as jasonfile.
    ratio: test/train+test ratio

        labels = [{'Image': 'frame_000000.jpg',
          'Class': [0, 0, 2, 2, 1, 1],
          'xc': [0.0143, 0.0467, 0.1238, 0.6733, 0.616, 0.9267],
          'yc': [0.2799, 0.3797, 0.4332, 0.5037, 0.531, 0.7879999999999999],
          'width': [0.0283, 0.0409, 0.0476, 0.0488, 0.0522, 0.0589],
          'height': [0.0504, 0.0564, 0.0625, 0.0585, 0.0605, 0.0625]},
         {'Image': 'frame_000001.jpg',
          'Class': [0, 0, 2, 2, 1, 1],
          'xc': [0.0143, 0.0467, 0.1277, 0.6719, 0.6158, 0.9285],
          'yc': [0.2798, 0.3798, 0.4338, 0.5034, 0.531, 0.7891],
          'width': [0.0283, 0.0408, 0.0476, 0.0488, 0.0522, 0.0589],
          'height': [0.0504, 0.0564, 0.0625, 0.0585, 0.0605, 0.0625]}]
        ]

    """

    df1 = df.groupby('Image').agg(lambda x: list(x)).reset_index('Image')

    msk = np.random.rand(len(df1)) < ratio
    df_test = df1[msk]
    df_train = df1[~msk]

    # Create a list of dictionaries for all records
    test_list = df_test.to_dict(orient='records')
    train_list = df_train.to_dict(orient='records')

    # dump the list of dictionaries into a jason file with sequential integer keys
    with open('test_' + jasonfile, 'w') as f1:
        f1.write(json.dumps(test_list))

    with open('train_' + jasonfile, 'w') as f2:
        f2.write(json.dumps(train_list))

    return


class JsonDataset(Dataset):
    """
    A class of multilabel dataset that is subclass of pytorch dataset
    return: image, classes, xcs, ycs, widths, heights
    """
    def __init__(self, ypath, json_file, transform):
        df = pd.read_json(json_file)
        self.json_df = df
        self.img_dir = os.path.join(ypath, 'images')
        self.transform = transform
        classitems = [item for sublist in df for item in sublist]
        self.classes = set(classitems)

    def __len__(self):
        return len(self.json_df)

    def __getitem__(self, idx):
        """
        for accessing list items, dictionary entries, array elements etc
        using an index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.json_df.loc[idx, 'Image'])
        # image = read_image(img_path)
        image = Image.open(img_path)

        classes = torch.LongTensor(self.json_df.loc[idx, 'Class'])
        xcs = torch.FloatTensor(self.json_df.loc[idx, 'xc'])
        ycs = torch.FloatTensor(self.json_df.loc[idx, 'yc'])
        widths = torch.FloatTensor(self.json_df.loc[idx, 'width'])
        heights = torch.FloatTensor(self.json_df.loc[idx, 'height'])

        if self.transform:
            image = self.transform(image)

        else:
            image = np.array(image)
            image = torch.from_numpy(np.transpose(image,(-1,0,1)))

        return image, classes, xcs, ycs, widths, heights


def collate_fn(batch):
    """
    batch : is a list of tuples of tensors(tensor1, tensor2, ...)
    with the length of batch size
    """
    images = []
    classes = []
    xcs = []
    ycs = []
    widths = []
    heights = []

    for b in batch:
        images.append(b[0])
        classes.append(b[1])
        xcs.append(b[2])
        ycs.append(b[3])
        widths.append(b[4])
        heights.append(b[5])

    images = torch.stack(images, dim=0)
    classes = torch.stack(classes, dim=0)
    xcs = torch.stack(xcs, dim=0)
    ycs = torch.stack(ycs, dim=0)
    widths = torch.stack(widths, dim=0)
    heights = torch.stack(heights, dim=0)

    return images, classes, xcs, ycs, widths, heights


def dataset_yolo(ypath, ratio, train_transform=None, test_transform=None):
    """
    converts multi label yolo to pytorch dataset
    ypath: yolo format source path with "images" and "labels" sub-directories
    ratio: split ratio for test data

    return: image, classes, xcs, ycs, widths, heights

    ex/ train_data, test_data = dataset_yolo("./yolo_format", 0.2, train_transform, test_transform)
    train_dataloader = DataLoader(train_data, transform=transform_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, transform=transform_test, batch_size=32, shuffle=True, collate_fn=collate_fn)

    """
    # create df from yolo formatted label files
    lf_df = labelfile_to_df(ypath)
    # create json file from labels dataframe
    df_to_json(lf_df, "labels.json", ratio)
    # instantiate a json dataset class
    train_data = JsonDataset(ypath, 'train_labels.json', train_transform)
    test_data = JsonDataset(ypath, 'test_labels.json', test_transform)
    # # split pytorch jasondataset object into train and test
    # train_size = int((1 - ratio) * len(data))
    # test_size = len(data) - train_size
    # train_data, test_data = random_split(data, [train_size, test_size])

    return train_data, test_data
