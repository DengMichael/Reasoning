import operator
import requests
import hashlib
import warnings
from torch.utils.data import IterableDataset
from torchvision import transforms
from torch import nn
import os
import torch
import random
import torch.nn.functional as nnf
# dataset
NUMBERS = [str(i) for i in range(10)]
OPERATORS = ['+', '-', 'x']
EQUALS = ['=']
TRAIN_AMOUNT = 0.9


def download_file_from_google_drive(id, destination):
    # https://stackoverflow.com/a/39225039
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        f.write(response.content)


def baseN(num, b, D="0123456789abcdefghijklmnopqrstuvwxyz"):
    return (baseN(num // b, b) + D[num % b]).lstrip("0") if num > 0 else "0"


def op_base(op, base):
    def run(a, b):
        c = op(int(a, base), int(b, base))
        if c < 0:
            return '-'+baseN(-c, base)
        else:
            return baseN(c, base)
    return run


label_size = 0

NUMBERS = [str(i) for i in range(10)]
OPERATORS = ['+', '-', 'x']
EQUALS = ['=']
TRAIN_AMOUNT = 0.9




class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random.random() < self.prob else self.fn_else
        return fn(x)

class DREAMNISTDataset(IterableDataset):
    """
    Arguments:
    root (string): Root directory of dataset.
    train (bool, optional): If True uses training data, otherwise test data.
    image_size (int or tuple, optional): Output size of image (128, 64, or 32 are recommended).
    operators (list, optional): List of operators to choose from; subset of [+, -, x].
    base (int, optional): Numerical base of digits.
    num_digits (int, optional): Number of digits per number.
    square (bool, optional): If True places equations in square grids.
    download (bool, optional): If True, downloads dataset to root if not found.
    correct (bool, optional): If True, constructs correct equations otherwise only incorrect
        equations are made.
    """

    def __init__(self,
                 root,
                 train=True,
                 image_size=32,
                 operators=['+'],
                 base=10,
                 num_digits=1,
                 square=True,
                 download=False,
                 use_keep_out=True,
                 correct=True,
                 aug_prob=0.
                 ):
        self.root = root
        self.train = train
        self.indiv_image_size = 28
        self.image_size = image_size
        self.operators = list(set(operators))
        self.base = base
        self.num_digits = num_digits
        self.square = square
        self.grid_width = num_digits * 2 + 1
        self.grid_height = self.grid_width if square else 2
        self.use_keep_out = use_keep_out
        self.correct = correct
        self.transform = transforms.Compose([
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(
                0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size))
        ])

        if num_digits < 1:
            raise Exception('Number of digits must be at least 1.')
        if self.base < 2:
            raise Exception('Base must be at least 2.')
        if base > 10:
            raise Exception('Base must be less than or equal to 10')
        if not set(self.operators).issubset(OPERATORS):
            raise Exception('operators must be a subset of [+, -, x]')
        if not self.image_size in [32, 64, 128]:
            warnings.warn(
                'Recommended images sizes are 32, 64, and 128. Other image sizes may not function correctly.')

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = os.path.join(root, 'train.pt')
        else:
            data_file = os.path.join(root, 'test.pt')

        self.data, self.targets = torch.load(data_file)

        self.length = len(self.data)

        # get a list of indices corresponding to images that contain each symbol
        self.symbol_indices = {symbol: (self.targets == ord(symbol)).nonzero()
                               for symbol in NUMBERS + operators + EQUALS}

        # construct list of equations as strings and optionally remove equations from training set
        percentage = TRAIN_AMOUNT if use_keep_out and train else 1
        self.equations = []

        self.digit_range = base ** num_digits
        self.ans_range = 0

        for operator in operators:
            start = -(base ** num_digits) + 1 if operator == '-' else 0
            if operator == '-':
                end = base ** num_digits - 1
            if operator == '+':
                end = (base ** num_digits) * 2 - 1
            if operator == 'x':
                end = (
                    base ** num_digits) * (base ** num_digits) - 1

            global label_size
            self.ans_range = max(end - start, self.ans_range)
#            label_size = max(self.digit_range * 2 + len(operators), label_size) #removed ans_range
            label_size = max(label_size, self.ans_range)

            operator_hashes = []
            for first_num in range(base ** num_digits):
                first_num = baseN(first_num, base)
                for second_num in range(base ** num_digits):
                    second_num = baseN(second_num, base)
                    ans = self._get_operator_function(
                        operator)(first_num, second_num)
                    if not correct:
                        for third_num in range(start, end):
                            incorrect_ans = baseN(
                                third_num, base) if third_num >= 0 else '-' + baseN(third_num, base)
                            if ans != incorrect_ans:
                                equation_str = str(
                                    first_num) + operator + str(second_num) + '=' + incorrect_ans
                                equation_hash = hashlib.md5(
                                    equation_str.encode()).digest()
                                operator_hashes.append(
                                    (first_num, operator, second_num, incorrect_ans, equation_str, equation_hash))
                    else:
                        equation_str = str(first_num) + \
                            operator + \
                            str(second_num) + '=' + ans
                        equation_hash = hashlib.md5(
                            equation_str.encode()).digest()
                        operator_hashes.append(
                            (first_num, operator, second_num, ans, equation_str, equation_hash))
            operator_hashes = sorted(operator_hashes, key=lambda x: x[-1])
            self.equations.extend(
                operator_hashes[:int(percentage*len(operator_hashes))])

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, 'train.pt')) and
                os.path.exists(os.path.join(self.root, 'test.pt')))

    def download(self):
        if not self._check_exists():
            print("Downloading...")
            download_file_from_google_drive(
                'keyremoved', os.path.join(self.root, 'test.pt'))
            download_file_from_google_drive(
                'keyremoved', os.path.join(self.root, 'train.pt'))
            print("Done!")

    def _get_operator_function(self, op):
        return {
            '+': op_base(operator.add, self.base),
            '-': op_base(operator.sub, self.base),
            'x': op_base(operator.mul, self.base)
        }[op]

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        first_num, operator, second_num, ans, equation_str, _ = random.sample(
            self.equations, 1)[0]

        ans_val = int(ans)

        # choose random handwritten symbols
        first_num = [random.choice(self.symbol_indices[digit])
                     for digit in first_num]
        second_num = [random.choice(self.symbol_indices[digit])
                      for digit in second_num]
        operator = [random.choice(self.symbol_indices[operator])]
        equals = [random.choice(self.symbol_indices['='])]
        ans = [random.choice(self.symbol_indices[digit]) for digit in ans]

        # build equation
        equation = [first_num, operator, second_num, equals, ans]
        # flatten list
        equation = [item for sublist in equation for item in sublist]

        # define grid shape
        grid = torch.zeros(1, self.indiv_image_size*self.grid_height,
                           self.indiv_image_size*self.grid_width)

        # load images into grid
        for i, index in enumerate(equation):
            row, col = i // self.grid_width, i % self.grid_width
            img = self.data[index]
            grid[0, row*self.indiv_image_size:(row+1)*self.indiv_image_size,
                 col*self.indiv_image_size:(col+1)*self.indiv_image_size] = img
        grid = grid / 255.0

        # resize to 128x128
        grid = nnf.interpolate(grid.unsqueeze(0), size=(int(
            self.grid_height/self.grid_width*128), 128), mode='bilinear', align_corners=False).squeeze(0)

        # resize to requested size
        if self.image_size == 128:
            pass
        elif self.image_size == 64:
            grid = nnf.avg_pool2d(grid, kernel_size=2)
        elif self.image_size == 32:
            grid = nnf.avg_pool2d(grid, kernel_size=4)
        else:
            size = self.image_size if isinstance(
                self.image_size, tuple) else (self.image_size, self.image_size)
            grid = nnf.interpolate(grid.unsqueeze(
                0), size=size, mode='bilinear', align_corners=False).squeeze(0)

        grid = self.transform(grid)
        return grid, ans_val, equation_str