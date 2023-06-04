import os
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete


def segmentation(input_location, output_location):
    # using cpu
    ctx = mx.cpu(0)

    # Loading the input images
    ListImages = sorted(os.listdir(input_location))  # Create list of images names

    for imgName in ListImages:
        img = image.imread(input_location + imgName)
        # plt.imshow(img.asnumpy())
        # plt.show()
        img = test_transform(img, ctx)
        model = gluoncv.model_zoo.get_model('psp_resnet101_citys', pretrained=True)
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        mask = get_color_pallete(predict, 'citys')
        mask.save(output_location + imgName)
