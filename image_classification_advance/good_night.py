
from inception_net_v1 import main as inception
from minception import main as minception
from pretrained_inception_resnet_v2 import main as inception_resnet_v2
from utils import init_gpus

if __name__ == '__main__':
    init_gpus()
    inception_resnet_v2()
    inception()
    minception()
