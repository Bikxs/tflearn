
from inception_net_v1 import main as inception
from minception import main as minception
from pretrained_inception_resnet_v2 import main as inception_resnet_v2
from utils import init_gpus
import os
def shutdown():
    print("Shutting down")
    # Execute the shutdown command based on the operating system
    if os.name == "posix":  # Unix/Linux/Mac

        os.system("sudo shutdown -h now")
    elif os.name == "nt":  # Windows
        os.system("shutdown /s /f /t 0")
    else:
        print("Unsupported operating system")
if __name__ == '__main__':
    init_gpus()
    inception_resnet_v2()
    inception()
    minception()
