from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['init_model_path'] = '../models/resnet50.pth'
opts['model_path'] = '../models/mdnet_res_imagenet.pth'

opts['batch_frames'] = 8
opts['batch_pos'] = 32 / 2
opts['batch_neg'] = 96 / 2

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 224
opts['padding'] = 16

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['conv', 'fc', 'layer']
opts['lr_mult'] = {'fc': 10}
opts['n_cycles'] = 50
