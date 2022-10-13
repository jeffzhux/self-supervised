#config file
#model
backbone = dict(
    type="ResNet",
    depth=18,
    num_classes=10,
    maxpool=False
)
model = dict(
    type='Linear',
    depth = '18',
    num_classes=10
)

# data
data_dir = './mydata/cifar-10'
batch_size = 2048
num_workers = 4   #num of CPU's worker are 4 times than num of GPU
data = dict(
    train=dict(
        type='CIFAR10',
        root = data_dir,
        download = False,
        train=True,
        transform = dict(
            type='cifar_linear'
        )
    ),
    test=dict(
        type='CIFAR10',
        root = data_dir,
        download = False,
        train=False,
        transform = dict(
            type='cifar_linear'
        )
    )
)


#training optimizer & scheduler
epochs = 100
lr = 0.1
optimizer= dict(type='SGD', lr=lr, momentum=0.9, weight_decay = 0)
lr_cfg = dict( # passed to adjust_learning_rate()
    type='MultiStep',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    decay_steps=[60,80]
)

# log, load & save
log_interval = 20
work_dir = 'linear_experiment/simclr'
resume = None
load = './experiment/cifar10/simclr/20221011_214921/epoch_200.pth'
port = 10001