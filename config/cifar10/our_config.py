#model
backbone = dict(
    type="ResNet",
    depth=18,
    num_classes=512,
    maxpool=False
)
model = dict(
    type='OUR'
)
loss = dict(
    type='OurLoss',
    gather_distributed=True
)
#data
data_dir = './mydata/cifar-10'
num_workers = 4 #cpu for dataloader, 可用GPU數量的4倍(根據經驗法則)，太大或太小會減慢速度
batch_size = 2048
data = dict(
    type='Cifar10Dataset',
    root = data_dir,
    train=True,
    download=False,
    transform = dict(
        type = 'SimCLRTransform',
        input_size=32,
        gaussian_blur=0.
    )
)

#training
epochs = 500
lr = 0.5
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)

lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    #start_step=0,
    warmup_steps=0, # 100
    #warmup_from=1e-6


)

#log & save
log_interval = 20
save_interval = 100
work_dir = './experiment/cifar10/our'
resume = None
load = None
port = 10001
