import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from retinanet import model, coco_eval, csv_eval
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--coco_path')
    parser.add_argument('--csv_train')
    parser.add_argument('--csv_classes')
    parser.add_argument('--csv_val')
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size',type=int, default=2)
    args = parser.parse_args()

    if args.dataset == 'coco':
        dataset_train = CocoDataset(args.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(args.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif args.dataset == 'csv':
        dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()])) if args.csv_val else None
    else:
        raise ValueError('Dataset type not understood, it must be csv or coco.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=AspectRatioBasedSampler(dataset_val, batch_size=args.batch_size, drop_last=False)) if dataset_val else None

    retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True) if args.depth == 50 else None  # Add conditions for other depths as needed

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    for epoch_num in range(args.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                print(f'Epoch: {epoch_num} | Iteration: {iter_num} | Classification loss: {float(classification_loss):1.5f} | Regression loss: {float(regression_loss):1.5f} | Running loss: {np.mean(loss_hist):1.5f}')
            except Exception as e:
                print(e)
                continue

        if args.dataset == 'coco':
            coco_eval.evaluate_coco(dataset_val, retinanet)
        elif args.dataset == 'csv' and args.csv_val:
            mAP = csv_eval.evaluate(dataset_val, retinanet)
            print(f'Epoch: {epoch_num} | mAP: {mAP}')
        
        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.module, f'{args.dataset}_retinanet_{epoch_num}.pt')

    torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':
    main()
