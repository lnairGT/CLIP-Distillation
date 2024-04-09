from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from dataloaders import get_original_cifar10_dataloaders
from dataloaders import get_original_cifar100_dataloaders
from dataloaders import get_imagenet_dataloaders
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import config as CFG
from vit_model import VisionTransformer, TeacherViT, BaselineViT
from utils import AvgMeter, get_lr
import argparse
from cluster import get_avg_embed_data
import PIL


def train_epoch(
    args,
    model,
    train_loader,
    loss_fn,
    optimizer,
    lr_scheduler,
    step,
    teacher=None,
    centroids=None
):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for img, labels in tqdm_object:
        img = img.to(CFG.device)
        labels = labels.to(CFG.device)
        if args.train_type == "vanilla":
            assert teacher is not None
            teacher.eval()
            tf = transforms.Resize(224)
            with torch.no_grad():
                teacher_pred = teacher(tf(img))
            _, pred, _ = model(img)
            loss = loss_fn(pred, labels) + CFG.clip_loss_wt * loss_fn(pred, teacher_pred)
        else:
            if args.train_type == "teacher-KD":
                tf = transforms.Resize(224)
                assert teacher is not None
                teacher.eval()
                with torch.no_grad():
                    centroids = teacher(tf(img))
                teacher_labels = torch.eye(img.shape[0]).to(CFG.device)
            else:
                assert centroids is not None
                teacher_labels = torch.zeros(img.shape[0], centroids.shape[0]).to(CFG.device)
                teacher_labels[torch.arange(img.shape[0]), labels] = 1 

            embed_student, hard_labels, proj_centroids = model(img, centroids)
            assert proj_centroids is not None
            proj_centroids = proj_centroids / proj_centroids.norm(dim=1, keepdim=True)
            embed_student = embed_student / embed_student.norm(dim=1, keepdim=True)
            student_logits = embed_student @ proj_centroids.t()
            loss = CFG.clip_loss_wt * torch.nn.functional.cross_entropy(
                student_logits, teacher_labels
            )
            # Include loss function for actual predicted labels
            loss += torch.nn.functional.cross_entropy(hard_labels, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step == "batch":
            lr_scheduler.step()

        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter.avg


def valid_epoch(model, valid_loader, loss_fn):
    loss_meter = AvgMeter()
    correct = 0
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for img, label in tqdm_object:
        img = img.to(CFG.device)
        label = label.to(CFG.device)
        with torch.no_grad():
            _, pred, _ = model(img)
        loss = loss_fn(pred, label)
        pred = torch.argmax(pred, dim=1)
        correct += pred.eq(label.view_as(pred)).sum().item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    accuracy = 100.0 * correct / len(valid_loader.dataset)
    return loss_meter.avg, accuracy


def get_dataloaders(dataset_name, image_size=None):
    image_size = CFG.size if image_size is None else image_size
    if dataset_name == "CIFAR10":
        root = '/data/datasets'
        num_classes = 10
        train_loader, valid_loader = get_original_cifar10_dataloaders(
            root, train_bsz=CFG.batch_size, val_bsz=CFG.batch_size, img_size=image_size
        )
    elif dataset_name == "CIFAR100":
        root = '/data/datasets'
        num_classes = 100
        train_loader, valid_loader = get_original_cifar100_dataloaders(
            root, train_bsz=CFG.batch_size, val_bsz=CFG.batch_size, img_size=image_size
        )
    else:
        root = '/data/datasets/imagenet'
        num_classes = 1000
        train_loader, valid_loader = get_imagenet_dataloaders(
            root, train_bsz=CFG.batch_size, val_bsz=CFG.batch_size, img_size=image_size
        )
    return train_loader, valid_loader, num_classes


def print_config():
    print("Batch size: ", CFG.batch_size)
    print("Learning rate: ", CFG.lr)
    print("Weight decay: ", CFG.weight_decay)
    print("Number of epochs: ", CFG.epochs)
    print("Device: ", CFG.device)
    print("Student Image size: ", CFG.size)
    print("Student Patch size: ", CFG.patch_sz)
    print("Student Width: ", CFG.width)
    print("Student Number of layers: ", CFG.layers)
    print("Student Number of heads: ", CFG.heads)


def main(args):
    assert args.dataset_name in ["CIFAR10", "CIFAR100", "Imagenet"]
    print(f"Using dataset: {args.dataset_name}")
    print("Logging results to: ", args.log_folder)
    assert args.train_type in ["vanilla", "teacher-KD", "embed-KD"]
    print("Training type: ", args.train_type)
    if args.train_type == "teacher-KD" or args.train_type == "embed-KD":
        print("Teacher model name: ", args.teacher_model)
        if args.train_type == "embed-KD":
            print("Cluster loss Wt: ", CFG.clip_loss_wt)
    print("**** Config details ****")
    print_config()
    writer = SummaryWriter(args.log_folder)

    centroids = None
    if args.train_type == "teacher-KD" or args.train_type == "embed-KD":
        teacher = TeacherViT(model_name=args.teacher_model)
        teacher.to(CFG.device)
        output_dim = teacher.model.config.hidden_size

        if args.train_type == "embed-KD":
            train_loader, _, num_classes = get_dataloaders(args.dataset_name, image_size=224)
            centroids, buf = get_avg_embed_data(
                teacher.to(CFG.device),
                train_loader,
                CFG.device,
                num_classes=num_classes,
                num_samples_per_class=100,
                viz=False
            )

            if buf is not None:
                image = PIL.Image.open(buf)
                image = ToTensor()(image)
                writer.add_image("Centroid visualization", image)

            del teacher
            teacher = None

        # Get dataloaders for student
        train_loader, valid_loader, num_classes = get_dataloaders(args.dataset_name)
        student = VisionTransformer(
            input_resolution=CFG.size,
            patch_size=CFG.patch_sz,
            width=CFG.width,  # Embed dim
            layers=CFG.layers,
            heads=CFG.heads,
            num_classes=num_classes,
            output_dim=output_dim  # Converts teacher dim to embed dim
        )
        student = student.to(CFG.device)
    else:
        teacher = BaselineViT(args.teacher_model)
        # Get dataloaders for student
        output_dim = teacher.model.config.hidden_size
        train_loader, valid_loader, num_classes = get_dataloaders(args.dataset_name, image_size=32)
        student = VisionTransformer(
            input_resolution=CFG.size,
            patch_size=CFG.patch_sz,
            width=CFG.width,  # Embed dim
            layers=CFG.layers,
            heads=CFG.heads,
            num_classes=num_classes,
            output_dim=output_dim  # Converts teacher dim to embed dim
        )
        student = student.to(CFG.device)
        teacher = teacher.to(CFG.device)
    
    # Get dataloaders for student
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.epochs)
    step = "epoch"

    best_acc = 0.0
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        student.train()
        train_loss = train_epoch(args, student, train_loader, loss_fn, optimizer, lr_scheduler, step, teacher, centroids)
        writer.add_scalar("Train loss", train_loss, epoch + 1)
        student.eval()
        with torch.no_grad():
            valid_loss, valid_acc = valid_epoch(student, valid_loader, loss_fn)
            writer.add_scalar("Val loss", valid_loss, epoch + 1)
            writer.add_scalar("Val acc.", valid_acc, epoch + 1)
            print("Accuracy at epoch {} is {}%".format(epoch + 1, valid_acc))
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(student.state_dict(), args.ckpt_save_name)
            print("Saved Best Model!")

    print(f"Best accuracy: {best_acc}%")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train-type", type=str, help="Distillation type: regular, teacher-KD, or embed-KD")
    argparser.add_argument("--teacher-model", type=str, help="Teacher model to use")
    argparser.add_argument("--dataset-name", type=str, help="Dataset to use: CIFAR10, CIFAR100, ImageNet")
    argparser.add_argument("--log-folder", type=str, help="Log folder to save Tensorboard files")
    argparser.add_argument("--ckpt-save-name", type=str, help="Filename for the best model checkpoint")
    args = argparser.parse_args()
    main(args)
