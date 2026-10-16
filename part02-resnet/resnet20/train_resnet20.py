import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm

from resnet20 import ResNet20

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_index, (images, labels) in enumerate(
        tqdm(data_loader, desc = "Training"),
        start = 1,
    ):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        
        optimizer.zero_grad(set_to_none = True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        batch_size = labels.size(0)
        
        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim = 1) == labels
        ).sum().item()
        total_samples += batch_size
        
    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy

def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval() #batchnorm使用训练期间累积的均值和方差 dropout停止随机丢弃神经元
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.inference_mode(): #关闭自动求导相关工作：不建立反向传播计算图、减少显存占用、提高推理速度
        for images, labels in data_loader:
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            batch_size = labels.size(0)
            
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim = 1) == labels
            ).sum().item()
            total_samples += batch_size
    
    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples
    
    return average_loss, average_accuracy

def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    checkpoint_dir = script_dir / "checkpoints"
    data_dir = project_dir / "data"

    best_validation_accuracy = -1.0

    history = {
        "train_loss":[],
        "train_accuracy":[],
        "validation_loss":[],
        "validation_accuracy":[],
        "learning_rate":[],
    }
    #transform
    train_transform = transforms.Compose([
        transforms.RandomCrop(
            size = 32,
            padding = 4,
            padding_mode = "reflect",
        ), #32x32->40x40 在随机裁剪回32x32
        transforms.RandomHorizontalFlip(p = 0.5), #有0.5概率水平翻转
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
        ),
    ])
    
    #base_dataset
    train_base_dataset = datasets.CIFAR10(
        root = data_dir,
        train = True,
        download = True,
        transform = train_transform,
    )

    validation_base_dataset = datasets.CIFAR10(
        root = data_dir,
        train = True,
        download = True,
        transform = eval_transform,
    )
    
    split_generator = torch.Generator().manual_seed(42)
    
    all_indices = torch.randperm(
        len(train_base_dataset),
        generator = split_generator,
    ).tolist()
    
    validation_size = 5000
    
    validation_indices = all_indices[:validation_size]
    train_indices = all_indices[validation_size:]
    
    #dataset
    train_dataset = Subset(
        train_base_dataset,
        train_indices,
    )
    
    validation_dataset = Subset(
        validation_base_dataset,
        validation_indices,
    )
    
    test_dataset = datasets.CIFAR10(
        root = data_dir,
        train = False,
        download = True,
        transform = eval_transform,
    )
    
    assert len(train_dataset) == 45000
    assert len(validation_dataset) == 5000
    assert len(test_dataset) == 10000
    
    assert set(train_indices).isdisjoint(
        validation_indices
    )
    
    print("training samples:", len(train_dataset))
    print("validation samples:", len(validation_dataset))
    print("test samples:", len(test_dataset))
    
    #Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size = 128,
        shuffle = True,
        num_workers = 8,
        pin_memory = True, # 使用锁页内存，可以提高cpu到gpu的数据传输的效率
        persistent_workers = True, # 表示每个epoch结束后不销毁数据读取进程
        prefetch_factor = 2,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size = 256,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True,
        prefetch_factor = 2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = 256,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = False,
        prefetch_factor = 2,
    )

    print("train batches:", len(train_loader))
    print("validation batches:", len(validation_loader))
    print("test batches:", len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
 
    model = ResNet20(
        num_classes = 10,
        zero_init_residual = True,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 1e-4
    )

    num_epochs = 200

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = [100, 150],
        gamma = 0.1,
    ) #阶梯下降
    
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
            model = model,
            data_loader = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
        )
    
        validation_loss, validation_accuracy = evaluate(
            model = model,
            data_loader = validation_loader,
            criterion = criterion,
            device = device,
        )
    
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"epoch {epoch:2d}/{num_epochs} | "
            f"lr {current_lr:.6f} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy * 100:6.2f}% | "
            f"val loss {validation_loss:.4f} | "
            f"val acc {validation_accuracy * 100:6.2f}%"
        )

        scheduler.step()
    
        is_best = validation_accuracy > best_validation_accuracy
        if is_best:
            best_validation_accuracy = validation_accuracy
    
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "history": history,
        }
    
        torch.save(
            checkpoint,
            checkpoint_dir / "last_checkpoint.pt",
        )
    
        if is_best:
            torch.save(
                checkpoint,
                checkpoint_dir / "best_checkpoint.pt",
            )

            print(
                f"  saved new best checkpoint: "
                f"{best_validation_accuracy * 100:.2f}%"
            )
    
    print("\nTraining finished.")
    print("Loading the best validation checkpoint...")
    
    best_checkpoint_path = (
        checkpoint_dir / "best_checkpoint.pt"
    )
    
    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location = device,
        weights_only = True,
    )
    
    best_model = ResNet20().to(device)
    
    best_model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )
    
    print(
        "best validation epoch:",
        best_checkpoint["epoch"],
    )
    
    print(
        "best validation accuracy:",
        f'{best_checkpoint["validation_accuracy"] * 100:.2f}%',
    )
    
    final_test_loss, final_test_accuracy = evaluate(
        model = best_model,
        data_loader = test_loader,
        criterion = criterion,
        device = device,
    )
    
    print(
        f"final test |"
        f"loss {final_test_loss:.4f} |"
        f"accuracy {final_test_accuracy * 100:.2f}%"
    )

if __name__ == "__main__":
    main()