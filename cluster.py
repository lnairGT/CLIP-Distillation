import torch
import io
from sklearn import manifold
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_avg_embed_data(
        model,
        train_dataloader,
        device,
        num_classes=10,
        viz=False,
        num_samples_per_class=20
):
    data = []
    i = 0
    rem_samples = {k: num_samples_per_class for k in range(num_classes)}
    for img, label in tqdm(train_dataloader, desc="Getting embeddings"):
        embed = model(img.to(device))  # Batch_size x embed_dim
        label = label.to(device)
        embeds_per_class = []
        labels_per_class = []
        for c in range(num_classes):
            idx = label == c
            if rem_samples[c] == 0:
                continue
            
            embed_c = embed[idx, ...]
            label_c = label[idx, ...]

            if embed_c.shape[0] > rem_samples[c]:
                embed_c = embed_c[:rem_samples[c], ...]
                label_c = label_c[:rem_samples[c], ...]
                
            rem_samples[c] -= embed_c.shape[0]
            embeds_per_class.append(embed_c)
            labels_per_class.append(label_c)
        
        embed = torch.cat(embeds_per_class, dim=0)
        label = torch.cat(labels_per_class, dim=0)
        batch = torch.cat([embed, label.unsqueeze(-1)], dim=-1)
        if i == 0:
            data = batch
        else:
            data = torch.cat([data, batch], dim=0)
        i += 1
        if sum(list(rem_samples.values())) == 0:
            break

    data, labels = data[..., :-1], data[..., -1]
    labels = labels.to(device)
    init_centroids = []
    for i in range(num_classes):
        idx = labels == i
        init_centroids.append(torch.mean(data[idx, ...], dim=0))
    init_centroids = torch.stack(init_centroids, dim=0).to(device)

    buf = None

    if viz:
        embeddings = torch.cat(
            [data.cpu(), torch.Tensor(init_centroids).cpu()], dim=0
        )
        tsne = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            n_iter=300,
        )
        Y = tsne.fit_transform(embeddings.cpu().numpy())
        centroids = Y[-num_classes:, ...]
        Y = Y[:-num_classes, ...]
        for i in range(num_classes):
            idx = labels == i
            plt.scatter(Y[idx.cpu(), 0], Y[idx.cpu(), 1], label=f"class: {i}")

        plt.scatter(centroids[..., 0], centroids[..., 1], marker="X", color="black")
        plt.title("ViT model's embeddings for the data (tSNE)")
        plt.savefig("Visualize_embed.png")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

    return init_centroids, buf

def get_embed_distribution(
    model,
    train_dataloader,
    device,
    num_classes=10
):
    data = []
    i = 0
    for img, label in train_dataloader:
        embed = model(img.to(device))  # Batch_size x embed_dim
        label = label.to(device)
        batch = torch.cat([embed, label.unsqueeze(-1)], dim=-1)
        if i == 0:
            data = batch
        else:
            data = torch.cat([data, batch], dim=0)
        i += 1
        if i == num_classes:
            break

    data, labels = data[..., :-1], data[..., -1]
    labels = labels.to(device)
    dist_params_mean = []
    dist_params_std = []
    for i in range(num_classes):
        idx = labels == i
        # Get distribution mean and std
        mean = torch.mean(data[idx, ...], dim=0) #, torch.std(data[idx, ...], dim=0)
        std = torch.std(data[idx, ...], dim=0)
        dist_params_mean.append(mean)
        dist_params_std.append(std)

    return torch.stack(dist_params_mean, dim=0), torch.stack(dist_params_std, dim=0)
