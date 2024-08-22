
from data.movie_lens import MovieLensMovieData, MovieLensMovieData_from_embeddings
from distributions.gumbel import TemperatureScheduler
from modules.rqvae import RqVae

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm




def train(
    # iterations=500000,
    epochs=100,
    batch_size=256,
    learning_rate=0.0001,
    #weight_decay=0.01,
    weight_decay=0,
    max_grad_norm=1,
    mem_ratio=0.1,
    dataset_folder="dataset/ml-1m",
    use_kmeans_init=False,
    clip_grad=False
):
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_per_process_memory_fraction(mem_ratio, device=0)
    dataset = MovieLensMovieData_from_embeddings(root=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RqVae(
        input_dim=768,
        # embed_dim=32,
        embed_dim=96,
        hidden_dim=32,
        codebook_size=18,
        n_layers=3
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = LinearLR(optimizer)


    temp_scheduler = TemperatureScheduler(
        t0=1,
        min_t=0.05,
        anneal_rate=0.0001,
        step_size=1
    )
    
    best_loss = float('inf')    
    model.to(device)
    for epoch in range(epochs):
        model.train()  # 設置模型為訓練模式

        t = temp_scheduler.get_t(epoch)
        # tqdm 進度條
        with tqdm(initial=0, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            total_loss = 0.0  # 累積損失

            for iter, data in enumerate(dataloader):
                if iter == 0 and use_kmeans_init and epoch == 0:
                    init_data = data.to(device)
                    model.kmeans_init(init_data)
                data = data.to(device)

                optimizer.zero_grad()  # 清除梯度

                
                loss = model(data, gumbel_t=t)
                total_loss += loss.item()

                loss.backward()

                # 梯度剪裁
                if clip_grad is True:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()  # 更新模型參數
                scheduler.step()  # 更新學習率

                # 更新 tqdm 進度條
                pbar.set_postfix({'loss': total_loss / (iter + 1)})
                pbar.update(1)
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), 'model/best_model.pth')


if __name__ == "__main__":
    train()
