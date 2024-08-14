import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.datasets import MovieLens1M

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


class MovieLensMovieData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        **kwargs
    ) -> None:

        raw_movie_lens = MovieLens1M(root=root, *args, **kwargs)
        raw_movie_lens.process()

        data = torch.load(root + PROCESSED_MOVIE_LENS_SUFFIX)
        self.movie_data = data[0]["movie"]["x"]

    def __len__(self):
        return self.movie_data.shape[0]

    def __getitem__(self, idx):
        return self.movie_data[idx, :]

class MovieLensMovieData_from_embeddings(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        **kwargs
    ) -> None:

        with open(f'{root}/processed/embeddings.pkl', 'rb') as f:
            self.embeddings_dict = pickle.load(f)
            self.ids = list(self.embeddings_dict.keys())
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings_dict[self.ids[idx]])