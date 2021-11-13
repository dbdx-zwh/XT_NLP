from torch.utils.data import Dataset, DataLoader

class test(Dataset):
    def __init__(self):
        self.data = []
        for i in range(0, 10):
            self.data.append((i, i + 100))
    
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    toy = test()
    data_loader = DataLoader(
        dataset = toy,
        batch_size=5,
        shuffle=True
    )
    for epoch in range(0, 5):
        print('-----------')
        for x, y in data_loader:
            print(x, y)