from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, seqs, grades):
        super(MyDataset, self).__init__()
        self.seqs = seqs
        self.grades = grades

    def __getitem__(self, index):
        seq = self.seqs[index]
        grade = self.grades[index]
        return seq, grade

    def __len__(self):
        return len(self.seqs)
