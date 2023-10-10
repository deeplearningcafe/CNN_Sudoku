import torch
import numpy as np
from torch.utils.data import Dataset




class Sudoku(Dataset):
    def __init__(self,
                 path="E:\Data\sudoku\sudoku.csv",
                 transform=None,
                 idx = None,
                 normalizeVal = False):
        
        self.path = path
        quizzes, solutions = self.init_dataset()
        if normalizeVal:
            quizzes = self.normalize(quizzes)
        
        # these are still narrays so we are using the index array to slice them
        quizzes = quizzes[idx]
        solutions = solutions[idx]
        
        # it is much faster to get the tensors from numpy than creating directly the tensors in the init
        self.quizzes = torch.from_numpy(quizzes)
        self.solutions = torch.from_numpy(solutions)
        self.transform = transform
        
    
    def init_dataset(self):
        quizzes = np.zeros((1000000, 81), np.float32)
        solutions = np.zeros((1000000, 81), np.int64)
        for i, line in enumerate(open(self.path, 'r').read().splitlines()[1:1000000]):
            quiz, solution = line.split(",")
            for j, q_s in enumerate(zip(quiz, solution)):
                q, s = q_s
                quizzes[i, j] = q
                solutions[i, j] = s

        quizzes = quizzes.reshape((-1, 9, 9))
        solutions = solutions.reshape((-1, 9, 9))
        solutions = solutions - 1
        # here solutions is correct, between 0 and 8
        
        return quizzes, solutions
    
    def normalize(self, quizzes):
        # divide by 9 as all the samples, as values range from 1 to 9
        # normalized_tensor = (input_tensor - min_value) / (max_value - min_value)
        # and by substracting 0.5 then the values will now range from -0.5 to 0.5.

        # this is 0 and 9
        min_value = quizzes.min()
        max_value = quizzes.max()

        # Normalize the tensor to the range [0, 1]
        normalized_tensor = (quizzes - min_value) / (max_value - min_value)

        # Center the normalized tensor around zero, [-0.5, 0.5]
        centered_tensor = normalized_tensor - 0.5
        return centered_tensor

        
    
    def __len__(self):
        return len(self.quizzes)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        quizzes = self.quizzes[index]
        # we add a dimension because the output of the model is [batch, 9, 9, 9], there are 9 classes so it need 9 dimensions
        quizzes = torch.unsqueeze(quizzes, dim=0)
        solutions = self.solutions[index]

        
        if self.transform:
            quizzes = self.transform(quizzes)
        
        return quizzes, solutions
        
