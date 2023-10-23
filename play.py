import numpy as np
import torch
from models import CNNModel_BaseV8

class Sudoku():
    def __init__(self):
        #self.grid = np.random.randint(0, 9, size=(9, 9))
        self.grid = np.zeros((9, 9), dtype=float)
        self.create_grid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        
    def insert_row(self, row_number):
        """row = input(f"Insert the row{row_number} (9 numbers, space-separated):\n")
        
        # slice the row
        assert len(row) == 9*2, "wrong length"
        row_list = np.zeros(9)
        for i, number in enumerate(row.split()):
            number = float(number)
            assert (number >= 0.0 and number < 10.0), "the numbers must be between 0 and 9"
            row_list[i] = number"""
            
        row = input(f"Enter row {row_number + 1} (9 numbers, space-separated): ")
        row = np.fromstring(row, dtype=int, sep=' ')
        
        if len(row) != 9:
            print("Please enter 9 numbers for each row.")
            return self.insert_row(row_number)  # Recurse to re-enter the row
        if row.max() > 9 or row.min() < 0:
            print("The numbers must be between 0 and 9")
            return self.insert_row(row_number)  # Recurse to re-enter the row
        
        self.grid[row_number, :] = row
        
    def create_grid(self):
        print("The input format should be like this: 1 2 3 4 5 6 7 8 9, with unknows as 0")
        for i in range(9):
            self.insert_row(i)
            
    """def input_sudoku(self):
        sudoku = np.zeros((9, 9), dtype=int)  # Initialize an empty 9x9 Sudoku grid

        print("Enter the Sudoku puzzle, row by row. Use 0 for empty cells.")
        
        for i in range(9):
            row = input(f"Enter row {i + 1} (9 numbers, space-separated): ")
            row = np.fromstring(row, dtype=int, sep=' ')
            
            if len(row) != 9:
                print("Please enter 9 numbers for each row.")
                return self.input_sudoku()  # Recurse to re-enter the row
            if row.max() > 9 or row.min < 0:
                print("The numbers must be between 0 and 9")
                return self.input_sudoku()  # Recurse to re-enter the row

            
            sudoku[i, :] = row
        
        return sudoku"""


            
    def init_model(self):
        seg_dict = torch.load('models/Sudoku/2023-10-09_21.45.40_CNNModel_BaseV8_newNorm.best.state', map_location='cpu')
        model = CNNModel_BaseV8()
        model.load_state_dict(seg_dict['model_state'])
        model.eval()
        # set parameters to no gradient
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.to(self.device)
        #print(model)
        return model
    
    def print_grid(self, grid):
        [print(i) for i in grid]
        
    
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
    
    def prepare_data(self):
        grid_norm = self.normalize(self.grid)
        grid_norm = torch.unsqueeze(torch.from_numpy(grid_norm), dim=0)
        grid_norm = torch.unsqueeze(grid_norm, dim=0)
        grid_norm = grid_norm.to(self.device)
        
        return grid_norm.float()
    
    def solve_sudoku(self):
        grid = self.prepare_data()
        #print(grid.shape) torch.Size([1, 1, 9, 9])
        

        with torch.no_grad():
            output = self.model(grid)
            
        predictions = torch.argmax(output, dim=1) + 1
        self.print_grid(predictions)
        
    def main(self):
        print()
        print("Input grid:")
        self.print_grid(self.grid)
        print()
        print("Solved:")
        print()
        self.solve_sudoku()
        
if __name__ == '__main__':
    app = Sudoku()  # クラスのインスタンスを作成
    app.main()  # インスタンスからメソッドを呼び出す
