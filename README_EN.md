[[Japanese](README.md)/English]

# CNN for solving sudokus

## Introduction
Sudoku puzzles are a well-known and popular pastime. The game is played on a 9x9 grid where players must fill in the grid's cells with numbers from 1 to 9. The challenge lies in ensuring that no number is repeated within the same row or column. In this project, our primary objective is to investigate whether a convolutional neural network can successfully solve Sudoku puzzles.

## Task Analysis
This task can be likened to a k-dimensional classification problem where the model is tasked with filling in the missing numbers within a 9x9 matrix, utilizing values ranging from 1 to 9. However, due to computers' zero-based indexing, the model predicts classes in the range of 0 to 8. This problem is analogous to a segmentation problem where the model predicts a class for each pixel across the entire image.
## Requirements

- pytorch==2.0.1
- pytorch-cuda==11.8
- numpy==1.25.2


## Files explanation
- `dsets.py`：Creates the dataset.
- `training.py`：Used for training, main file of the project
- `models.py`: Includes all the models.
- `play.py`: Input your own sudoku, for model testing。

## Training
To set up training, adjust the path in `dsets.py` to point to the location where the dataset is stored. Afterwards, modify the tensorboard comments to monitor the training metrics. In your terminal, initiate training by running the following command:
```bash
python -m training.py
```

## Dataset
The dataset utilized for this project was sourced from Kaggle, containing a vast collection of 1 million Sudoku games. For preprocessing, we converted the data into NumPy arrays and transformed them into tensors. Given that this project employs a convolutional neural network, we normalize the input data to the range of [-1, 1] to expedite and simplify the training process.

## Training Loop
The program structure is organized into classes, with essential tasks accomplished through functions, providing the flexibility for further customization. The majority of the functions are responsible for executing the training loop, but one of the most significant functions is `computeBatchLoss`.

### Loss Function
In this project, we make use of the PyTorch `CrossEntropyLoss` as the loss function. This loss function takes logits as inputs, applies the `SoftMax` function to compute class probabilities, and selects the class with the highest probability. In our case, it takes the 9x9x9 output, where applying `SoftMax` results in a 9x9 matrix that can be compared with the target tensors.

For example, if we consider the first element at [0,0], among the extra 9 dimensions, the second dimension holds the highest probability. Consequently, the model would output 1 for the position [0,0].

### Performance Metrics
To gauge the model's performance, we employ various metrics, including the loss metric, along with others such as accuracy, precision, recall, and F1 score. These metrics provide a more comprehensive evaluation of the model's performance.

However, it's important to note that the performance measurements conducted in this project may not be perfect. In Sudoku, some numbers are initially provided, which means the model is not predicting all the numbers in the 9x9 grid. To obtain a more realistic accuracy measurement for the model, we should exclude the numbers known from the start when calculating metrics.

For instance, even if the model predicts everything incorrectly, as it starts with approximately 20 known numbers, the accuracy would be calculated as 20 divided by 81. Thus, the model's baseline accuracy is typically around 25%.

## Models
The primary objective of this project is to assess the performance of Convolutional Neural Networks (CNNs). Consequently, the models used are CNNs. An important distinction to be aware of is that, unlike CNNs employed for image recognition, these models do not alter the dimensions of the 9x9 grid. In fact, they introduce additional dimensions to the grid, and the final step involves utilizing a kernel size 1 convolutional layer.

This last convolutional layer with a kernel size of 1 reduces the dimensions of the additional channels, yielding an output tensor with a shape of 9x9x9. The first 9 dimensions represent each class, enabling the loss function to select the class with the highest probability.

The top-performing model in this project boasts around 18 million parameters. In this task, considering the substantial number of samples, it appears that augmenting the number of parameters contributes to improved performance.

## Conclusion

Detailed information about the project of solving Sudoku using Convolutional Neural Networks is provided within this GitHub repository. By employing convolutional layers, it becomes feasible to tackle Sudoku puzzles, and the repository includes a wealth of information regarding data processing and learning methodologies.

The three graphs presented depict accuracy, F1 score, and loss, in that order. The blue line represents the validation data, while the red one represents the training data.
### Accuracy
![制度](results/Accuracy.svg)

### F1 Score
<img src="results/F1 Score.svg" alt="F1 Score"/>

### Loss
![Loss](results/Loss.svg)

For more details, please refer to the project's files and code. If you have any questions or suggestions, feel free to reach out.

## Author
[fuwafuwamoemoekissaten](https://github.com/fuwafuwamoemoekissaten)

## References
https://github.com/Kyubyong/sudoku

https://www.kaggle.com/datasets/bryanpark/sudoku

Stevens, E., Antiga, L., Viehmann, T., & Chintala, S. (2020). Deep learning with pytorch: Build, train, and tune neural networks using python tools. Manning Publications.

## LICENSE
This project is licensed under the MIT license. Details are in the [LICENSE.md](LICENSE) file.