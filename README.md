# Eye Fixation Project


## Project requirements and instructions
To successfully pass the course project, you must:

1. Implement and train an eye fixation prediction system using PyTorch.
2. Submit a written description of your system, along with your source code and results.

### Deadlines
Code and results submission deadline: June 12th, 2020, at 23:59.

### Guidelines
- You may use any neural network architecture to implement the system. References from Pan et al. (2017); Kummerer et al. (2017); Cornia et al. (2016); Cornia et al. (2018); Kruthiventi et al. (2017) can provide useful ideas.
- You may use any dataset except the given test set to train your system.
- Transfer learning is allowed.
- Do not copy existing implementations; original work is required.

## Dataset
The dataset is available on the course Moodle page, divided into:

- Training: 3006 images and eye fixation maps
- Validation: 1128 images and eye fixation maps
- Testing: 1032 images
Each image is of size 224-by-224 with 3 channels, and each fixation map is of size 224-by-224 with one channel.

## Results
|            | ROC-AUC                |
|------------|------------------------|
| Best (2022)| 0.743304130918697     |
| Baseline   | 0.646241722479868     |


## How to submit the results?
- Predicted eye fixation maps should be saved as 224-by-224 PNG images with a single channel.
- Archive the predicted maps as lastname_firstname_predictions.zip.
- Submit source code as lastname_firstname_source.zip.
- Prepare a system summary in a PDF named lastname_firstname_summary.pdf.
- Upload all files through Moodle before the deadline.

## References
- Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, and Rita Cucchiara. A Deep Multi-Level Network for Saliency Prediction. In International Conference on Pattern Recognition (ICPR), 2016.
- Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, and Rita Cucchiara. Predicting human eye fixations via an lstm-based saliency attentive model. IEEE Transactions on Image Processing, 27(10):5142–5154, October 2018.
- S. S. S. Kruthiventi, K. Ayush, and R. V. Babu. Deepfix: A fully convolutional neural network for predicting human eye fixations. IEEE Transactions on Image Processing, 26(9):4446–4456, 2017.
- Matthias Kummerer, Thomas S. A. Wallis, Leon A. Gatys, and Matthias Bethge. Understanding low- and high-level contributions to fixation prediction. In The IEEE International Conference on Computer Vision (ICCV), Oct 2017.
- Junting Pan, Cristian Canton, Kevin McGuinness, Noel E. O’Connor, Jordi Torres, Elisa Sayrol, and Xavier and Giro-i Nieto. Salgan: Visual saliency prediction with generative adversarial networks. In arXiv, January 2017.
