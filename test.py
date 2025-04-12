# Import required libraries and modules
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader, Dataset
from model.vgg import vgg16                 # Custom VGG16 model
from model.resnet import ResNetModel        # Custom ResNet model
# from dataset.dataset_retrieval import custom_dataset  # Dataset loading function
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm                       # For progress visualization

dataset = pd.read_csv('/Users/gultajseyid/Desktop/ai_project/asl_alphabet_project/dataset/data_labeled.csv')
test_df = dataset[dataset['mode'] == 'test'].reset_index()
test_df['class1'] = test_df['class'].astype('category')
test_df['class_code'] = test_df['class1'].cat.codes

# Test function to evaluate model on test dataset
def test(model, data_test, loss_function, writer, epoch):
    # Initialize metrics for evaluation
    f1 = F1Score(num_classes=36, task='multiclass', average='macro')  # Macro-averaged F1 for multi-class
    accuracy = MulticlassAccuracy(num_classes=36)                     # Multi-class accuracy metric

    data_iterator = enumerate(data_test)  # Create an iterator over the dataloader
    accuracy_list = []  # Predictions
    accuracyt_list = [] # Ground truth
    f1_list = []        # Predictions
    f1t_list = []       # Ground truth

    with torch.no_grad():  # No gradient calculation needed during evaluation
        model.eval()       # Switch model to evaluation mode
        tq = tqdm(total=len(data_test))  # Progress bar
        tq.set_description('Testing')

        total_loss = 0  # Accumulate total loss
        confidence_score = None  # To store last computed confidence score

        for _, batch in data_iterator:
            # Unpack batch
            image, label = batch
            image = image.cuda()
            label = label.cuda()

            # Forward pass through the model
            pred = model(image)

            # Convert logits to probabilities
            probs = softmax(pred, dim=1)

            # Compute loss
            loss = loss_function(pred, label.float())
            loss = loss.cuda()

            # Convert predictions to one-hot label indices
            pred = pred.softmax(dim=1)

            # Compute confidence score (maximum probability per prediction)
            confidence_score = round(torch.max(probs, dim=1)[0].item(), 4)

            # Store predictions and targets for metrics
            accuracy_list.extend(torch.argmax(pred, dim=1).tolist())
            accuracyt_list.extend(torch.argmax(label, dim=1).tolist())
            f1_list.extend(torch.argmax(pred, dim=1).tolist())
            f1t_list.extend(torch.argmax(label, dim=1).tolist())

            # Accumulate total loss
            total_loss += loss.item()
            tq.update(1)

    tq.close()

    # Compute and print final metrics
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))
    print("Accuracy: ", accuracy(torch.tensor(accuracy_list), torch.tensor(accuracyt_list)))
    print("Confidence score: ", confidence_score)

    # Log results to TensorBoard
    writer.add_scalar("Validation mIoU", f1(torch.tensor(f1_list), torch.tensor(f1t_list)), epoch)
    writer.add_scalar("Validation Loss", total_loss / len(data_test), epoch)

    return None

# Main block to run test evaluation
if __name__ == '__main__':
    # Load test dataset using custom loader
    test_data = Dataset(test_df, 256)

    test_loader = DataLoader(
        test_data,
        batch_size=1,     # Batch size of 1 for fine-grained evaluation
        shuffle=False,    # Don't shuffle test data
    )

    # Initialize models and move to GPU
    model1 = ResNetModel(36).cuda()   # ResNet version
    model2 = vgg16(36).cuda()         # VGG16 version (used below)

    # Define optimizer (Adam here, but can try SGD)
    # optimizer = SGD(model2.parameters(), lr=0.002)  # Uncomment to try SGD
    optimizer = Adam(model2.parameters(), lr=0.0001, weight_decay=1e-5)

    # Load the trained model checkpoint
    checkpoint = torch.load("checkpoints/Resnet18_model.pth")
    model2.load_state_dict(checkpoint['state_dict'])   # Load model weights
    optimizer.load_state_dict(checkpoint['optimizer']) # Load optimizer state

    # Define loss function
    loss = nn.CrossEntropyLoss()

    # TensorBoard writer for logging
    writer = SummaryWriter()

    # Run testing
    test(model2, test_loader, loss, writer, 1)  # Using VGG16 model here
    print('Finished Testing: Resnet Adam (lr=0.0001)')  # Status message
