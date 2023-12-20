# Code adapted from the following tutorials:
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb 

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.vocab import GloVe
from tqdm import tqdm
import pandas as pd

# hyperparameters
size_vocab = None
embed_dim = 300
hidden_dim = 128
output_dim = 1
learning_rate = 0.001
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# define rnn model
class RNNModel(nn.Module):
    def __init__(self, size_vocab, embed_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(size_vocab, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)


# train rnn model
def train_model(train_data, TEXT):
    # initialize model, loss function, and optimizer
    model = RNNModel(size_vocab, embed_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_iterator = data.Iterator(train_data, batch_size=30, device=device)

    # train model for each epoch
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_iterator, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)

        # pbar to visualize loss for each epoch
        for batch in pbar:
            text, labels = batch.text.to(device), batch.Sentiment.float().to(device)
            text = torch.where(text < size_vocab, text, torch.tensor(TEXT.vocab.stoi[TEXT.unk_token], device=device, dtype=torch.long))
            optimizer.zero_grad()
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            pbar.set_postfix({'Loss': total_loss / len(train_data)}) 
        
        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}')

    # save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'models/non_proprocessed_checkpoint.pth')


# evaluate the tained model on test data
def evaluate(model, test_data, TEXT):
    test_iterator = data.Iterator(test_data, batch_size=30, device=device, train=False, sort=False)
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_iterator:
            text, labels = batch.text.to(device), batch.Sentiment.to(device)
            text = torch.where(text < size_vocab, text, torch.tensor(TEXT.vocab.stoi[TEXT.unk_token], device=device, dtype=torch.long))
            predictions = model(text).squeeze(1)

            # clamp predictions between -1 and 1 
            predictions = torch.clamp(predictions, min=-1, max=1)

            # discretize the continuous results
            sentiment = torch.where(predictions > 0.05, 1, torch.where((predictions >= -0.05) & (predictions <= 0.05), 0, -1))
            correct_mask = torch.eq(sentiment, labels)
            total_correct += correct_mask.sum().item()
            total_samples += len(labels)

    # print accuracy
    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


# apply model to unlabeled dataset
def label_dataset(model, dataset, csv, TEXT):
    predicted_sentiments = []

    with torch.no_grad():
        for example in dataset.examples:
            text = example.text
            predicted_sentiment = predict(model, text, TEXT)
            predicted_sentiments.append(predicted_sentiment)

    # Add the predicted sentiments to the dataset
    dataset.fields['Sentiment'] = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    for example, sentiment in zip(dataset.examples, predicted_sentiments):
        example.Sentiment = sentiment

    # create new dataframe
    original_dataframe = pd.DataFrame({
        'text': [' '.join(example.text) for example in dataset.examples],
        'Num_Comments': [row['Num_Comments'] for index, row in csv.iterrows()],
        'Score': [row['Score'] for index, row in csv.iterrows()],
        'Gilded_Status': [row['Gilded_Status'] for index, row in csv.iterrows()],
    })

    # add new "Sentiment" row to dataframe
    original_dataframe['Sentiment'] = [example.Sentiment for example in dataset.examples]

    # save the new dataframe
    original_dataframe.to_csv('data/analysis_data.csv', index=False)


# predict label for given text
def predict(model, text, TEXT):
    numericalized_text = [TEXT.vocab.stoi[token] for token in text]
    input_tensor = torch.tensor(numericalized_text, dtype=torch.long).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        model.eval()
        prediction = model(input_tensor).squeeze(1)
        prediction = torch.clamp(prediction, min=-1, max=1)

    return prediction.item()


def main():
    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', batch_first=True)
    LABEL = data.LabelField(dtype=torch.float, use_vocab=False)

    data_fields = [('text', TEXT), ('Num_Comments', None), ('Score', None), ('Gilded_Status', None), ('Sentiment', LABEL)]

    # load training data
    train_data = data.TabularDataset(
        path='data/auto_labeled_dataset.csv',
        format='csv',
        fields=data_fields,
        skip_header=True
    )

    # load test data
    test_data = data.TabularDataset(
        path='data/hand_labeled_dataset.csv',
        format='csv',
        fields=data_fields,
        skip_header=True
    )

    # load unlabled dataset for analysis
    unlabeled_data = data.TabularDataset(
        path='data/unlabeled_dataset.csv',
        format='csv',
        fields=data_fields,
        skip_header=True
    )

    # load unlabeled dataset for new csv creation
    csv = pd.read_csv('data/unlabeled_dataset.csv')

    # build vocabulary
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300), unk_init=torch.Tensor.mean)
    LABEL.build_vocab(train_data)

    # set global variable
    global size_vocab 
    size_vocab = len(TEXT.vocab)
    
    # train model
    train_model(train_data, TEXT)

    # load model for evaluation and data labeling
    model = RNNModel(size_vocab, embed_dim, hidden_dim, output_dim).to(device)
    checkpoint = torch.load('models/non_preprocessed_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # evaluate trained model
    evaluate(model, test_data, TEXT)

    # label unlabeled dataset with trained model
    label_dataset(model, unlabeled_data, csv, TEXT)


if __name__ == "__main__":
    main()