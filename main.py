import os
import json
import pandas as pd
from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split


PARENT_FOLDER = "PAN2020-authorship-verification"
DATASET1_TRAIN = "pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
DATASET2_TRAIN = "pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
FILE_PATH_1 = f'{PARENT_FOLDER}/{DATASET1_TRAIN}'
FILE_PATH_2 = f'{PARENT_FOLDER}/{DATASET2_TRAIN}'


def get_dataframe_from_file (file_path : str) -> List:
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                parsed_data = json.loads(line)
                data.append(parsed_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

    return pd.DataFrame(data)


def check_not_nulls(df: pd.DataFrame) -> None:
    print(df.isnull().sum())


def count_duplicate_ids(df: pd.DataFrame) -> pd.Series:
    # Find duplicate IDs
    duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]

    # Calculate the sum of repetitions
    sum_repetitions = len(duplicate_ids)

    return sum_repetitions





df_ground_truth = get_dataframe_from_file(FILE_PATH_1)
df_inputs = get_dataframe_from_file(FILE_PATH_2)

df_combined = pd.merge(df_ground_truth, df_inputs, on='id')


print(df_ground_truth.head())


print(f'Hay: {len(df_ground_truth)} datos en df_ground truth')
print(check_not_nulls(df_ground_truth))
print(f'nulos en df_inputs? {check_not_nulls(df_inputs)}')

      
assert count_duplicate_ids(df_ground_truth) == count_duplicate_ids(df_inputs)
assert len(df_combined) - len(df_inputs) == 22


df_combined = df_combined.drop("authors", axis=1).drop("fandoms", axis=1)
df_combined = df_combined.rename(columns={'same': 'y'})
df_combined.head()

df_combined[['text1', 'text2']] = df_combined['pair'].apply(pd.Series)
df_combined = df_combined.drop("pair", axis=1)


print(df_combined.head)

##############################################
#                   MODEL                    #
##############################################
class CustomDataset(Dataset):
    def __init__(self, df, model_name, max_len=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = df
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        encoded_input_text1 = self.tokenizer(self.data.iloc[index, 2], max_length=512, padding=True, truncation=True, return_tensors='pt')
        encoded_input_text2 = self.tokenizer(self.data.iloc[index, 3], max_length=512, padding=True, truncation=True, return_tensors='pt')

        return {
            "encoded_input_text1": encoded_input_text1,
            "encoded_input_text2": encoded_input_text2,
            "targets": torch.tensor(int(self.data.iloc[index, 1]), dtype=torch.float)
        }
    

# transformer without woth pairs
class TransformerModel(nn.Module):
    def __init__(self, model_name):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dense1 = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.1)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.dense = nn.Linear(1, 1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input_text1, encoded_input_text2):
        model_output_text1 = self.transformer(
            input_ids=encoded_input_text1['input_ids'][0, :, :],
            attention_mask=encoded_input_text1['attention_mask'],
        ).last_hidden_state[:, 0]
        model_output_text2 = self.transformer(
            input_ids=encoded_input_text2['input_ids'][0, :, :],
            attention_mask=encoded_input_text2['attention_mask'],
        ).last_hidden_state[:, 0]

        x_a, x_b = self.dense1(model_output_text1), self.dense1(model_output_text2)
        x_a, x_b = self.gelu(self.dropout(x_a)), self.gelu(self.dropout(x_b))
        sem_sim = self.cosine(x_a, x_b)
        weighted_sem_sim = self.dense(sem_sim)

        return self.sigmoid(weighted_sem_sim)
    

##############################################
#          TEST DE BLITTY                    #
#   PARA COMPROBAR QUE FUNCIONA EL MODELO    #
##############################################

mean_length = 0
for i in range(len(df_combined)):
    mean_length += len(df_combined.iloc[i, 2]) + len(df_combined.iloc[i, 3])

mean_length /= len(df_combined) * 2
mean_length = int(mean_length)
mean_length

model_name = 'AnnaWegmann/Style-Embedding' # 'bert-base-uncased'  # Choose the appropriate pretrained model
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_df, model_name, max_len=mean_length)
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# anna weinman style embeddings - hard negative mininng
model = TransformerModel(model_name=model_name)
model.train() # tell model we are going to train -> https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

for batch in train_data_loader:
    x = model.forward(batch["encoded_input_text1"], batch["encoded_input_text2"])
    print(x)
    break



# Define your loss function (customize based on your task)
criterion = nn.MSELoss()  # Example: Mean Squared Error

# Define optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split your data into training and validation sets
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)

# Define a function to compute accuracy or other evaluation metrics
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            pairs = batch['pairs']
            fandoms = batch['fandoms']
            y = batch['y']

            # Forward pass
            y_pred = model(pairs, fandoms)

            # Calculate loss (customize based on your task)
            loss = criterion(y_pred, y)

            total_loss += loss.item()

    return total_loss / len(data_loader)

# Training loop
num_epochs = 10
batch_size = 32

train_dataset = CustomDataset(train_df, model_name)
validate_dataset = CustomDataset(val_df, model_name)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Create a DataLoader for training and validation data
    # You'll need to customize this part based on your dataset and preprocessing
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    
    for batch in train_data_loader:
        # print(batch)
        # print(type(batch))
        
        input_text1 = batch['encoded_input_text1']        # pairs
        input_text2 = batch['encoded_input_text2']        # fandoms
        targets = batch['targets']                        # y
        
        print("------------")
        print(f'Batch keys: {batch.keys()}')
        print("------------")
        print(f'Batch belongs to type: {type(batch)}')
        print("------------")
        print(input_text1)
        print(input_text2)
        print(targets)
        print("------------")
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        print("empiezo forward pass")
        # Forward pass
        y_pred = model.forward(input_text1, input_text2)
        print("termino forward pass")

        # Calculate loss
        loss = criterion(y_pred, targets)
        print("he calculado la loss")

        # Backpropagation and optimization
        print("empiezo backpropagation")
        loss.backward()
        optimizer.step()
        print("termine!!!!!!!")
        running_loss += loss.item()

    # Evaluate the model on the validation set
    val_loss = evaluate(model, val_data_loader)

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_data_loader)}, Val Loss: {val_loss}')

print('Finished Training')



