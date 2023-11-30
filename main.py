#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install -q torch transformers numpy pandas sentence-transformers -U scikit-learn')


# In[2]:


get_ipython().system('jupyter nbextension enable --py --sys-prefix widgetsnbextension')


# In[3]:


import os
import json
import pandas as pd
from typing import List


# In[4]:


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split


# In[5]:


PARENT_FOLDER = "PAN2020-authorship-verification"
DATASET1_TRAIN = "pan20-authorship-verification-training-small/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"
DATASET2_TRAIN = "pan20-authorship-verification-training-small/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
# DATASET1_TRAIN = "pan20-authorship-verification-training-small-truth.jsonl"
# DATASET2_TRAIN = "pan20-authorship-verification-training-small.jsonl"
FILE_PATH_1 = f'./{PARENT_FOLDER}/{DATASET1_TRAIN}'
FILE_PATH_2 = f'./{PARENT_FOLDER}/{DATASET2_TRAIN}'


# In[6]:


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


# In[7]:


df_ground_truth = get_dataframe_from_file(FILE_PATH_1)
df_inputs = get_dataframe_from_file(FILE_PATH_2)

df_combined = pd.merge(df_ground_truth, df_inputs, on='id')

######################
#     CUIDADO!!!!!   #
######################
df_combined = df_combined.head(32)
print(len(df_combined))


# In[8]:


df_ground_truth.head()


# In[9]:


len(df_ground_truth)


# In[10]:


def check_not_nulls(df: pd.DataFrame) -> None:
    print(df.isnull().sum())


# In[11]:


def count_duplicate_ids(df: pd.DataFrame) -> pd.Series:
    # Find duplicate IDs
    duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]

    # Calculate the sum of repetitions
    sum_repetitions = len(duplicate_ids)

    return sum_repetitions


# In[12]:


check_not_nulls(df_ground_truth)


# In[13]:


check_not_nulls(df_inputs)


# Only on training data

# ## Generate Dataset
# 
# - Robust dataset: Separate pairs and with its fandoms. Use fandoms to generate new dataset of pairs.

# In[14]:


assert count_duplicate_ids(df_ground_truth) == count_duplicate_ids(df_inputs)


# In[15]:


# assert len(df_combined) - len(df_inputs) == 22


# Se elimina la columna "same" ya que no da información relevante para el entrenamiento del modelo. Debido a que es una comparación entre dos ids que son las salidas del modelo.

# In[16]:


df_combined = df_combined.drop("authors", axis=1).drop("fandoms", axis=1)


# Rename "authors" to "y"

# In[17]:


df_combined = df_combined.rename(columns={'same': 'y'})


# In[18]:


df_combined.head()


# In[19]:


df_combined.iloc[0]


# In[20]:


df_combined[['text1', 'text2']] = df_combined['pair'].apply(pd.Series)
df_combined = df_combined.drop("pair", axis=1)


# In[21]:


df_combined.head()


# In[22]:


df_combined.iloc[1, 1]


# In[23]:


mean_length = 0
for i in range(len(df_combined)):
    mean_length += len(df_combined.iloc[i, 2]) + len(df_combined.iloc[i, 3])

mean_length /= len(df_combined) * 2
mean_length = int(mean_length)
mean_length


# In[24]:


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


# # Model

# In[25]:


# transformer without woth pairs
class TransformerModel(nn.Module):
    def __init__(self, model_name, freeze_transformer):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        self.dense1 = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.4)
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
        sem_sim = sem_sim.view(sem_sim.size(0), -1)
        weighted_sem_sim = self.dense(sem_sim)

        return self.sigmoid(weighted_sem_sim)


# ## Test mio para comprobar que funciona y corre el modelo

# In[26]:


model_name = 'AnnaWegmann/Style-Embedding' # 'bert-base-uncased'  # Choose the appropriate pretrained model #'AnnaWegmann/Style-Embedding'


# In[27]:


train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_df, model_name, max_len=mean_length)
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# In[28]:


print(train_df.index)


# Small test to see that everything works

# In[29]:


# anna weinman style embeddings - hard negative mininng
model = TransformerModel(model_name=model_name, freeze_transformer=True)
model.train() # tell model we are going to train -> https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

for batch in train_data_loader:
    x = model.forward(batch["encoded_input_text1"], batch["encoded_input_text2"])
    print(x)
    break


# # Training model
# 
# See diapos a partir de la 152 y usar anotación de la diapos (ejemplo: bs_sl -> Batch size - Sequence Length)

# In[30]:


model = TransformerModel(model_name=model_name, freeze_transformer=True)

# Define your loss function (customize based on your task)
criterion = nn.MSELoss()  # Example: Mean Squared Error

# Define optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split your data into training and validation sets
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            encoded_input_text1 = batch['encoded_input_text1']
            encoded_input_text2 = batch['encoded_input_text2']
            targets = batch['targets']

            y_pred = model.forward(encoded_input_text1, encoded_input_text2)

            if targets.dim() == 1:
                targets = targets.view(-1, 1)

            # Calculate loss
            loss = criterion(y_pred, targets)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = (y_pred > 0.5).float()  # Assuming a binary classification task
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(data_loader)

    print(f"predictions (real): {y_pred}")
    print(f"predictions: {predictions}")
    print(f"ground_truth: {targets}")

    return average_loss, accuracy


def training_step(encoded_input_text1, encoded_input_text2, targets, model, optimizer, criterion):
    # !!!! necessary to set the model to training mode before
    
    # forward pass
    y_pred = model.forward(encoded_input_text1, encoded_input_text2)
    
    if targets.dim() == 1:
        targets = targets.view(-1, 1)

    loss = criterion(y_pred, targets)
    
    # baccpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


# Training loop
num_epochs = 10
batch_size = 16

train_dataset = CustomDataset(train_df, model_name)
validate_dataset = CustomDataset(val_df, model_name)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

i = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_data_loader):
        input_text1 = batch['encoded_input_text1']
        input_text2 = batch['encoded_input_text2']
        targets = batch['targets']

        loss = training_step(input_text1, input_text2, targets, model, optimizer, criterion)
        running_loss += loss

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Step [{i + 1}/{len(train_data_loader)}], "
                  f"Loss: {running_loss / 100}")
            running_loss = 0.0

    # Save the model weights after each epoch
    checkpoint_path = f"model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model weights saved to {checkpoint_path}")

    # Evaluate the model on the validation set after each epoch
    val_loss, val_accuracy = evaluate(model, val_data_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss}, Accuracy: {val_accuracy}')

print('Finished Training')


# QUEDA:
# 
# 1. El validate-test serían pasar en un bucle el forward del otro zip y capturar resultados para conseguir las métricas
# 2. Fine tunning (mínimo)
# 3. Escribir cosas
# 
# PD: Quitar el print de los shape

# In[34]:


FOLDER = f"{PARENT_FOLDER}/pan20-authorship-verification-test/pan20-authorship-verification-test"
VALUES_FILE = "pan20-authorship-verification-test.jsonl"
GROUND_TRUTH = "pan20-authorship-verification-test-truth.jsonl"


# In[35]:


df_ground_truth = get_dataframe_from_file(f"{FOLDER}/{GROUND_TRUTH}")
df_inputs = get_dataframe_from_file(f"{FOLDER}/{VALUES_FILE}")

df_combined_val = pd.merge(df_ground_truth, df_inputs, on='id')


# In[38]:


df_combined_val = df_combined.drop("authors", axis=1).drop("fandoms", axis=1)
df_combined_val = df_combined_val.rename(columns={'same': 'y'})


# In[39]:


df_combined_val.head()


# In[40]:


df_combined_val.iloc[0]


# In[41]:


df_combined_val[['text1', 'text2']] = df_combined_val['pair'].apply(pd.Series)
df_combined_val = df_combined_val.drop("pair", axis=1)


# In[43]:


model.load_state_dict(torch.load("model_epoch_10.pt"))
model.eval()

correct = 0
total = 0

test_dataset = CustomDataset(train_df, model_name)
validate_dataset = CustomDataset(val_df, model_name)

test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    for i, batch in enumerate(test_data_loader):
        input_text1 = batch['encoded_input_text1']
        input_text2 = batch['encoded_input_text2']
        targets = batch['targets']

        outputs = model.forward(input_text1, input_text2)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {100 * accuracy:.2f}%')


# In[ ]:




