import torch
import torchvision
# import os
import gc
# import numpy as np
import pandas as pd
import torch.nn as nn
from os import system,listdir
import pandas as pd
from dataset import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision
from network import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"


# Config Dictionary
config = {
    "epochs" : 2,
    "batch_size" : 50,
    "write_text" : "Wheat-Growth-Stage\n\n\n\n",
    "clear_logs" : True,
    "lr" : 1e-3,
    "weight_decay" : 1e-3,
    "T_0" : 20,
    "T_mult" : 2,
    "eta_min" : 1e-11,

}


# Data Path
path = "Images/"


# Train Method
def train(model,dataloader,optimizer,criterion):
    model.train()
    correct,total_loss = 0,0
    
    for i,data in enumerate(dataloader):

        # For testing commenting below lines

        optimizer.zero_grad()
        images,labels = data
        images,labels = images.to(DEVICE),labels.to(DEVICE)

        output = model(images)
        loss = criterion(output,labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        correct += int((torch.argmax(output,axis = 1) == labels).detach().cpu().numpy().sum())

    cur_loss = total_loss / len(dataloader)
    acc = correct / (len(dataloader)*config["batch_size"])
    return cur_loss,100*acc  


# Validation Method
def val(model,dataloader,criterion):
    model.eval()

    correct,total_loss = 0,0
    for i,data in enumerate(dataloader):
        images,labels = data
        images,labels = images.to(DEVICE),labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs,labels)
        correct += ((torch.argmax(outputs,axis = 1) == labels).detach().cpu().numpy().sum())
        total_loss += loss.item()

    cur_loss = total_loss/len(dataloader)
    acc = correct / (config["batch_size"]*len(dataloader))
    return cur_loss,100*acc


# Test Method
def test(model,dataloader):
    predictions = [[],[]]
    for i,data in enumerate(dataloader):
        images,ids = data
        images = images.to(DEVICE)

        outputs = model(images)
        predicted_vals = torch.argmax(outputs,axis = 1)
        
        predictions[0].extend(ids)
        predictions[1].extend(predicted_vals.tolist())
    return predictions


# Write Submission Method
def write_submissions(prediction_lists,datetime):
    cols = ["UID","growth_stage"]
    sub = {
        cols[0] : prediction_lists[0],
        cols[1] : prediction_lists[1],
    }
    prediction_df = pd.DataFrame(data=sub,columns=cols)
    prediction_df["growth_stage"] = prediction_df["growth_stage"] + 1
    prediction_df.to_csv("submissions/submission_{}.csv".format(datetime),index=False)
    return None

# Write Logs Method
def write_logs(val_list,fname):
    with open(fname,'a') as f:
        write_str = "Epoch: {}, Train L: {:.4f}, Train A: {:.3f}, Val L: {:.4f}, Val A: {:.3f}, lr: {:.12f}\n".format(*val_list)
        f.write(write_str)
    return None


# Get datetime string
def get_datetimestr():
    from datetime import datetime
    cur_datetime = datetime.now()
    cur_datetime = datetime.strftime(cur_datetime,"%m_%d_%Y_%H_%M_%S")
    return cur_datetime


if __name__ == "__main__":
    # Clear Logs
    if config["clear_logs"]:
        system("sh scripts/clear_logs.sh")

    # Define transforms
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
    ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
    ])
    test_transforms = val_transforms

    # Load train,val data and initialize data loader
    train_df = pd.read_csv("files/Train.csv")
    train_df["growth_stage"] = train_df["growth_stage"] - 1
    X_train,X_test,y_train,y_test = train_test_split(train_df["UID"],train_df["growth_stage"],test_size=0.3)
    X_train,y_train = X_train.values.tolist(),y_train.values.tolist()
    train_dataset = LoadDataSet(fids = X_train,flabels = y_train,path = path,transforms=train_transforms)
    X_test,y_test = X_test.values.tolist(),y_test.values.tolist()
    val_dataset = LoadDataSet(fids= X_test,flabels= y_test,path= path,transforms= val_transforms)
    train_dataloader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False)

    #Load test data and intialize dataloader object
    test_df = pd.read_csv("files/SampleSubmission.csv")
    test_flist = test_df["UID"].values.tolist()
    test_dataset = TestDataset(fids = test_flist,path = path,transforms=test_transforms)
    test_dataloader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle = False)

    # Initialize model object
    model = ResNet()
    model.to(DEVICE)

    print("Train Batch Size:",len(train_dataloader))
    print("Val Batch Size:",len(val_dataloader))
    print("Test Batch Size:",len(test_dataloader))
    # Define loss fun, optimizer, scheduler,etc.
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0 = config["T_0"],T_mult=config["T_mult"],eta_min=config["eta_min"])
    
    # Get datetime string
    date_time = get_datetimestr()
    logs_fname = "logs/logs_{}.txt".format(date_time)
    with open(logs_fname,'w') as f:
        f.write(config["write_text"])
    
    # Log dataframe
    cols = ["epoch","train_loss","train_acc","val_loss","val_acc","lr"]
    logs_df = pd.DataFrame(columns=cols)

    # Loop over dataset for epochs.
    best_acc = 0.1
    for epoch in range(config["epochs"]):
        gc.collect()
        torch.cuda.empty_cache()

        train_loss, train_acc = train(model=model,dataloader=train_dataloader,optimizer=optimizer,criterion=loss_fn)

        val_loss,val_acc = val(model=model,dataloader=val_dataloader,criterion=loss_fn)

        lr = float(optimizer.param_groups[0]["lr"])
        log_vals = [epoch, train_loss,train_acc,val_loss,val_acc,lr]

        logs_df.loc[len(logs_df)] = log_vals
        write_logs(log_vals,logs_fname)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict" : model.state_dict(),
                "val_acc" : val_acc,
                "optimizer_state_dict" : optimizer.state_dict(),
            },"checkpoints/checkpoint_{:.2f}_{}.pth".format(val_acc,date_time))
    # Write submission file
    predictions = test(model = model,dataloader = test_dataloader)
    write_submissions(predictions,date_time)




