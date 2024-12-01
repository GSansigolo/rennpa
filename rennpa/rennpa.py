import csv
import random
import pandas as pd
import requests
import urllib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

def shuffle_data (input, output):
    # Open the input CSV file for reading
    with open(input, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Read all rows except the header
        rows = list(reader)[1:]
        
        # Shuffle the rows
        random.shuffle(rows)
        
        # Open the output CSV file for writing
        with open(output, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            # Write the header row to the output file
            writer.writerow(next(csv.reader(open(input))))
            # Write the shuffled rows to the output file
            writer.writerows(rows)

    print("CSV file shuffle successfully!")

def get_timeseries(cube, geom, cloud_filter=None):

    url_wtss = 'https://data.inpe.br/bdc/wtss/v4'

    cloud_dict = {
        'S2-16D-2':{
            'cloud_band': 'SCL',
            'non_cloud_values': [4,5,6],
            'cloud_values': [0,1,2,3,7,8,9,10,11]
        }
    }

    for point in geom:
        tss = {}
        for band in cube['bands']:
            query = dict(
                coverage=cube['collection'],
                attributes=band,
                start_date=cube['start_date'],
                end_date=cube['end_date'],
                latitude=point['coordinates'][1],
                longitude=point['coordinates'][0],
            )
            url_suffix = '/time_series?'+urllib.parse.urlencode(query)
            #print(url_wtss + url_suffix)
            data = requests.get(url_wtss + url_suffix) 
            data_json = data.json()
            if data.status_code:
                try:
                    tss[band] = data_json['result']['attributes'][0]['values']
                    timeline = data_json['result']['timeline']
                except:
                    tss[band] = []
                    timeline = []
            else:
                tss[band] = []
                timeline = []

            if cloud_filter:
                if data.status_code:
                    cloud = cloud_dict[cube['collection']]
                    cloud_query = dict(
                        coverage=cube['collection'],
                        attributes=cloud['cloud_band'],
                        start_date=cube['start_date'],
                        end_date=cube['end_date'],
                        latitude=point['coordinates'][1],
                        longitude=point['coordinates'][0],
                    )
                    cloud_url_suffix = '/time_series?'+urllib.parse.urlencode(cloud_query)
                    #print(url_wtss + cloud_url_suffix)
                    cloud_data = requests.get(url_wtss + cloud_url_suffix) 
                    cloud_data_json = cloud_data.json()
                    try:
                        cloud_array = create_filter_array(cloud_data_json['result']['attributes'][0]['values'], cloud['cloud_values'], cloud['non_cloud_values'])
                        ts = data_json['result']['attributes'][0]['values']
                        for i in range(len(ts)):
                            if cloud_array[i] == 0:
                                ts[i] = -9999
                    except:
                        cloud_array = []
                        ts = []
                        timeline = []
                else:
                    ts = []
                    timeline = []
    
    return dict(values=tss, timeline=timeline)

def create_filter_array(array, filter_true, filter_false):
    filter_arr = []
    for element in array:
        if element in filter_true:
            filter_arr.append(0)
        if element in filter_false:
            filter_arr.append(1)
    return filter_arr

def rennpa_get_timeseries(cube, input, output=None):

    S2_cube = cube

    df = pd.read_csv(input)
    timeseries = []
    df = df.reset_index()  # make sure indexes pair with number of rows

    for row in tqdm(df.itertuples(index=True, name='Pandas'), total=len(df)):
        #print(row.longitude, row.latitude)
        ts = get_timeseries(
            cube=S2_cube, 
            geom=[dict(coordinates = [row.longitude, row.latitude])],
            cloud_filter = False
        )

        timeseries.append(ts['values'])

    if output:
        with open(output, 'w') as fp:
            json.dump(dict(timeseries = timeseries), fp)
        
    print("Timeseries successfully fetched!")

    return timeseries

def rennpa_create_indicies (formula, bands, timeseries):
    
    results = []

    for row in timeseries:

        if formula == "OSAVI":  #OSAVI = (NIR – VERMELHO) / (NIR + VERMELHO + 0.16)

            results.append(np.divide((np.subtract(row[bands["NIR"]],row[bands["RED"]])),(np.add(np.add(row[bands["NIR"]],row[bands["RED"]]),[0.16]*23))))

        if formula == "NDWI": #NDWI = (VERDE – NIR) / (VERDE + NIR)
            
            results.append(np.divide((np.subtract(row[bands["GREEN"]],row[bands["NIR"]])),(np.add(row[bands["GREEN"]],row[bands["NIR"]]))))
            
        if formula == "RECI": #ReCI = (NIR / VERMELHO) – 1

            results.append(np.divide(row[bands["NIR"]],(np.add(row[bands["RED"]],[1]*23))))

    return results

def rennpa_dataframe(timeseries, input_csv, indicies, timeseries_file=None):

    if timeseries_file:
        with open(timeseries_file) as f:
            j = json.load(f)
            f.close()
            X_raw = json["timeseries"]
    else:
        X_raw = timeseries

    B02 = []
    B03 = []
    B04 = []
    B08 = []
    NDVI = []
    if ("OSAVI" in indicies):
        OSAVI = []
    if ("NDWI" in indicies):
        NDWI = []
    if ("RECI" in indicies):
        RECI = []
        
    for i,row in enumerate(X_raw): 
        B02.append(row["B02"])
        B03.append(row["B03"])
        B04.append(row["B04"])
        B08.append(row["B08"])
        NDVI.append(row["NDVI"])  
        if ("OSAVI" in indicies):
            osavi = indicies["OSAVI"][i]
            OSAVI.append(osavi.tolist())
        if ("NDWI" in indicies):
            ndwi = indicies["NDWI"][i]
            NDWI.append(ndwi.tolist())
        if ("RECI" in indicies):
            reci = indicies["RECI"][i]
            RECI.append(reci.tolist())

    if ("OSAVI" in indicies and "NDWI" in indicies and "RECI" in indicies):
        df = pd.DataFrame({"B02":B02,"B03":B03,"B04":B04,"B08":B08,"NDVI":NDVI,"OSAVI":OSAVI,"NDWI":NDWI, "RECI": RECI})
    else:
        df = pd.DataFrame({"B02":B02,"B03":B03,"B04":B04,"B08":B08,"NDVI":NDVI})
        
    labels_df = pd.read_csv(input_csv)

    df["label"] = labels_df["label_id"]

    return df

def rennpa_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cuda_id = torch.cuda.current_device()
        print(torch.cuda.get_device_name(cuda_id))
    return device

class TimeseriesDataset():
    
    def __init__(self,df):
        self.df = df
        
    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self,item):
        item_B02 = self.df.loc[item,"B02"]
        item_B03 = self.df.loc[item,"B03"]
        item_B04 = self.df.loc[item,"B04"]
        item_B08 = self.df.loc[item,"B08"]
        item_NDVI = self.df.loc[item,"NDVI"]
        item_OSAVI = self.df.loc[item,"OSAVI"]
        item_NDWI = self.df.loc[item,"NDWI"]
        item_RECI = self.df.loc[item,"RECI"]
        item_label = self.df.loc[item,"label"]
                               
        X_train_tensor = torch.tensor([item_B02,item_B03,item_B04,item_B08,item_NDVI,item_OSAVI,item_NDWI,item_RECI], dtype=torch.float32, device=rennpa_gpu())
        y_train_tensor = torch.tensor(item_label, dtype=torch.long, device=rennpa_gpu()) 

        
        return (X_train_tensor, y_train_tensor)
    
def rennpa_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)

def rennpa_split(dataset, split):
    train = dataset.head(int(len(dataset)*split))
    val = dataset.tail(int(len(dataset)*(1-split))).reset_index(drop=True)
    return train, val

class rennpa_lstm(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(rennpa_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(rennpa_gpu())
        self.fc = nn.Linear(hidden_size, output_size).to(rennpa_gpu())
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(rennpa_gpu())
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(rennpa_gpu())
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1])
    
def rennpa_crossentropyloss():
    return nn.CrossEntropyLoss()

def rennpa_adam(model, lr):
    return optim.Adam(model,lr=lr)

def rennpa_train(train_ds, val_ds, model, num_epochs, optimizer, criterion, print_epochs=True):

    datasets = {
        'train': train_ds,
        'validation': val_ds
    }

    dataloaders = {
        'train': rennpa_dataloader(datasets['train'], batch_size=256, shuffle=False),
        'validation': rennpa_dataloader(datasets['validation'], batch_size=256, shuffle=False)
    }

    for epoch in tqdm(range(num_epochs)):  
        
        epoch_loss = []
        # Iterate over data.
            
        for x,y in tqdm(dataloaders['train'],total=len(dataloaders['train'])):
            
            # zero the parameter gradients
            optimizer.zero_grad()   #resetting the optimizer

            # forward
            outputs = model(x).squeeze()
            loss = criterion(outputs, y) # evaluating loss
            
            # backward + optimize only if in training phase
            
            loss.backward()  # calculating gradients
            optimizer.step()  # using optimizer to recalculate parameters
            epoch_loss.append(loss.item())

        if print_epochs:
            print(f"Epoch {epoch}, Loss:",np.array(epoch_loss).mean())

    return model

def rennpa_validade(model, sample):
    with torch.no_grad():
        prediction = torch.max(torch.sigmoid(model(sample[0].unsqueeze(dim=0))), -1)
        prediction = prediction.indices.item()
        print("Predicted Label:", prediction)

def rennpa_accuracy(model, val_ds):
    predictions_list = []
    correct_list = []
    correct_count = 0

    for s in tqdm(range(len(val_ds))): 
        prediction = torch.max(torch.sigmoid(model(val_ds[s][0].unsqueeze(dim=0))), -1)
        prediction = prediction.indices.item()
        predictions_list.append(prediction)
        correct_list.append(val_ds[s][1].item())
        if prediction == val_ds[s][1].item():
            correct_count+=1
    acc_sc = (correct_count)/len(val_ds)*100
    print('Accuracy is %s' % acc_sc)

def rennpa_open_ts_json(input):
    with open(input) as f:
        content = json.load(f)
        f.close()

    return content["timeseries"]

def rennpa_confusion_matrix(model, val_ds):
    predictions_list = []
    correct_list = []
    correct_count = 0

    for s in tqdm(range(len(val_ds))): 
        prediction = torch.max(torch.sigmoid(model(val_ds[s][0].unsqueeze(dim=0))), -1)
        prediction = prediction.indices.item()
        predictions_list.append(prediction)
        correct_list.append(val_ds[s][1].item())
        if prediction == val_ds[s][1].item():
            correct_count+=1

    confusion_matrix = metrics.confusion_matrix(correct_list, predictions_list)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = range(0,9))

    cm_display.plot()
    #plt.xticks(rotation=90)
    plt.show() 