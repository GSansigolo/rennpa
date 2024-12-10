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
from torcheval.metrics.functional import multiclass_f1_score
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import Point, within
import pyproj
from geopandas import GeoSeries
import xarray as xr

BDC_PROJ = 'PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown based on GRS80 ellipsoid",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",-12],PARAMETER["longitude_of_center",-54],PARAMETER["standard_parallel_1",-2],PARAMETER["standard_parallel_2",-22],PARAMETER["false_easting",5000000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

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
    EVI = []
    NBR = []
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
        EVI.append(row["EVI"])
        NBR.append(row["NBR"])
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
        df = pd.DataFrame({"B02":B02,"B03":B03,"B04":B04,"B08":B08,"NDVI":NDVI,"EVI":EVI,"NBR":NBR,"OSAVI":OSAVI,"NDWI":NDWI, "RECI": RECI})
    else:
        df = pd.DataFrame({"B02":B02,"B03":B03,"B04":B04,"B08":B08,"NDVI":NDVI,"EVI":EVI,"NBR":NBR})
        
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
        item_EVI = self.df.loc[item,"EVI"]
        item_NBR = self.df.loc[item,"NBR"]
        item_OSAVI = self.df.loc[item,"OSAVI"]
        item_NDWI = self.df.loc[item,"NDWI"]
        item_RECI = self.df.loc[item,"RECI"]
        item_label = self.df.loc[item,"label"]
                               
        X_train_tensor = torch.tensor([item_B02,item_B03,item_B04,item_B08,item_NDVI,item_EVI, item_NBR,item_OSAVI,item_NDWI,item_RECI], dtype=torch.float32, device=rennpa_gpu())
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

def rennpa_accuracy(model, val_ds, num_classes):
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
    input = torch.tensor(predictions_list)
    target = torch.tensor(correct_list)
    f1_score = multiclass_f1_score(input, target, num_classes=num_classes)
    print('F1 Score is %s' % f1_score)

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

def rennpa_get_pixels_from_polygon(p):
    minx, miny, maxx, maxy = p.bounds
    x1, y1 = pyproj.transform(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_wkt(BDC_PROJ), minx, miny, always_xy=True)
    x2, y2 = pyproj.transform(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_wkt(BDC_PROJ), maxx, maxy, always_xy=True)
    nx = int((x2-x1)/10)
    ny = int((y2-y1)/10)
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    meshgrid = np.meshgrid(x, y)
    list_x = []
    list_y = []
    for col in meshgrid[0]:
        for item in col:
            list_x.append(item)
    for col in meshgrid[1]:
        for item in col:
            list_y.append(item)

    pts = [Point(X,Y) for X,Y in zip(list_x, list_y)]
    points = [pt for pt in pts if within(pt, p)]
    return np.array([[pt.x,pt.y] for pt in points])

def rennpa_get_pixels_from_bounds(p):
    minx, miny, maxx, maxy = p.bounds
    x1, y1 = pyproj.transform(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_wkt(BDC_PROJ), minx, miny, always_xy=True)
    x2, y2 = pyproj.transform(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_wkt(BDC_PROJ), maxx, maxy, always_xy=True)
    nx = int((x2-x1)/10)
    ny = int((y2-y1)/10)
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    meshgrid = np.meshgrid(x, y)
    list_x = []
    list_y = []
    for col in meshgrid[0]:
        for item in col:
            list_x.append(item)
    for col in meshgrid[1]:
        for item in col:
            list_y.append(item)

    return np.array([(x,y) for x,y in zip(list_x, list_y)])

def rennpa_plot_pts_region(polygon, points):
    x, y = [],[]
    for p in points:
        x.append(p[0])
        y.append(p[1])
    gpd.GeoSeries(polygon).plot(color='red', alpha=0.25)
    plt.scatter(x,y) 
    plt.show() 

def get_ts(cube, geom, attribute):

    url_wtss = 'https://data.inpe.br/bdc/wtss/v4'

    for point in geom:
        query = dict(
            coverage=cube['collection'],
            attributes=attribute,
            start_date=cube['start_date'],
            end_date=cube['end_date'],
            latitude=point['coordinates'][1],
            longitude=point['coordinates'][0],
        )
        url_suffix = '/time_series?'+urllib.parse.urlencode(query)
        data = requests.get(url_wtss + url_suffix) 
        data_json = data.json()
        if data.status_code:
            try:
                ts = data_json['result']['attributes'][0]['values']
                timeline = data_json['result']['timeline']
            except:
                ts = []
                timeline = []
        else:
            ts = []
            timeline = []
        return dict(values=ts, timeline=timeline)
    
def rennpa_classify(cube, points, region, model):

    GEOM = []
    B02 = []
    B03 = []
    B04 = []
    B08 = []
    NDVI = []
    EVI = []
    NBR = []
    NDVI = []
    OSAVI = []
    NDWI = []
    RECI = []
    label = []

    for input in tqdm(points):
        attributes = ["B02", "B03", "B04", "B08", "EVI", "NBR", "NDVI"]
        tss = {}

        for attribute in attributes:

            # Get time series 
            ts = get_ts(
                cube=cube, 
                geom=[dict(coordinates = input)],
                attribute=attribute
            )

            tss[attribute] = ts['values']
            
        GEOM.append(Point(input))
        B02.append(tss["B02"])
        B03.append(tss["B03"])
        B04.append(tss["B04"])
        B08.append(tss["B08"])
        NDVI.append(tss["NDVI"])  
        EVI.append(tss["EVI"])
        NBR.append(tss["NBR"])
        osavi = np.divide((np.subtract(tss["B08"],tss["B04"])),(np.add(np.add(tss["B08"],tss["B04"]),[0.16]*23)))
        OSAVI.append(osavi.tolist())
        ndwi = np.divide((np.subtract(tss["B03"],tss["B08"])),(np.add(tss["B03"],tss["B08"])))
        NDWI.append(ndwi.tolist())
        reci = np.divide(tss["B08"],(np.add(tss["B04"],[1]*23)))
        RECI.append(reci.tolist())
        label.append(0)
        
    gleba_df = pd.DataFrame({"GEOM":GEOM,"B02":B02,"B03":B03,"B04":B04,"B08":B08,"NDVI":NDVI,"EVI":EVI,"NBR":NBR,"OSAVI":OSAVI,"NDWI":NDWI,"RECI": RECI,"label": label})
    
    gleba_ds = TimeseriesDataset(gleba_df) 

    geoms = []
    labels_id = []
    labels_titles = []

    for s in tqdm(range(len(gleba_ds))): 
        prediction = torch.max(torch.sigmoid(model(gleba_ds[s][0].unsqueeze(dim=0))), -1)
        prediction = prediction.indices.item()
        geoms.append(str(gleba_df.loc[s,"GEOM"]))
        labels_id.append(prediction)

    for p in tqdm(region):
        wkt = 'POINT ('+str(p[0])+' '+str(p[1])+')'
        if(wkt not in geoms):
            geoms.append(wkt)
            labels_id.append(-9999)
            labels_titles.append("no data")

    result_df = pd.DataFrame({"geometry":geoms,"label_id":labels_id})

    points = gpd.GeoSeries.from_wkt(result_df.geometry)
    classification = gpd.GeoDataFrame(data=result_df, geometry=points, crs={"init": "epsg:4326"})
    
    classification['label_id'] = [9 if i ==-9999 else i for i in classification['label_id']]
    classification['x'] = classification['geometry'].x
    classification['y'] = classification['geometry'].y
    classification = classification.rename(columns={'label_id': 'band_data'})
    classification = classification.drop("geometry", axis=1)
    classification = classification.set_index(['x', 'y'])
    xr_classification = classification.to_xarray()  
    xr_classification = xr_classification.assign_coords(time='2023')
    xr_classification = xr_classification.expand_dims('time')
    xr_classification = xr_classification.assign_coords(band=1)
    xr_classification = xr_classification.expand_dims('band')

    return xr_classification

def rennpa_plot_classification(xr_classification):

    cube_classification = xr_classification['band_data']

    cube_classification.plot(cmap="tab20")

def rennpa_save(model, output):

    torch.save(model.state_dict(), output)