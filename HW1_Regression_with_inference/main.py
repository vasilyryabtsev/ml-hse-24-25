from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import model
import pandas as pd
import pickle

app = FastAPI()
# uvicorn main:app --reload --port 8000

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

class Item(BaseModel):
    '''
    Данные об автомобиле, выставленного на продажу
    '''
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
    
    def to_DataFrame(self) -> pd.DataFrame:
        '''
        Возвращает датасет со всеми полями класса
        '''
        return pd.DataFrame({
            'name': self.name,
            'year': self.year,
            'selling_price': self.selling_price,
            'km_driven': self.km_driven,
            'fuel': self.fuel,
            'seller_type': self.seller_type,
            'transmission': self.transmission,
            'owner': self.owner,
            'mileage': self.mileage,
            'engine': self.engine,
            'max_power': self.max_power,
            'torque': self.torque,
            'seats': self.seats}, index=[0])
        

class Items(BaseModel):
    '''
    Автомобили, выставленные на продажу
    '''
    objects: List[Item]
    

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    '''
    Возвращает ожидаемую стоимость автомобиля
    в формате float
    '''
    data = item.to_DataFrame()
    return float(loaded_model.predict(data))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    '''
    Возвращает список ожидаемых стоимостей для
    автомобилей
    '''
    res = items[0].to_DataFrame()
    
    for i, item in enumerate(items):
        if i == 0:
            continue
        cur_item = item.to_DataFrame()
        res = pd.concat([res, cur_item])
    res =  res.reset_index(drop=True)
    
    return list(loaded_model.predict(res))

@app.post("/predict_csv")
def predict(file: UploadFile = File(...)) -> FileResponse:
    data = pd.read_csv(file.file)
    data['prediction'] = loaded_model.predict(data)
    data.to_csv('data_with_prediction.csv')
    return FileResponse(path='data_with_prediction.csv', filename='data_with_prediction.csv')