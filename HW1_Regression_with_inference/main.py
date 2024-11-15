from fastapi import FastAPI
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