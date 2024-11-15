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

# obj = df_train.sample(1)
# obj_dict = obj.to_dict('index')
# obj_dict[obj.index[0]]

class Items(BaseModel):
    objects: List[Item]
    

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = item.to_DataFrame()
    return float(loaded_model.predict(data))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return ...