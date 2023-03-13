
from pydantic import BaseModel, validator, Field


CUT_LIST = [ "Fair", "Good", "Ideal", "Signature-Ideal", "Very Good", ]
COLOR_LIST = [ "D", "E", "F", "G", "H", "I"]
CLARITY_LIST = [ 'FL', 'IF', 'SI1', 'VS1', 'VS2', 'VVS1', 'VVS2', ]
POLISH_LIST = [ 'EX', 'G', 'ID', 'VG',  ]
SYMMETRY_LIST = [ 'EX', 'G', 'ID', 'VG',  ]
REPORT_LIST = [ 'AGSL', 'GIA', ]

def validate_v_in_list(var, v, lst): 
    if v not in lst:            
        raise ValueError(f'Variable {var} must have one of these values:  {lst}')



class DataModel(BaseModel): 
    Id: str
    Carat_Weight: float = Field(alias='Carat Weight', ge=0.75, le=3.0)
    Cut: str
    Color: str
    Clarity: str
    Polish: str
    Symmetry: str
    Report: str
    
    @validator("Cut")
    def Cut_must_be_one_of(cls, v): 
        if v is None: return v
        validate_v_in_list("Cut", v, CUT_LIST)   
        return v     
    
    @validator("Color")
    def Color_must_be_one_of(cls, v): 
        if v is None: return v
        validate_v_in_list("Color", v, COLOR_LIST)   
        return v          
    
    @validator("Clarity")
    def Clarity_must_be_one_of(cls, v): 
        if v is None: return v
        validate_v_in_list("Clarity", v, CLARITY_LIST)  
        return v           
    
    @validator("Polish")
    def Polish_must_be_one_of(cls, v): 
        if v is None: return v
        validate_v_in_list("Polish", v, POLISH_LIST)  
        return v        
    
    @validator("Symmetry")
    def Symmetry_must_be_one_of(cls, v): 
        if v is None: return v
        validate_v_in_list("Symmetry", v, SYMMETRY_LIST) 
        return v       
    
    @validator("Report")
    def Report_must_be_one_of(cls, v): 
        if v is None: return v
        validate_v_in_list("Report", v, REPORT_LIST)
        return v     