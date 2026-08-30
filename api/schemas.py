from pydantic import BaseModel, Field


class ChurnFeatures(BaseModel):
    tenure_months: float = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: float = Field(..., ge=0)
    num_products: float = Field(..., ge=0)
    contract_score: float
    usage_score: float
    satisfaction_score: float

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months": 12,
                "monthly_charges": 89.5,
                "total_charges": 1074.0,
                "support_tickets": 4,
                "num_products": 2,
                "contract_score": -0.5,
                "usage_score": 1.2,
                "satisfaction_score": -1.1,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_version: str
