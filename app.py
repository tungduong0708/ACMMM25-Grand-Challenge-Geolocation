from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from g3.g3_simple_prediction import G3Predictor

base_path = Path(__file__).parent


class PredictRequest(BaseModel):
    image: Annotated[
        str,
        Field(
            description="Base64-encoded input image.",
        ),
    ]


class PredictResponse(BaseModel):
    lat: Annotated[
        float,
        Field(
            ge=-90.0,
            le=90.0,
            description="Predicted latitude of the image, in degree.",
        ),
    ]
    lon: Annotated[
        float,
        Field(
            ge=-180.0,
            le=180.0,
            description="Predicted longitude of the image, in degree.",
        ),
    ]


predictor: G3Predictor


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor

    checkpoint_path = (
        base_path / "g3/checkpoints/mercator_finetune_weight.pth"
    ).resolve()
    index_path = (base_path / "g3/index/G3.index").resolve()

    predictor = G3Predictor(
        checkpoint_path=checkpoint_path.as_posix(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        index_path=index_path.as_posix(),
    )

    yield

    del predictor


app = FastAPI(
    lifespan=lifespan,
    title="G3",
    description="An endpoint to predict GPS coordinate from static image,"
    " using G3 Framework.",
)


@app.post("/g3/predict")
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    gps = predictor.predict(
        base64_image=request.image,
        database_csv_path=(base_path / "g3/data/mp16/MP16_Pro_filtered.csv")
        .resolve()
        .as_posix(),
    )
    return PredictResponse(lat=gps[0], lon=gps[1])
