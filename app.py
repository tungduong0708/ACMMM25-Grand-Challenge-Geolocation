import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.g3_batch_prediction import G3BatchPredictor


class EvidenceResponse(BaseModel):
    analysis: Annotated[
        str,
        Field(description="A supporting analysis for the prediction."),
    ]
    references: Annotated[
        list[str],
        Field(description="Links or base64-encoded JPEG supporting the analysis."),
    ] = []


class LocationPredictionResponse(BaseModel):
    latitude: Annotated[
        float,
        Field(description="Latitude of the predicted location, in degree."),
    ]
    longitude: Annotated[
        float,
        Field(description="Longitude of the predicted location, in degree."),
    ]
    location: Annotated[
        str,
        Field(description="Textual description of the predicted location."),
    ]
    evidence: Annotated[
        list[EvidenceResponse],
        Field(description="List of supporting analyses for the prediction."),
    ]


class PredictionResponse(BaseModel):
    prediction: Annotated[
        LocationPredictionResponse,
        Field(description="The location prediction and accompanying analysis."),
    ]
    transcript: Annotated[
        str | None,
        Field(description="The extracted and concatenated transcripts, if any."),
    ] = None


predictor: G3BatchPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("openapi.json", "wt") as api_file:
        json.dump(app.openapi(), api_file, indent=4)

    global predictor
    predictor = G3BatchPredictor(device="cuda" if torch.cuda.is_available() else "cpu")

    yield

    del predictor


app = FastAPI(
    lifespan=lifespan,
    title="G3",
    description="An endpoint to predict GPS coordinate from static image,"
    " using G3 Framework.",
)


@app.post(
    "/g3/predict",
    description="Provide location prediction.",
)
async def predict_endpoint(
    files: Annotated[
        list[UploadFile],
        File(description="Input images, videos and metadata json."),
    ],
) -> PredictionResponse:
    # Write files to disk
    try:
        for file in files:
            filename = file.filename if file.filename is not None else uuid.uuid4().hex
            filepath = predictor.input_dir / filename
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )

    # Get prediction
    result = await predictor.predict(model_name="gemini-2.5-pro")
    response = predictor.get_response(result)
    prediction = LocationPredictionResponse(
        latitude=response.latitude,
        longitude=response.longitude,
        location=response.location,
        evidence=[
            EvidenceResponse(analysis=ev.analysis, references=ev.references)
            for ev in response.evidence
        ],
    )
    # Get transcript if available
    transcript = predictor.get_transcript()
    # Clear directories
    predictor.clear_directories()
    return PredictionResponse(prediction=prediction, transcript=transcript)


@app.get(
    "/g3/openapi",
    description="Provide the OpenAPI JSON describing this service's endpoints.",
)
async def openapi():
    return app.openapi()
