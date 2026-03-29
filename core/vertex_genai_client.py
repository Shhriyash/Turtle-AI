import os
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

def get_vertex_model(model_name: str):
    """
    Initialize and return a Gemini model using Vertex AI.

    Requires GOOGLE_CLOUD_PROJECT; optionally GOOGLE_CLOUD_LOCATION (defaults to "global").
    """
    provider = GoogleProvider(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )
    return GoogleModel(model_name=model_name, provider=provider)

