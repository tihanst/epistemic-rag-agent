from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PG_HOST: str  # = Field(..., env="PG_HOST")
    PG_DATABASE: str  # = Field(..., env="PG_DATABASE")
    PG_PORT: str  # = Field(..., env="PG_PORT")
    EMBEDDING_MODEL: str  # = Field(..., env="EMBEDDING_MODEL")
    DENSE_TOP_K: int  # = Field(..., env="DENSE_TOP_K")
    SPARSE_TOP_K: int  # = Field(..., env="SPARSE_TOP_K")
    RRF_TOP_K: int  # = Field(..., env="RRF_TOP_K")
    CROSS_ENCODER_MODEL: str  # = Field(..., env="CROSS_ENCODER_MODEL")
    CROSS_ENCODER_TOP_K: int  # = Field(..., env="CROSS_ENCODER_TOP_K")
    LLM_API_KEY: str
    LLM_ENDPOINT: str
    GRADIO_SERVER_NAME: str
    GRADIO_SERVER_PORT: int

    class Config:
        env_file = ".env"


if __name__ == "__main__":
    from pprint import pprint

    settings = Settings()
    pprint(settings.model_dump())
