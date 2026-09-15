import uvicorn

from src.shared.config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("src.api.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
