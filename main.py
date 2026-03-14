import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from src.code_assistant.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
