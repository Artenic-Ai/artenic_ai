"""Run the Artenic AI Platform: ``uv run python -m artenic_ai_platform``."""  # pragma: no cover

from __future__ import annotations  # pragma: no cover

import uvicorn  # pragma: no cover

from artenic_ai_platform.app import create_app  # pragma: no cover
from artenic_ai_platform.settings import PlatformSettings  # pragma: no cover


def main() -> None:  # pragma: no cover
    """Entry-point for the platform server."""
    settings = PlatformSettings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    main()
