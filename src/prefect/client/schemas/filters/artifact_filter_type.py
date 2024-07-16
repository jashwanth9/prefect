from typing import List, Optional

from pydantic import Field

from prefect._internal.schemas.bases import PrefectBaseModel


class ArtifactFilterType(PrefectBaseModel):
    """Filter by `Artifact.type`."""

    any_: Optional[List[str]] = Field(
        default=None, description="A list of artifact types to include"
    )
    not_any_: Optional[List[str]] = Field(
        default=None, description="A list of artifact types to exclude"
    )